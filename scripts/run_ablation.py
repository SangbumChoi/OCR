#!/usr/bin/env python3
"""Run one ablation ARM end-to-end for the Part-2 fine-tuning bases and record before/after.

Pipeline (GPU): generate the arm's synthetic data -> LoRA fine-tune -> evaluate on the probe suite
-> append to docs/results/ablation_results.json under models[<model>][<arm>]. The visualization
notebook (notebooks/finetune_ablation.ipynb) reads that file to draw the side-by-side before/after.

Progress is logged as "[done/total] (N left) stage" so it is clear how much remains.

    # baseline (no training) for both bases, then the A1 spotting arm:
    python scripts/run_ablation.py --models qwen3_5-0.8b lfm2_5-vl-1.6b --arm baseline
    python scripts/run_ablation.py --models qwen3_5-0.8b lfm2_5-vl-1.6b --arm A1_spotting_on \
        --placement connector --steps 200
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "docs" / "results" / "ablation_results.json"
# Single base (compute): qwen3_5-0.8b. (LFM2.5-VL dropped for our compute budget.)
HF_ID = {"qwen3_5-0.8b": "Qwen/Qwen3.5-0.8B"}
PROBES = {"capability": "data/probes/capability_probe/capability.jsonl",
          "spatial": "data/probes/spatial_context_probe/probe.jsonl",
          "realistic": "data/probes/realistic_cases/realistic_cases.jsonl"}
# Score EVERY arm on the whole suite to see cross-capability transfer (does spotting help CER/KIE?
# does multilingual hurt EN?), not just the arm's target axis.
EVAL_SUITE = ["capability", "spatial", "realistic"]


def _record(model: str, arm: str, payload: dict) -> None:
    """Merge one arm's per-probe summaries into ablation_results.json (preserving other keys)."""
    doc = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    doc.setdefault("models", {}).setdefault(model, {})[arm] = payload
    RESULTS.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(HF_ID))
    p.add_argument("--arm", required=True,
                   help="'baseline' (eval only) or a configs/synth_data.yaml ablation id "
                        "(e.g. A1_spotting_on, A2_reasoning_on, A4_ko_en, A7_dynamic_tiling)")
    p.add_argument("--placement", default="all", help="A5 LoRA group: vision|connector|llm_attn|llm_mlp|all")
    # CONTROL FACTOR — held fixed across arms so a delta is attributable to the factor alone:
    p.add_argument("--count", type=int, default=50,
                   help="variants per case = the fixed #images/iterations control (keep equal across arms)")
    p.add_argument("--steps", type=int, default=300, help="fixed max training steps (the iteration control)")
    p.add_argument("--heldout-seed", type=int, default=None,
                   help="A0 memorization test: also eval on a realistic set generated with THIS seed "
                        "(different from training) -> unseen content; reports the train/held-out gap")
    args = p.parse_args()

    from docvlm_eval.finetune.lora_vlm import LoraVLMConfig, eval_vlm, train_lora_vlm

    heldout_jsonl = None
    if args.heldout_seed is not None:    # build a held-out TEST split with a different seed
        test_dir = ROOT / "data" / "probes" / "_realistic_heldout"
        subprocess.run([sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                        "--seed", str(args.heldout_seed), "--count", str(max(1, args.count // 5)),
                        "--out", str(test_dir)], cwd=ROOT, check=True)
        heldout_jsonl = str(test_dir / "realistic_cases.jsonl")
        from docvlm_eval.synth import load_realistic_samples
        from docvlm_eval.benchmarks import save_jsonl
        save_jsonl(load_realistic_samples(test_dir), heldout_jsonl)

    def eval_all(hf, adapter=None):
        """Score the whole suite -> {probe: summary} (cross-capability transfer), + held-out if set."""
        out = {pb: eval_vlm(hf, str(ROOT / PROBES[pb]), adapter_path=adapter) for pb in EVAL_SUITE}
        if heldout_jsonl:
            out["heldout"] = eval_vlm(hf, heldout_jsonl, adapter_path=adapter)
        return out

    total = len(args.models); done = 0
    for model in args.models:
        done += 1
        hf = HF_ID[model]
        print(f"[{done}/{total}] ({total-done} left) {model} :: arm={args.arm} placement={args.placement} "
              f"count={args.count} steps={args.steps}")
        if args.arm == "baseline":
            payload = {"control": {"count": args.count, "steps": args.steps},
                       "probes": eval_all(hf)}
            _record(model, "baseline", payload)
        else:
            # 1) data for this arm (count = the fixed #images control)
            subprocess.run([sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                            "--ablation", args.arm, "--count", str(args.count)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "scripts/build_realistic_benchmark.py"], cwd=ROOT, check=True)
            train_jsonl = str(ROOT / PROBES["realistic"])
            # 2) LoRA fine-tune (steps = the fixed iteration control)
            out = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                max_steps=args.steps, output_dir=f"outputs/{model}/{args.arm}_{args.placement}"))
            # 3) evaluate the adapted model on the WHOLE suite (cross-capability transfer)
            payload = {"control": {"count": args.count, "steps": args.steps, "placement": args.placement},
                       "probes": eval_all(hf, adapter=out)}
            _record(model, f"{args.arm}:{args.placement}", payload)
        cap = payload["probes"].get("capability", {})
        print(f"    capability score={cap.get('score')}  by_axis={cap.get('by_answer_type')}")
    print(f"[done] -> {RESULTS}")


if __name__ == "__main__":
    main()

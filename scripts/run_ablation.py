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
HF_ID = {"qwen3_5-0.8b": "Qwen/Qwen3.5-0.8B", "lfm2_5-vl-1.6b": "LiquidAI/LFM2.5-VL-1.6B"}
PROBES = {"capability": "data/probes/capability_probe/capability.jsonl",
          "spatial": "data/probes/spatial_context_probe/probe.jsonl",
          "realistic": "data/probes/realistic_cases/realistic_cases.jsonl"}


def _record(model: str, arm: str, summary: dict) -> None:
    """Merge one arm's summary into ablation_results.json (preserving the demo/other keys)."""
    doc = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    doc.setdefault("models", {}).setdefault(model, {})[arm] = summary
    RESULTS.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(HF_ID))
    p.add_argument("--arm", required=True,
                   help="'baseline' (eval only) or a configs/synth_data.yaml ablation id "
                        "(e.g. A1_spotting_on, A2_reasoning_on, A4_ko_en, A7_dynamic_tiling)")
    p.add_argument("--placement", default="all", help="A5 LoRA group: vision|connector|llm_attn|llm_mlp|all")
    p.add_argument("--probe", default="capability", choices=list(PROBES))
    p.add_argument("--count", type=int, default=20, help="variants per case for the training set")
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    from docvlm_eval.finetune.lora_vlm import LoraVLMConfig, eval_vlm, train_lora_vlm

    probe_jsonl = str(ROOT / PROBES[args.probe])
    total = len(args.models); done = 0
    for model in args.models:
        done += 1
        hf = HF_ID[model]
        print(f"[{done}/{total}] ({total-done} left) {model} :: arm={args.arm} placement={args.placement}")
        if args.arm == "baseline":
            summary = eval_vlm(hf, probe_jsonl)
        else:
            # 1) data for this arm
            subprocess.run([sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                            "--ablation", args.arm, "--count", str(args.count)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "scripts/build_realistic_benchmark.py"], cwd=ROOT, check=True)
            train_jsonl = str(ROOT / PROBES["realistic"])
            # 2) LoRA fine-tune on the arm's data
            out = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                max_steps=args.steps, output_dir=f"outputs/{model}/{args.arm}_{args.placement}"))
            # 3) evaluate the adapted model on the probe suite
            summary = eval_vlm(hf, probe_jsonl, adapter_path=out)
        _record(model, f"{args.arm}:{args.placement}" if args.arm != "baseline" else "baseline", summary)
        print(f"    score={summary.get('score')}  by_axis={summary.get('by_answer_type')}")
    print(f"[done] -> {RESULTS}")


if __name__ == "__main__":
    main()

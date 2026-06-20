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


def _gen_realistic(out_dir: Path, seed: int, count: int) -> str:
    """Generate a realistic_cases set into out_dir and build its benchmark jsonl; return the jsonl."""
    subprocess.run([sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                    "--seed", str(seed), "--count", str(count), "--out", str(out_dir)],
                   cwd=ROOT, check=True)
    from docvlm_eval.benchmarks import save_jsonl
    from docvlm_eval.synth import load_realistic_samples
    jsonl = str(out_dir / "realistic_cases.jsonl")
    save_jsonl(load_realistic_samples(out_dir), jsonl)
    return jsonl


def _n_samples(jsonl: str) -> int:
    return sum(1 for ln in Path(jsonl).read_text().splitlines() if ln.strip())


def run_a0(args, eval_vlm, train_lora_vlm, LoraVLMConfig) -> None:
    """A0 PREREQUISITE — memorization vs understanding as a function of training-data SIZE.

    For each size N (variants/case): train LoRA for a FIXED #epochs on seed=7 data, then score BOTH
    (a) the TRAIN set it just fit (memorization signal) and (b) a FIXED held-out set on a DIFFERENT
    seed (understanding/generalization signal). A big train≫held-out gap that grows as N shrinks =
    memorization; the held-out curve flattening = the recommended synthetic size. Result lands in
    ablation_results.json -> models[<model>]["A0"], read by the notebook's prerequisite section."""
    # held-out TEST built ONCE so every training size is scored on the identical unseen set.
    test_jsonl = _gen_realistic(ROOT / "data" / "probes" / "_a0_heldout",
                                seed=args.a0_test_seed, count=args.a0_test_count)
    n_test = _n_samples(test_jsonl)
    sizes = sorted(set(args.a0_sizes))
    total = len(args.models) * len(sizes); done = 0
    for model in args.models:
        hf = HF_ID[model]
        rec = {"epochs": args.a0_epochs, "test_seed": args.a0_test_seed,
               "n_test_samples": n_test, "sizes": {}}
        for n in sizes:
            done += 1
            print(f"[{done}/{total}] ({total-done} left) {model} :: A0 size={n} (x14 cases) "
                  f"epochs={args.a0_epochs}")
            train_jsonl = _gen_realistic(ROOT / "data" / "probes" / "realistic_cases", seed=7, count=n)
            n_train = _n_samples(train_jsonl)
            # per-epoch eval on BOTH train (memorization) and held-out (understanding) -> W&B curves
            _, last = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                epochs=args.a0_epochs, max_steps=None,
                output_dir=f"outputs/{model}/A0_n{n}",
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run_prefix}A0-{model}-n{n}"),
                eval_specs=[("train", train_jsonl), ("heldout", test_jsonl)])
            train_s, held_s = last["train"], last["heldout"]   # final-epoch eval (no model reload)
            ts, hs = train_s.get("score"), held_s.get("score")
            gap = (ts - hs) if (ts is not None and hs is not None) else None
            rec["sizes"][str(n)] = {"variants_per_case": n, "n_train_samples": n_train,
                                    "train": train_s, "heldout": held_s, "gap": gap}
            print(f"    train={ts}  heldout={hs}  gap(train-heldout)={gap}")
            _record(model, "A0", rec)   # checkpoint after every size
    print(f"[done] A0 -> {RESULTS}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(HF_ID))
    p.add_argument("--arm", required=True,
                   help="'baseline' (eval only), 'A0' (memorization-vs-size prerequisite sweep), or a "
                        "configs/synth_data.yaml ablation id (e.g. A1_spotting_on, A4_ko_en)")
    p.add_argument("--placement", default="all", help="A5 LoRA group: vision|connector|llm_attn|llm_mlp|all")
    # CONTROL FACTOR — held fixed across arms so a delta is attributable to the factor alone:
    p.add_argument("--count", type=int, default=50,
                   help="variants per case = the fixed #images/iterations control (keep equal across arms)")
    p.add_argument("--steps", type=int, default=300, help="fixed max training steps (the iteration control)")
    p.add_argument("--heldout-seed", type=int, default=None,
                   help="A0 memorization test: also eval on a realistic set generated with THIS seed "
                        "(different from training) -> unseen content; reports the train/held-out gap")
    # --- A0 (PREREQUISITE) memorization-vs-understanding size sweep (configs/ablations.yaml A0) ---
    p.add_argument("--a0-sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800],
                   help="A0: train-data scale sweep = variants/case (x14 cases = #images). Span a wide "
                        "range so the held-out plateau is visible; the config's full curve adds 3200.")
    p.add_argument("--a0-epochs", type=int, default=3,
                   help="A0: epochs per size (FIXED across sizes so each example is seen equally -> the "
                        "only thing changing is dataset size). Train-to-fit reveals memorization.")
    p.add_argument("--a0-test-seed", type=int, default=999, help="A0: held-out TEST seed (never trained on)")
    p.add_argument("--a0-test-count", type=int, default=5,
                   help="A0: held-out TEST scale (variants/case). FIXED across sizes so the test set is "
                        "identical for every training size -> comparable held-out curve.")
    # --- Weights & Biases (optional): log per-epoch loss + train/held-out eval metrics ---
    p.add_argument("--wandb-project", default=None,
                   help="W&B project to log per-epoch loss + eval metrics to (needs `wandb login`). "
                        "Omit to train without logging.")
    p.add_argument("--wandb-run-prefix", default="",
                   help="prefix for the W&B run name (run = <prefix><arm>-<model>[-n<size>])")
    args = p.parse_args()

    # Fail fast with one actionable line instead of a deep 'model type qwen3_5 not recognized' trace.
    import transformers
    if int(transformers.__version__.split(".")[0]) < 5:
        sys.exit(f"[run_ablation] needs transformers>=5 for Qwen3.5-VL (got {transformers.__version__}). "
                 f"Run: pip install -U 'transformers>=5'  (and restart the kernel if it was imported).")

    from docvlm_eval.finetune.lora_vlm import LoraVLMConfig, eval_vlm, train_lora_vlm

    if args.arm == "A0":
        run_a0(args, eval_vlm, train_lora_vlm, LoraVLMConfig)
        return

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
            # 2) LoRA fine-tune (steps = the fixed iteration control); per-epoch loss/train-score -> W&B
            out, _ = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                max_steps=args.steps, output_dir=f"outputs/{model}/{args.arm}_{args.placement}",
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run_prefix}{args.arm}-{model}-{args.placement}"),
                eval_specs=[("train", train_jsonl)])
            # 3) evaluate the adapted model on the WHOLE suite (cross-capability transfer)
            payload = {"control": {"count": args.count, "steps": args.steps, "placement": args.placement},
                       "probes": eval_all(hf, adapter=out)}
            _record(model, f"{args.arm}:{args.placement}", payload)
        cap = payload["probes"].get("capability", {})
        print(f"    capability score={cap.get('score')}  by_axis={cap.get('by_answer_type')}")
    print(f"[done] -> {RESULTS}")


if __name__ == "__main__":
    main()

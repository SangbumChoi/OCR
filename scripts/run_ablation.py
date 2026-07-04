#!/usr/bin/env python3
"""Run one ablation ARM end-to-end for the Part-2 fine-tuning bases and record before/after.

Pipeline (GPU): generate the arm's synthetic data -> LoRA fine-tune -> evaluate on the probe suite
-> append to docs/results/ablation_results.json under models[<model>][<arm>]. The visualization
notebook (notebooks/finetune_ablation.ipynb) reads that file to draw the side-by-side before/after.

Progress is logged as "[done/total] (N left) stage" so it is clear how much remains.

    # default base is LFM2.5-VL (fast on a T4); pass --models qwen3_5-0.8b to use Qwen instead.
    python scripts/run_ablation.py --arm A0                                   # memorization sweep (LFM)
    python scripts/run_ablation.py --arm A1_spotting_on --placement connector --steps 200
    python scripts/run_ablation.py --models qwen3_5-0.8b --arm baseline       # opt back into Qwen
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
# Part-2 bases (select with --models). LFM2.5-VL is ~10x faster to fine-tune than Qwen3.5-VL on a T4
# (Qwen's full-attention prefill ~0.05 it/s vs LFM's hybrid-conv) — see notebooks/latency_profile.ipynb.
HF_ID = {"qwen3_5-0.8b": "Qwen/Qwen3.5-0.8B",
         "lfm2_5-vl-1.6b": "LiquidAI/LFM2.5-VL-1.6B",
         # tiny base for --smoke wiring proofs: small enough that a few LoRA steps + a 16-sample
         # eval complete on CPU (NOT a measurement base)
         "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct"}
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


def _subsample_jsonl(src: str, n: int, out_path: Path, seed: int = 7) -> tuple[str, int]:
    """Deterministically subsample ``n`` rows from a jsonl (for the public-data A0 scale sweep)."""
    import random as _random
    rows = [ln for ln in Path(src).read_text().splitlines() if ln.strip()]
    _random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(out_path), len(rows)


def run_a0(args, eval_vlm, train_lora_vlm, LoraVLMConfig) -> None:
    """A0 PREREQUISITE — memorization vs understanding as a function of training-data SIZE.

    For each size N (variants/case): train LoRA for a FIXED #epochs on seed=7 data, then score BOTH
    (a) the TRAIN set it just fit (memorization signal) and (b) a FIXED held-out set on a DIFFERENT
    seed (understanding/generalization signal). A big train≫held-out gap that grows as N shrinks =
    memorization; the held-out curve flattening = the recommended synthetic size. Result lands in
    ablation_results.json -> models[<model>]["A0"], read by the notebook's prerequisite section."""
    # Validation/held-out set is ALWAYS synthetic (built ONCE so every size scores the identical set).
    # In public-data mode (--train-jsonl) this means: TRAIN on public benchmark data, VALIDATE on
    # synthetic — the train-vs-synthetic gap then also reflects domain shift, not memorization alone.
    public = getattr(args, "train_jsonl", None)
    test_jsonl = _gen_realistic(ROOT / "data" / "probes" / "_a0_heldout",
                                seed=args.a0_test_seed, count=args.a0_test_count)
    n_test = _n_samples(test_jsonl)
    sizes = sorted(set(args.a0_sizes))
    if public:
        n_pub = _n_samples(public)
        sizes = [n for n in sizes if n <= n_pub] or [n_pub]   # can't subsample beyond available rows
        print(f"[A0 public] train source = {public} ({n_pub} rows); sizes={sizes}; "
              f"validation = synthetic (seed {args.a0_test_seed}, {n_test} samples)")
    total = len(args.models) * len(sizes); done = 0
    for model in args.models:
        hf = HF_ID[model]
        rec = {"epochs": args.a0_epochs, "test_seed": args.a0_test_seed,
               "n_test_samples": n_test, "mode": "public" if public else "synthetic",
               "train_source": public or "synthetic(realistic_cases)", "sizes": {}}
        for n in sizes:
            done += 1
            unit = f"{n} images (public)" if public else f"{n} (x14 cases)"
            print(f"[{done}/{total}] ({total-done} left) {model} :: A0 size={unit} "
                  f"epochs={args.a0_epochs}")
            if public:
                train_jsonl, n_train = _subsample_jsonl(
                    public, n, ROOT / "data" / "probes" / "_public_a0" / f"train_n{n}.jsonl")
            else:
                train_jsonl = _gen_realistic(ROOT / "data" / "probes" / "realistic_cases", seed=7, count=n)
                n_train = _n_samples(train_jsonl)
            # per-epoch eval on BOTH train (memorization) and held-out (understanding) -> W&B curves
            _, last = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                epochs=args.a0_epochs, max_steps=None,
                output_dir=f"outputs/{model}/A0_n{n}",
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run_prefix}A0-{model}-n{n}",
                grad_checkpointing=not args.no_grad_ckpt, max_image_long_side=args._mils,
                batch_size=args.batch_size, eval_max_samples=args.eval_max_samples),
                eval_specs=[("train", train_jsonl), ("heldout", test_jsonl)])
            train_s, held_s = last["train"], last["heldout"]   # final-epoch eval (no model reload)
            ts, hs = train_s.get("score"), held_s.get("score")
            gap = (ts - hs) if (ts is not None and hs is not None) else None
            rec["sizes"][str(n)] = {"variants_per_case": n, "n_images": n_train,
                                    "n_train_samples": n_train,
                                    "train": train_s, "heldout": held_s, "gap": gap}
            print(f"    train={ts}  heldout={hs}  gap(train-heldout)={gap}")
            _record(model, "A0", rec)   # checkpoint after every size
    print(f"[done] A0 -> {RESULTS}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["lfm2_5-vl-1.6b"],
                   help=f"Part-2 base(s) to run; default LFM2.5-VL (fast on a T4). Choices: {list(HF_ID)}")
    p.add_argument("--arm", required=True,
                   help="'baseline' (eval only), 'A0' (memorization-vs-size prerequisite sweep), or a "
                        "configs/synth_data.yaml ablation id (e.g. A1_spotting_on, A4_ko_en)")
    p.add_argument("--placement", default="all", help="A5 LoRA group: vision|connector|llm_attn|llm_mlp|all")
    # PUBLIC-DATA path: train on a prebuilt benchmark jsonl (scripts/build_benchmark_trainset.py) and
    # validate on the SYNTHETIC suite. Use --arm public for a single run, or --arm A0 for the scale sweep.
    p.add_argument("--train-jsonl", default=None,
                   help="train on THIS jsonl (e.g. data/benchmark_trainset/train.jsonl) instead of "
                        "generating synthetic data; validation stays synthetic. Supports --arm public/A0.")
    p.add_argument("--record-key", default=None,
                   help="key to store results under in ablation_results.json (default: <arm>:<placement>)")
    p.add_argument("--results", default=None,
                   help="results JSON path (default docs/results/ablation_results.json; the public-data "
                        "notebook uses a separate file so its A0 doesn't collide with the synthetic one)")
    # CONTROL FACTOR — held fixed across arms so a delta is attributable to the factor alone:
    p.add_argument("--count", type=int, default=50,
                   help="variants per case = the fixed #images/iterations control (keep equal across arms)")
    p.add_argument("--steps", type=int, default=300, help="fixed max training steps (the iteration control)")
    p.add_argument("--heldout-seed", type=int, default=None,
                   help="A0 memorization test: also eval on a realistic set generated with THIS seed "
                        "(different from training) -> unseen content; reports the train/held-out gap")
    p.add_argument("--heldout-jsonl", default=None,
                   help="use THIS prebuilt jsonl as the held-out eval set instead of generating a "
                        "synthetic one (e.g. data/udd_tasks/heldout_all.jsonl — the UDD public "
                        "heldout fold from build_task_trainsets.py)")
    p.add_argument("--before-after", action="store_true",
                   help="public arms: ALSO evaluate the UN-tuned base on the full suite before "
                        "training, record probes_before/probes_after + per-axis deltas, and save "
                        "per-sample predictions for both phases (outputs/<model>/<arm>/preds_"
                        "{before,after}/) — the cross-capability 'did the factor help OVERALL, "
                        "not just its own axis' comparison")
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
    # --- A6 HPO knobs (threaded into LoraVLMConfig; defaults = the config's defaults) ---
    p.add_argument("--lora-r", type=int, default=16, help="A6: LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="A6: LoRA alpha (convention: 2*r)")
    p.add_argument("--lr", type=float, default=1e-4, help="A6: learning rate")
    # --- throughput / OOM controls ---
    p.add_argument("--batch-size", type=int, default=2,
                   help="micro-batch (images/forward). LFM at 768px uses ~5GB on a T4 -> bs=2 fits; "
                        "raise to use more VRAM, lower to 1 if OOM.")
    p.add_argument("--max-image-long-side", type=int, default=768,
                   help="downscale each image's long side before the processor (caps vision tokens = "
                        "the main OOM/speed lever). Lower to 640 if OOM; 0 = native res.")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (faster but much more activation memory)")
    p.add_argument("--eval-max-samples", type=int, default=64,
                   help="cap eval samples per probe (fixed subsample). The arm regenerates "
                        "realistic_cases at --count -> thousands of samples; without this the suite "
                        "eval would score the whole TRAINING set. 0 = score all.")
    p.add_argument("--grounding-repeat", type=int, default=8,
                   help="A1 curriculum: repeat grounding rows this many times during training so "
                        "box targets are not diluted by ordinary QA/table rows.")
    p.add_argument("--grounding-target", choices=["pixel", "norm"], default="norm",
                   help="A1 curriculum target format. 'norm' teaches 0-1 boxes, which are scale-stable "
                        "and still accepted by the grounding metric.")
    args = p.parse_args()
    args._mils = args.max_image_long_side or None      # 0 -> None (native resolution)
    if args.results:
        global RESULTS
        RESULTS = Path(args.results)

    # Fail fast with one actionable line instead of a deep model-type import trace.
    # Only the 2025-26 bases (LFM2.5-VL / Qwen3.5) need transformers>=5; the smolvlm-256m smoke
    # base runs on v4, so don't block a wiring proof on the big-model requirement.
    import transformers
    needs_v5 = any(m in ("qwen3_5-0.8b", "lfm2_5-vl-1.6b") for m in args.models)
    if needs_v5 and int(transformers.__version__.split(".")[0]) < 5:
        sys.exit(f"[run_ablation] needs transformers>=5 for the selected 2025-26 VLM "
                 f"(got {transformers.__version__}). "
                 f"Run: pip install -U 'transformers>=5'  (and restart the kernel if it was imported).")

    from docvlm_eval.finetune.lora_vlm import LoraVLMConfig, eval_vlm, train_lora_vlm

    if args.arm == "A0":
        run_a0(args, eval_vlm, train_lora_vlm, LoraVLMConfig)
        return

    heldout_jsonl = args.heldout_jsonl   # prebuilt public heldout (UDD fold), if given
    if heldout_jsonl is None and args.heldout_seed is not None:    # else synth with a different seed
        test_dir = ROOT / "data" / "probes" / "_realistic_heldout"
        heldout_cmd = [sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                       "--seed", str(args.heldout_seed), "--count", str(max(1, args.count // 5)),
                       "--out", str(test_dir)]
        if args.arm not in ("baseline", "A0"):
            heldout_cmd += ["--ablation", args.arm]
        subprocess.run(heldout_cmd, cwd=ROOT, check=True)
        heldout_jsonl = str(test_dir / "realistic_cases.jsonl")
        from docvlm_eval.synth import load_realistic_samples
        from docvlm_eval.benchmarks import save_jsonl
        save_jsonl(load_realistic_samples(test_dir), heldout_jsonl)

    from docvlm_eval.finetune.lora_vlm import score_suite

    def eval_all(hf, adapter=None, save_preds_dir=None):
        """Score the whole suite -> {probe: summary} (cross-capability transfer), + held-out if set.
        Loads the model ONCE for all probes (not once per probe)."""
        jsonls = {pb: str(ROOT / PROBES[pb]) for pb in EVAL_SUITE}
        if heldout_jsonl:
            jsonls["heldout"] = heldout_jsonl
        return score_suite(hf, jsonls, adapter_path=adapter, max_image_long_side=args._mils,
                           max_samples=args.eval_max_samples or None,
                           save_preds_dir=save_preds_dir)

    def _deltas(before: dict, after: dict) -> dict:
        """Per-probe score delta + per-axis (answer_type/task) deltas — after minus before."""
        out = {}
        for pb, aft in after.items():
            bef = before.get(pb, {})
            d = {"score": (aft.get("score") or 0) - (bef.get("score") or 0)}
            b_ax, a_ax = bef.get("by_answer_type") or {}, aft.get("by_answer_type") or {}
            d["by_axis"] = {k: round((a_ax.get(k) or 0) - (b_ax.get(k) or 0), 4)
                            for k in set(b_ax) | set(a_ax)}
            out[pb] = d
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
        elif args.arm == "public":
            # PUBLIC-DATA single run: LoRA-train on the benchmark jsonl (optionally subsampled to
            # --count), validate on the SYNTHETIC suite. Feasible regardless of spotting/reasoning GT;
            # used for the A5 placement and A7 preprocessing sweeps in the public-dataset notebook.
            if not args.train_jsonl:
                sys.exit("[run_ablation] --arm public requires --train-jsonl <benchmark jsonl>")
            n_avail = _n_samples(args.train_jsonl)
            if args.count and args.count < n_avail:
                train_jsonl, n_train = _subsample_jsonl(
                    args.train_jsonl, args.count, ROOT / "data" / "probes" / "_public_train.jsonl")
            else:
                train_jsonl, n_train = args.train_jsonl, n_avail
            print(f"    [public] train on {n_train} rows from {args.train_jsonl}; validate on synthetic")
            arm_dir = ROOT / "outputs" / model / (args.record_key or f"public_{args.placement}").replace(":", "_")
            probes_before = None
            if args.before_after:      # BASE model on the full suite BEFORE any training
                print("    [before] scoring the un-tuned base on the full suite")
                probes_before = eval_all(hf, save_preds_dir=str(arm_dir / "preds_before"))
            out, _ = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                max_steps=args.steps, output_dir=f"outputs/{model}/public_{args.placement}_{args._mils}",
                lora_r=args.lora_r, lora_alpha=args.lora_alpha, learning_rate=args.lr,
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run_prefix}public-{model}-{args.placement}-img{args._mils}",
                grad_checkpointing=not args.no_grad_ckpt, max_image_long_side=args._mils,
                batch_size=args.batch_size, eval_max_samples=args.eval_max_samples),
                eval_specs=[("train", train_jsonl)] + ([("heldout", heldout_jsonl)] if heldout_jsonl else []))
            probes_after = eval_all(hf, adapter=out,
                                    save_preds_dir=str(arm_dir / "preds_after")
                                    if args.before_after else None)
            payload = {"control": {"count": n_train, "steps": args.steps, "placement": args.placement,
                                   "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lr": args.lr,
                                   "max_image_long_side": args._mils, "train_source": args.train_jsonl},
                       "probes": probes_after}
            if probes_before is not None:
                payload["probes_before"] = probes_before
                payload["delta"] = _deltas(probes_before, probes_after)
                payload["preds_dir"] = str(arm_dir)
                hd = payload["delta"].get("heldout", {})
                print(f"    [before/after] heldout Δscore={hd.get('score', 0):+.2f}  "
                      f"by-axis={hd.get('by_axis')}")
            _record(model, args.record_key or f"public:{args.placement}", payload)
        else:
            # 1) data for this arm (count = the fixed #images control)
            subprocess.run([sys.executable, "scripts/make_realistic_cases.py", "--no-degrade",
                            "--ablation", args.arm, "--count", str(args.count)], cwd=ROOT, check=True)
            subprocess.run([sys.executable, "scripts/build_realistic_benchmark.py"], cwd=ROOT, check=True)
            train_jsonl = str(ROOT / PROBES["realistic"])
            is_a1 = args.arm.startswith("A1_") or args.arm.startswith("A3_spot")
            eval_specs = [("train", train_jsonl)]
            if heldout_jsonl:
                eval_specs.append(("heldout", heldout_jsonl))
            # 2) LoRA fine-tune (steps = the fixed iteration control); per-epoch loss/train-score -> W&B
            out, _ = train_lora_vlm(LoraVLMConfig(
                model_id=hf, train_jsonl=train_jsonl, placement=args.placement,
                max_steps=args.steps, output_dir=f"outputs/{model}/{args.arm}_{args.placement}",
                wandb_project=args.wandb_project,
                wandb_run=f"{args.wandb_run_prefix}{args.arm}-{model}-{args.placement}",
                grad_checkpointing=not args.no_grad_ckpt, max_image_long_side=args._mils,
                batch_size=args.batch_size, eval_max_samples=args.eval_max_samples,
                grounding_repeat=args.grounding_repeat if is_a1 else 1,
                grounding_target=args.grounding_target if is_a1 else "pixel"),
                eval_specs=eval_specs)
            # 3) evaluate the adapted model on the WHOLE suite (cross-capability transfer)
            payload = {"control": {"count": args.count, "steps": args.steps, "placement": args.placement,
                                   "grounding_repeat": args.grounding_repeat if is_a1 else 1,
                                   "grounding_target": args.grounding_target if is_a1 else "pixel",
                                   "heldout_seed": args.heldout_seed},
                       "probes": eval_all(hf, adapter=out)}
            _record(model, f"{args.arm}:{args.placement}", payload)
        cap = payload["probes"].get("capability", {})
        print(f"    capability score={cap.get('score')}  by_axis={cap.get('by_answer_type')}")
    print(f"[done] -> {RESULTS}")


if __name__ == "__main__":
    main()

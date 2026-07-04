#!/usr/bin/env python3
"""Run the ablation arms on PUBLIC data (UDD) — the public-data realization of ablation_plan.md.

Each arm pair differs in exactly ONE factor at an EQUAL total sample count (the plan's control),
composed from the equal-N sets built by ``build_task_trainsets.py``:

* **A1 spotting**   — on: vqa+kie+grounding thirds; off: vqa+kie halves. Same total N, same steps;
  the only difference is whether *where* (bbox) targets are present.
* **A2 reasoning**  — chain: derived spatial rationales; answer: the SAME derived records with
  position-only targets (``derived_reasoning_{chain,answer}.jsonl``). Identical images, elements
  and question count — the delta is attributable to the rationale text alone.
* **A3 combination** — {base} vs {+spot} vs {+reason} vs {+spot+reason}, equal totals.
* **A4 language mixing** — {en} vs {en+X} pairs from ``lang_<code>.jsonl``, equal totals;
  transfer(L1←L2) read off per-language heldout scores.
* A5 placement / A7 preprocessing are knobs on ANY arm: ``--placement``, ``--max-image-long-side``
  pass through.

Every run trains via ``run_ablation.py --arm public`` and evaluates BOTH the synthetic probe suite
and the UDD public heldout fold (``heldout_all.jsonl`` — leakage-safe, image-keyed split). Results
land under ``models[<model>]["U-<arm>"]`` in ``docs/results/udd_ablation_results.json``.

    python scripts/build_task_trainsets.py --per-task 300 --merge-qa --derive-spatial-reasoning
    python scripts/run_udd_ablation.py --arm A1 --dry-run     # compose + inspect, no GPU
    python scripts/run_udd_ablation.py --arm A1 A2 --count 300 --steps 300   # GPU
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "udd_ablation_results.json"


def _mix(parts: list[tuple[Path, float]], n_total: int, seed: int, out_path: Path) -> int:
    """Compose a training jsonl from (file, share) parts, subsampled to n_total in the given shares."""
    rng = random.Random(seed)
    rows: list[str] = []
    for path, share in parts:
        pool = [ln for ln in path.read_text().splitlines() if ln.strip()]
        rng.shuffle(pool)
        rows += pool[: max(1, int(n_total * share))]
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def arm_definitions(td: Path) -> dict[str, dict]:
    """Arm -> {parts: equal-total composition, extra: per-arm run_ablation flags}.
    Only arms whose ingredient files exist are offered."""
    t = lambda name: td / f"task_{name}.jsonl"          # noqa: E731
    d = lambda name: td / f"derived_reasoning_{name}.jsonl"   # noqa: E731
    lang = lambda code: td.parent / "udd_langs" / f"lang_{code}.jsonl"  # noqa: E731
    # the full-signal composite mix: A5/A6 vary HOW we train, so the data stays fixed at this mix
    composite = [(t("vqa"), 1 / 4), (t("kie"), 1 / 4), (t("localization"), 1 / 4), (d("chain"), 1 / 4)]
    arms: dict[str, dict] = {
        # A1: does adding WHERE (grounding rows) help, at equal N?
        "A1_spotting_on": {"parts": [(t("vqa"), 1 / 3), (t("kie"), 1 / 3),
                                     (t("localization"), 1 / 3)]},
        "A1_spotting_off": {"parts": [(t("vqa"), 1 / 2), (t("kie"), 1 / 2)]},
        # A2: rationale target vs answer-only target on the SAME derived records
        "A2_reason_chain": {"parts": [(d("chain"), 1.0)]},
        "A2_reason_answer": {"parts": [(d("answer"), 1.0)]},
        # A3: is it the structured signals, or just task combination?
        "A3_base": {"parts": [(t("vqa"), 1 / 2), (t("kie"), 1 / 2)]},
        "A3_spot": {"parts": [(t("vqa"), 1 / 3), (t("kie"), 1 / 3), (t("localization"), 1 / 3)]},
        "A3_reason": {"parts": [(t("vqa"), 1 / 3), (t("kie"), 1 / 3), (d("chain"), 1 / 3)]},
        "A3_spot_reason": {"parts": composite},
    }
    # A4: en alone vs en+X pairs, for every language set that exists
    if lang("en").exists():
        arms["A4_en"] = {"parts": [(lang("en"), 1.0)]}
        for code in ("ko", "ja", "ar", "zh", "id", "fr", "de"):
            if lang(code).exists():
                arms[f"A4_en_{code}"] = {"parts": [(lang("en"), 1 / 2), (lang(code), 1 / 2)]}
    # A5: LoRA placement over the FIXED composite mix — which modules move which capability
    for grp in ("vision", "connector", "llm_attn", "llm_mlp"):
        arms[f"A5_{grp}"] = {"parts": composite, "extra": ["--placement", grp]}
    # A6: HPO over the FIXED composite mix (alpha = 2r convention; placement stays the CLI default
    # until the A5 winner is known, then rerun with --placement <winner>)
    for r in (8, 16, 32, 64):
        arms[f"A6_r{r}"] = {"parts": composite,
                            "extra": ["--lora-r", str(r), "--lora-alpha", str(2 * r)]}
    return {name: spec for name, spec in arms.items()
            if all(p.exists() for p, _ in spec["parts"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasksets-dir", default=str(ROOT / "data" / "udd_tasks"))
    p.add_argument("--arm", nargs="+", required=True,
                   help="arm ids or families: A1 A2 A3 A4 (family expands to all its variants)")
    p.add_argument("--models", nargs="+", default=["lfm2_5-vl-1.6b"])
    p.add_argument("--count", type=int, default=300, help="EQUAL total samples per arm (the control)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--placement", default="all", help="A5 knob: vision|connector|llm_attn|llm_mlp|all")
    p.add_argument("--max-image-long-side", type=int, default=768, help="A7 knob")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--results", default=str(RESULTS))
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--dry-run", action="store_true", help="compose arm jsonls + report, no training")
    p.add_argument("--smoke", action="store_true",
                   help="SIMPLE TEST of train+validate for every selected arm: tiny budget "
                        "(count=24, steps=8, eval 16 samples/probe, 512px) so all arms run "
                        "end-to-end fast — the wiring/before-after proof, NOT a measurement")
    p.add_argument("--before-after", action="store_true", default=True,
                   help="record base-vs-adapted on the full suite with per-axis deltas and saved "
                        "predictions (default ON — the cross-capability comparison is the point)")
    args = p.parse_args()
    if args.smoke:
        args.count, args.steps = min(args.count, 24), min(args.steps, 8)
        args.max_image_long_side = min(args.max_image_long_side, 512)
        print("[smoke] tiny train+validate: count=24 steps=8 eval=16/probe 512px — wiring proof, "
              "not a measurement\n")

    td = Path(args.tasksets_dir)
    arms = arm_definitions(td)
    selected = []
    for a in args.arm:
        fam = [k for k in arms if k == a or k.startswith(a + "_")]
        if not fam:
            sys.exit(f"[udd-ablation] unknown arm '{a}'. Available: {sorted(arms)}")
        selected += fam
    heldout = td / "heldout_all.jsonl"

    print(f"[udd-ablation] {len(selected)} arm runs, N={args.count}/arm, "
          f"heldout={'yes' if heldout.exists() else 'MISSING'}\n")
    for i, name in enumerate(selected, 1):
        spec = arms[name]
        train = td / "arms" / f"{name}.jsonl"
        n = _mix(spec["parts"], args.count, args.seed, train)
        extra = spec.get("extra", [])
        print(f"[{i}/{len(selected)}] {name}: {n} samples <- "
              + " + ".join(f"{p.name}×{s:.2f}" for p, s in spec["parts"])
              + (f"   [{' '.join(extra)}]" if extra else ""))
        if n < 0.95 * args.count:
            print(f"    [warn] {name} fell short of the equal-N target ({n} < {args.count}): a "
                  f"source pool is too small. Compare it only against arms at the SAME total, or "
                  f"lower --count to {n} for this family.")
        if args.dry_run:
            continue
        cmd = [sys.executable, "scripts/run_ablation.py", "--models", *args.models,
               "--arm", "public", "--train-jsonl", str(train),
               "--record-key", f"U-{name}", "--results", args.results,
               "--count", str(args.count), "--steps", str(args.steps),
               "--placement", args.placement,
               "--max-image-long-side", str(args.max_image_long_side)] + extra
        if args.before_after:
            cmd.append("--before-after")
        if args.smoke:
            cmd += ["--eval-max-samples", "16"]
        if heldout.exists():
            cmd += ["--heldout-jsonl", str(heldout)]
        if args.wandb_project:
            cmd += ["--wandb-project", args.wandb_project]
        subprocess.run(cmd, cwd=ROOT, check=True)
    if args.dry_run:
        print(f"\n[dry-run] composed arm jsonls under {td/'arms'}; rerun without --dry-run on a GPU.")
    else:
        print(f"\n[done] -> {args.results}")


if __name__ == "__main__":
    main()

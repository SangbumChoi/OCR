#!/usr/bin/env python3
"""Task-value ablation (GPU): is each UDD task *worth adding* to the training mix?

For every per-task training set built by ``scripts/build_task_trainsets.py`` (equal N samples each),
LoRA-fine-tune the base **on that task alone** and evaluate on the fixed synthetic probe suite, then
diff against the un-tuned base. The per-task Δ answers "does this task, at a fixed data budget, buy
capability?" — a positive Δ means the task is worth its slot; ≈0 or negative means it is not (or
transfers poorly to the validation distribution). A mixed ``all`` run is included as the reference.

This is a thin orchestrator over ``scripts/run_ablation.py --arm public`` (which does the actual
train→eval and appends to a results JSON). Results land in ``docs/results/task_value_results.json``
under ``models[<model>]["baseline"|"task:<task>"|"task:all"]``; render them with
``scripts/analyze_task_value.py``.

    # 1) build equal-N per-task sets (offline, CPU)
    python scripts/build_task_trainsets.py --per-task 30
    # 2) train + eval each task (GPU); default base LFM2.5-VL
    python scripts/run_task_value.py --count 30 --steps 300
    # 3) render the value table + chart
    python scripts/analyze_task_value.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "task_value_results.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasksets-dir", default=str(ROOT / "data" / "udd_tasks"),
                   help="dir of task_<task>.jsonl from build_task_trainsets.py")
    p.add_argument("--models", nargs="+", default=["lfm2_5-vl-1.6b"],
                   help="Part-2 base(s); default LFM2.5-VL (fast on a T4)")
    p.add_argument("--count", type=int, default=30,
                   help="samples/task to train on (the FIXED data budget; keep = build --per-task)")
    p.add_argument("--steps", type=int, default=300, help="fixed max training steps (iteration control)")
    p.add_argument("--placement", default="all", help="LoRA group (vision|connector|llm_attn|llm_mlp|all)")
    p.add_argument("--include-all", action="store_true",
                   help="also train on the mixed all.jsonl (reference for the full-mix ceiling)")
    p.add_argument("--skip-baseline", action="store_true", help="do not (re)run the un-tuned baseline")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--results", default=str(RESULTS))
    args = p.parse_args()

    td = Path(args.tasksets_dir)
    task_jsonls = sorted(td.glob("task_*.jsonl"))
    if not task_jsonls:
        sys.exit(f"[task-value] no task_*.jsonl in {td} — run build_task_trainsets.py first.")
    if args.include_all and (td / "all.jsonl").exists():
        task_jsonls.append(td / "all.jsonl")

    def run(extra: list[str]) -> None:
        cmd = [sys.executable, "scripts/run_ablation.py", "--models", *args.models,
               "--results", args.results] + extra
        if args.wandb_project:
            cmd += ["--wandb-project", args.wandb_project]
        print("      $ " + " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)

    steps = [("baseline", ["--arm", "baseline"])] if not args.skip_baseline else []
    for jl in task_jsonls:
        task = jl.stem.replace("task_", "") if jl.stem.startswith("task_") else "all"
        steps.append((f"task:{task}",
                      ["--arm", "public", "--train-jsonl", str(jl), "--placement", args.placement,
                       "--count", str(args.count), "--steps", str(args.steps),
                       "--record-key", f"task:{task}"]))

    total = len(steps)
    print(f"[task-value] {total} runs (baseline + {len(task_jsonls)} task sets) x {len(args.models)} "
          f"model(s); N={args.count}, steps={args.steps}\n")
    for i, (name, extra) in enumerate(steps, 1):
        print(f"[{i}/{total}] ({total - i} left) {name}")
        run(extra)
    print(f"\n[done] task-value results -> {args.results}\n"
          f"       render with: python scripts/analyze_task_value.py --results {args.results}")


if __name__ == "__main__":
    main()

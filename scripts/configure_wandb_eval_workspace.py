#!/usr/bin/env python3
"""Create a W&B workspace with paired train and held-out evaluation curves."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping

_EVAL_KEY = re.compile(r"^eval/(train|heldout|held)_(.+)$")
_AXIS_FIRST_KEY = re.compile(
    r"^eval_by_axis/(.+)/(train|heldout|held)$"
)


def collect_eval_metric_pairs(
    metric_keys: Iterable[str],
) -> dict[str, tuple[str, str]]:
    """Return paired train and held-out metrics, preferring axis-first keys."""
    by_namespace: dict[str, dict[str, dict[str, str]]] = {
        "axis_first": {},
        "split_first": {},
    }
    for key in metric_keys:
        axis_first = _AXIS_FIRST_KEY.fullmatch(key)
        split_first = _EVAL_KEY.fullmatch(key)
        if axis_first is not None:
            axis, split = axis_first.groups()
            namespace = "axis_first"
        elif split_first is not None:
            split, axis = split_first.groups()
            namespace = "split_first"
        else:
            continue
        canonical_split = "heldout" if split == "held" else split
        split_keys = by_namespace[namespace].setdefault(axis, {})
        if canonical_split not in split_keys or split == canonical_split:
            split_keys[canonical_split] = key

    pairs: dict[str, tuple[str, str]] = {}
    axes = set(by_namespace["split_first"]) | set(
        by_namespace["axis_first"]
    )
    for axis in sorted(axes):
        for namespace in ("axis_first", "split_first"):
            keys = by_namespace[namespace].get(axis, {})
            if "train" in keys and "heldout" in keys:
                pairs[axis] = (keys["train"], keys["heldout"])
                break
    return pairs


def choose_workspace_x_axis(
    metric_keys: Iterable[str],
    requested: str = "auto",
) -> str:
    if requested != "auto":
        return requested
    keys = set(metric_keys)
    if "epoch" in keys:
        return "epoch"
    if "evaluation/checkpoint_step" in keys:
        return "evaluation/checkpoint_step"
    return "_step"


def _project_metric_keys(runs: Iterable[object]) -> set[str]:
    keys: set[str] = set()
    for run in runs:
        summary = getattr(run, "summary", {})
        if isinstance(summary, Mapping):
            keys.update(str(key) for key in summary)
        else:
            keys.update(str(key) for key in summary.keys())
    return keys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a W&B workspace where each evaluation axis has one panel "
            "containing its train and held-out curves."
        )
    )
    parser.add_argument("--entity", required=True, help="W&B entity or team name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument(
        "--name",
        default="Train vs heldout by evaluation axis",
        help="Name of the saved W&B workspace view",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=100,
        help="Maximum runs shown in each panel",
    )
    parser.add_argument(
        "--x-axis",
        choices=("auto", "epoch", "evaluation/checkpoint_step", "_step"),
        default="auto",
        help=(
            "Panel x-axis; auto prefers legacy epoch, then native "
            "evaluation/checkpoint_step, then W&B _step"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        import wandb
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as error:
        raise SystemExit(
            "Install the workspace dependencies first: "
            "pip install wandb wandb-workspaces"
        ) from error

    api = wandb.Api()
    runs = list(api.runs(f"{args.entity}/{args.project}"))
    metric_keys = _project_metric_keys(runs)
    pairs = collect_eval_metric_pairs(metric_keys)
    if not pairs:
        raise SystemExit(
            "No paired train and held-out evaluation metrics were found."
        )
    x_axis = choose_workspace_x_axis(metric_keys, args.x_axis)

    panels = []
    for axis, (train_key, heldout_key) in pairs.items():
        panels.append(
            wr.LinePlot(
                title=axis,
                x=x_axis,
                y=[train_key, heldout_key],
                range_y=(0.0, 1.0),
                smoothing_type="none",
                max_runs_to_show=args.max_runs,
                legend_position="south",
                line_titles={train_key: "train", heldout_key: "heldout"},
                line_colors={train_key: "#2563EB", heldout_key: "#DC2626"},
            )
        )

    workspace = ws.Workspace(
        entity=args.entity,
        project=args.project,
        name=args.name,
        sections=[
            ws.Section(
                name="Train vs heldout evaluation",
                panels=panels,
                is_open=True,
            )
        ],
        settings=ws.WorkspaceSettings(
            x_axis=x_axis,
            smoothing_type="none",
            sort_panels_alphabetically=True,
            max_runs=args.max_runs,
        ),
    )
    workspace.save()
    print(
        f"Created {len(panels)} paired panels with x={x_axis}: "
        f"{workspace.url}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a W&B workspace with paired train and held-out evaluation curves."""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Mapping

_EVAL_KEY = re.compile(r"^eval/(train|heldout|held)_(.+)$")


def collect_eval_metric_pairs(
    metric_keys: Iterable[str],
) -> dict[str, tuple[str, str]]:
    """Return common ``eval/train_*`` and held-out metrics keyed by axis."""
    by_axis: dict[str, dict[str, str]] = {}
    for key in metric_keys:
        match = _EVAL_KEY.fullmatch(key)
        if match is None:
            continue
        split, axis = match.groups()
        canonical_split = "heldout" if split == "held" else split
        split_keys = by_axis.setdefault(axis, {})
        if canonical_split not in split_keys or split == canonical_split:
            split_keys[canonical_split] = key

    return {
        axis: (keys["train"], keys["heldout"])
        for axis, keys in sorted(by_axis.items())
        if "train" in keys and "heldout" in keys
    }


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
    pairs = collect_eval_metric_pairs(_project_metric_keys(runs))
    if not pairs:
        raise SystemExit(
            "No matching eval/train_<axis> and eval/heldout_<axis> metrics were found."
        )

    panels = []
    for axis, (train_key, heldout_key) in pairs.items():
        panels.append(
            wr.LinePlot(
                title=axis,
                x="epoch",
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
            x_axis="epoch",
            smoothing_type="none",
            sort_panels_alphabetically=True,
            max_runs=args.max_runs,
        ),
    )
    workspace.save()
    print(f"Created {len(panels)} paired panels: {workspace.url}")


if __name__ == "__main__":
    main()

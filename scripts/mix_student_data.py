#!/usr/bin/env python3
"""Build one weighted student-pretraining corpus from on-disk UDD components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.student.mixture import MixtureComponent, build_weighted_mixture


def _parse_component(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--component must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if not name or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid component {value!r}")
    return name, path


def _parse_weight(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--weight must use NAME=FLOAT")
    name, raw_weight = value.split("=", 1)
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid weight {value!r}") from exc
    return name, weight


def _parse_fold(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--fold must use NAME=FOLD")
    name, fold = value.split("=", 1)
    if not name or not fold:
        raise argparse.ArgumentTypeError(f"invalid fold {value!r}")
    return name, fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", action="append", type=_parse_component, required=True)
    parser.add_argument("--weight", action="append", type=_parse_weight, required=True)
    parser.add_argument("--fold", action="append", type=_parse_fold, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = dict(args.component)
    weights = dict(args.weight)
    folds = dict(args.fold)
    if set(paths) != set(weights):
        raise SystemExit(
            "--component and --weight names must match exactly: "
            f"components={sorted(paths)}, weights={sorted(weights)}"
        )
    unknown_folds = set(folds) - set(paths)
    if unknown_folds:
        raise SystemExit(f"--fold references unknown components: {sorted(unknown_folds)}")
    manifest = build_weighted_mixture(
        [
            MixtureComponent(name, str(path), weights[name], folds.get(name))
            for name, path in paths.items()
        ],
        args.output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

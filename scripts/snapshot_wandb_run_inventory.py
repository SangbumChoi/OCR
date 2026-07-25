#!/usr/bin/env python3
"""Capture compact authenticated W&B run identity and state."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from docvlm_eval.student.wandb_inventory import (
    build_wandb_run_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="sbdc")
    parser.add_argument("--project", default="docvlm-ablation")
    parser.add_argument("--max-runs", type=int, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "docvlm_ablation_wandb_run_inventory.json"
        ),
    )
    args = parser.parse_args()
    if args.max_runs <= 0:
        parser.error("--max-runs must be positive")

    import wandb

    api = wandb.Api()
    api_runs = api.runs(
        f"{args.entity}/{args.project}",
        order="-created_at",
        per_page=min(args.max_runs, 100),
    )
    records = [
        {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "created_at": _timestamp(getattr(run, "created_at", None)),
            "updated_at": _timestamp(getattr(run, "updated_at", None)),
        }
        for run in islice(api_runs, args.max_runs + 1)
    ]
    if len(records) > args.max_runs:
        raise SystemExit(
            "W&B project exceeds --max-runs; increase the bound so the "
            "inventory remains complete"
        )
    inventory = build_wandb_run_inventory(
        records,
        entity=args.entity,
        project=args.project,
        observed_at=datetime.now(UTC).isoformat(),
    )
    _atomic_write(args.output, inventory)
    print(
        json.dumps(
            {
                "fingerprint": inventory["fingerprint"],
                "output": str(args.output.resolve()),
                "run_count": inventory["run_count"],
                "states": inventory["states"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

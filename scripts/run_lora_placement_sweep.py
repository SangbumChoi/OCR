#!/usr/bin/env python3
"""Compile or run the paired vision versus vision+connector LoRA sweep."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def compile_commands(
    raw: dict[str, Any],
    *,
    python: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """Validate the matched design and return one command per paired cell."""
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    name = str(raw.get("name") or "")
    model = str(raw.get("model") or "")
    arm = str(raw.get("arm") or "")
    placements = raw.get("placements")
    if (
        not name
        or not model
        or not arm
        or not isinstance(placements, list)
        or placements != ["vision", "vision_connector"]
    ):
        raise ValueError(
            "the sweep requires one vision and one vision_connector cell"
        )
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("controls must be a mapping")
    if controls.get("lora_budget_reference") != "vision":
        raise ValueError("lora_budget_reference must be vision")
    grounding_target = controls.get("grounding_target")
    if grounding_target not in {"pixel", "norm"}:
        raise ValueError("grounding_target must be pixel or norm")
    int_controls = {
        key: _positive_int(controls.get(key), f"controls.{key}")
        for key in (
            "count",
            "steps",
            "heldout_seed",
            "grounding_repeat",
            "max_image_long_side",
            "batch_size",
            "lora_rank",
            "lora_alpha",
            "eval_max_samples",
        )
    }
    learning_rate = controls.get("learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or learning_rate <= 0
    ):
        raise ValueError("controls.learning_rate must be positive")
    replicates = raw.get("replicates")
    if not isinstance(replicates, list) or len(replicates) < 2:
        raise ValueError("at least two paired replicates are required")
    replicate_ids: set[str] = set()
    seed_pairs: set[tuple[int, int]] = set()
    normalized_replicates = []
    for replicate in replicates:
        if not isinstance(replicate, dict):
            raise ValueError("each replicate must be a mapping")
        replicate_id = str(replicate.get("id") or "")
        optimizer_seed = _positive_int(
            replicate.get("optimizer_seed"),
            f"{replicate_id}.optimizer_seed",
        )
        data_seed = _positive_int(
            replicate.get("data_seed"),
            f"{replicate_id}.data_seed",
        )
        if not replicate_id or replicate_id in replicate_ids:
            raise ValueError("replicate ids must be non-empty and unique")
        pair = (optimizer_seed, data_seed)
        if pair in seed_pairs:
            raise ValueError("replicate seed pairs must be unique")
        replicate_ids.add(replicate_id)
        seed_pairs.add(pair)
        normalized_replicates.append((replicate_id, optimizer_seed, data_seed))
    wandb = raw.get("wandb") or {}
    if not isinstance(wandb, dict):
        raise ValueError("wandb must be a mapping")

    commands = []
    runner = repo_root / "scripts" / "run_ablation.py"
    for replicate_id, optimizer_seed, data_seed in normalized_replicates:
        for placement in placements:
            record_key = f"{name}:{placement}:{replicate_id}"
            command = [
                python,
                str(runner),
                "--models",
                model,
                "--arm",
                arm,
                "--placement",
                placement,
                "--count",
                str(int_controls["count"]),
                "--steps",
                str(int_controls["steps"]),
                "--heldout-seed",
                str(int_controls["heldout_seed"]),
                "--grounding-repeat",
                str(int_controls["grounding_repeat"]),
                "--grounding-target",
                grounding_target,
                "--max-image-long-side",
                str(int_controls["max_image_long_side"]),
                "--batch-size",
                str(int_controls["batch_size"]),
                "--lora-r",
                str(int_controls["lora_rank"]),
                "--lora-alpha",
                str(int_controls["lora_alpha"]),
                "--lora-budget-reference",
                "vision",
                "--lr",
                str(float(learning_rate)),
                "--eval-max-samples",
                str(int_controls["eval_max_samples"]),
                "--seed",
                str(optimizer_seed),
                "--data-seed",
                str(data_seed),
                "--record-key",
                record_key,
            ]
            if wandb.get("project"):
                command.extend(
                    ["--wandb-project", str(wandb["project"])]
                )
            if wandb.get("run_prefix"):
                command.extend(
                    ["--wandb-run-prefix", str(wandb["run_prefix"])]
                )
            commands.append(
                {
                    "variant": placement,
                    "replicate": replicate_id,
                    "optimizer_seed": optimizer_seed,
                    "data_seed": data_seed,
                    "record_key": record_key,
                    "command": command,
                }
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lora_vision_connector_sweep.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("sweep config must contain a mapping")
    commands = compile_commands(
        raw,
        python=sys.executable,
        repo_root=ROOT,
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "runs": commands}, indent=2))
        return
    for index, item in enumerate(commands, 1):
        print(
            f"[{index}/{len(commands)}] "
            f"{item['variant']} :: {item['replicate']}",
            flush=True,
        )
        subprocess.run(item["command"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

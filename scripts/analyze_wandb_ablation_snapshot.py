#!/usr/bin/env python3
"""Audit a compact W&B ablation snapshot without treating run state as evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "results" / "lfm_ablation_wandb_snapshot.json"
DEFAULT_OUTPUT = ROOT / "docs" / "results" / "lfm_ablation_wandb_analysis.json"

OBSERVED_CONTROLS = (
    "epochs",
    "learning_rate",
    "lora_rank",
    "train_jsonl",
)
REQUIRED_PROMOTION_CONTROLS = (
    "optimizer_seed",
    "data_seed",
    "max_steps",
    "training_sample_count",
    "heldout_sample_ids_fingerprint",
    "base_model_revision",
)


def _score_map(run: dict[str, Any]) -> dict[str, float]:
    heldout = run.get("heldout")
    if not isinstance(heldout, dict):
        return {}
    scores: dict[str, float] = {}
    for key, value in heldout.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"run {run.get('id')!r} has non-numeric heldout metric {key!r}")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"run {run.get('id')!r} has out-of-range heldout metric {key!r}")
        scores[str(key)] = number
    if "score" not in scores:
        raise ValueError(f"run {run.get('id')!r} has no heldout score")
    return scores


def _validate_runs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported W&B snapshot schema")
    runs = snapshot.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("W&B snapshot contains no runs")
    ids = []
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each W&B run must be a mapping")
        for field in ("id", "name", "arm", "state", "placement"):
            if not isinstance(run.get(field), str) or not run[field]:
                raise ValueError(f"W&B run is missing {field!r}")
        ids.append(run["id"])
        _score_map(run)
    duplicates = sorted(run_id for run_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate W&B run IDs: {duplicates}")
    return runs


def analyze_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Separate comparable evidence from incomplete or confounded W&B rows."""

    runs = _validate_runs(snapshot)
    evaluated = [run for run in runs if _score_map(run)]
    unevaluated_finished = [
        run["id"] for run in runs if run["state"] == "finished" and not _score_map(run)
    ]
    crashed = [run["id"] for run in runs if run["state"] == "crashed"]
    crash_diagnostics = {
        run["id"]: run["termination"]
        for run in runs
        if run["state"] == "crashed" and isinstance(run.get("termination"), dict)
    }
    duplicate_names = {
        name: count
        for name, count in sorted(Counter(run["name"] for run in runs).items())
        if count > 1
    }

    baseline = [run for run in evaluated if run["arm"] == "A0_generalization"]
    spotting = [run for run in evaluated if run["arm"] == "A1_spotting_on"]
    by_placement = {
        run["placement"]: run for run in spotting if run["placement"] in {"vision", "connector"}
    }
    if set(by_placement) != {"vision", "connector"}:
        raise ValueError("snapshot needs evaluated vision and connector A1 spotting runs")
    vision = by_placement["vision"]
    connector = by_placement["connector"]
    mismatched_controls = {
        field: [vision.get(field), connector.get(field)]
        for field in OBSERVED_CONTROLS
        if vision.get(field) != connector.get(field)
    }
    if mismatched_controls:
        raise ValueError(f"A1 vision and connector controls differ: {mismatched_controls}")

    vision_scores = _score_map(vision)
    connector_scores = _score_map(connector)
    common_metrics = sorted(set(vision_scores) & set(connector_scores))
    deltas = {
        metric: round(vision_scores[metric] - connector_scores[metric], 10)
        for metric in common_metrics
    }
    train_gap = {}
    for placement, run in by_placement.items():
        train_score = run.get("train_score")
        if isinstance(train_score, (int, float)) and not isinstance(train_score, bool):
            train_gap[placement] = round(
                float(train_score) - _score_map(run)["score"],
                10,
            )

    missing_controls = sorted(
        field
        for field in REQUIRED_PROMOTION_CONTROLS
        if any(run.get(field) is None for run in (vision, connector))
    )
    promotion_eligible = (
        len(baseline) > 0 and not missing_controls and not unevaluated_finished and not crashed
    )
    return {
        "schema_version": 1,
        "source_snapshot_sha256": (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    snapshot,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "source": snapshot["source"],
        "observed_at": snapshot["observed_at"],
        "run_quality": {
            "observed_runs": len(runs),
            "evaluated_runs": len(evaluated),
            "finished_without_evaluation": unevaluated_finished,
            "crashed_runs": crashed,
            "crash_diagnostics": crash_diagnostics,
            "duplicate_names": duplicate_names,
            "evaluated_A0_baselines": [run["id"] for run in baseline],
        },
        "comparable_preliminary_pair": {
            "arm": "A1_spotting_on",
            "vision_run": vision["id"],
            "connector_run": connector["id"],
            "matched_observed_controls": {field: vision[field] for field in OBSERVED_CONTROLS},
            "missing_promotion_controls": missing_controls,
            "heldout_vision_minus_connector": deltas,
            "train_minus_heldout_score": train_gap,
        },
        "evidence_status": (
            "promotion_eligible" if promotion_eligible else "preliminary_direction_only"
        ),
        "promotion_eligible": promotion_eligible,
        "recommended_next_experiment": ("configs/lora_vision_connector_sweep.yaml"),
        "interpretation": [
            (
                "W&B state alone is not completion evidence: finished runs "
                "without heldout metrics were excluded."
            ),
            (
                "Only the A1 vision and connector rows share the same arm and "
                "the controls visible in this snapshot."
            ),
            (
                "The single observed pair favors vision-only on overall score, "
                "grounding, and L1-locate, but it is not a promotion result."
            ),
            (
                "The A0 baseline crashed, so cross-arm scores cannot be "
                "interpreted as treatment effects."
            ),
            (
                "Run the budget-matched three-replicate vision versus "
                "vision_connector sweep before changing the default placement."
            ),
        ],
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    quality = result["run_quality"]
    pair = result["comparable_preliminary_pair"]
    deltas = pair["heldout_vision_minus_connector"]
    lines = [
        "# LFM W&B ablation snapshot audit",
        "",
        f"- Evidence status: `{result['evidence_status']}`",
        f"- Promotion eligible: `{str(result['promotion_eligible']).lower()}`",
        (
            f"- Run quality: {quality['evaluated_runs']} evaluated of "
            f"{quality['observed_runs']} observed"
        ),
        (f"- Evaluated A0 baselines: {len(quality['evaluated_A0_baselines'])}"),
        "",
        "## Comparable preliminary pair",
        "",
        (
            f"`{pair['vision_run']}` (vision) versus "
            f"`{pair['connector_run']}` (connector), both on "
            f"`{pair['arm']}`."
        ),
        "",
        "| Heldout metric | Vision minus connector |",
        "| --- | ---: |",
    ]
    for metric in (
        "score",
        "grounding",
        "L1-locate",
        "L1-region",
        "kie",
        "ocr-full",
        "H-comprehension",
    ):
        if metric in deltas:
            lines.append(f"| `{metric}` | {deltas[metric]:+.4f} |")
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            (f"- Finished without evaluation: {len(quality['finished_without_evaluation'])}"),
            f"- Crashed runs: {len(quality['crashed_runs'])}",
            (
                "- A0 terminal progress: "
                + (
                    f"{quality['crash_diagnostics']['r0t65g1h']['last_reported_micro_step']}"
                    f"/{quality['crash_diagnostics']['r0t65g1h']['planned_micro_steps']} "
                    "micro-steps"
                    if "r0t65g1h" in quality["crash_diagnostics"]
                    else "not recorded"
                )
            ),
            (
                "- Missing promotion controls: "
                + ", ".join(f"`{field}`" for field in pair["missing_promotion_controls"])
            ),
            "",
            (
                "The result is direction-only. Execute "
                f"[`{result['recommended_next_experiment']}`]"
                f"(../../{result['recommended_next_experiment']}) before "
                "promoting a placement."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    snapshot = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_snapshot(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output.with_suffix(".md"), result)
    print(
        json.dumps(
            {
                "evidence_status": result["evidence_status"],
                "promotion_eligible": result["promotion_eligible"],
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

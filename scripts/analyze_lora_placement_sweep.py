#!/usr/bin/env python3
"""Aggregate the paired vision versus vision+connector LoRA experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _axis_score(summary: dict[str, Any], axis: str) -> float:
    value = (summary.get("by_answer_type") or {}).get(axis)
    if isinstance(value, dict):
        value = value.get("score")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing finite heldout metric {axis!r}")
    return float(value)


def _flatten(summary: dict[str, Any]) -> dict[str, float]:
    metrics = {}
    score = summary.get("score")
    if isinstance(score, (int, float)) and math.isfinite(float(score)):
        metrics["score"] = float(score)
    for axis in (summary.get("by_answer_type") or {}):
        metrics[str(axis)] = _axis_score(summary, str(axis))
    return metrics


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_statistics(
    values: list[float],
    *,
    key: str,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    seed = int.from_bytes(
        hashlib.sha256(key.encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    bootstraps = [
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(bootstrap_samples)
    ]
    ci95 = [
        _percentile(bootstraps, 0.025),
        _percentile(bootstraps, 0.975),
    ]
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else None
        ),
        "ci95": ci95,
        "conclusion": (
            "improved"
            if ci95[0] > 0
            else "regressed"
            if ci95[1] < 0
            else "inconclusive"
        ),
    }


def aggregate_results(
    config: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    """Build paired deltas and a fail-closed promotion decision."""
    name = str(config["name"])
    model = str(config["model"])
    model_results = (results.get("models") or {}).get(model)
    if not isinstance(model_results, dict):
        raise ValueError(f"results contain no model {model!r}")
    placements = list(config["placements"])
    if placements != ["vision", "vision_connector"]:
        raise ValueError("placements must be ordered vision, vision_connector")
    replicate_ids = [str(item["id"]) for item in config["replicates"]]
    paired = {}
    budget_errors = []
    gap_deltas = []
    for replicate_id in replicate_ids:
        cells = {}
        for placement in placements:
            key = f"{name}:{placement}:{replicate_id}"
            payload = model_results.get(key)
            if not isinstance(payload, dict):
                raise ValueError(f"missing completed result {key!r}")
            heldout = (payload.get("probes") or {}).get("heldout")
            if not isinstance(heldout, dict):
                raise ValueError(f"{key!r} has no heldout probe summary")
            budget = (payload.get("control") or {}).get("lora_budget")
            if not isinstance(budget, dict):
                raise ValueError(f"{key!r} has no LoRA budget evidence")
            budget_error = budget.get("realized_relative_budget_error")
            if not isinstance(budget_error, (int, float)):
                raise ValueError(f"{key!r} has no realized budget error")
            budget_errors.append(float(budget_error))
            training_eval = payload.get("training_eval") or {}
            train_score = (training_eval.get("train") or {}).get("score")
            heldout_score = (training_eval.get("heldout") or {}).get("score")
            if not isinstance(train_score, (int, float)) or not isinstance(
                heldout_score,
                (int, float),
            ):
                raise ValueError(f"{key!r} has no train/heldout gap evidence")
            cells[placement] = {
                "metrics": _flatten(heldout),
                "generalization_gap": float(train_score) - float(heldout_score),
                "budget_error": float(budget_error),
            }
        common = sorted(
            set(cells["vision"]["metrics"])
            & set(cells["vision_connector"]["metrics"])
        )
        deltas = {
            metric: (
                cells["vision_connector"]["metrics"][metric]
                - cells["vision"]["metrics"][metric]
            )
            for metric in common
        }
        gap_delta = (
            cells["vision_connector"]["generalization_gap"]
            - cells["vision"]["generalization_gap"]
        )
        gap_deltas.append(gap_delta)
        paired[replicate_id] = {
            "cells": cells,
            "delta_vision_connector_minus_vision": deltas,
            "generalization_gap_delta": gap_delta,
        }

    common_metrics = sorted(
        set.intersection(
            *[
                set(
                    item["delta_vision_connector_minus_vision"]
                )
                for item in paired.values()
            ]
        )
    )
    statistics_by_metric = {
        metric: _paired_statistics(
            [
                paired[replicate_id][
                    "delta_vision_connector_minus_vision"
                ][metric]
                for replicate_id in replicate_ids
            ],
            key=f"{name}:{metric}",
        )
        for metric in common_metrics
    }
    gap_statistics = _paired_statistics(
        gap_deltas,
        key=f"{name}:generalization-gap",
    )
    analysis = config.get("analysis") or {}
    primary_metrics = [str(value) for value in analysis["primary_metrics"]]
    guard_metrics = [str(value) for value in analysis["guard_metrics"]]
    required = set(primary_metrics) | set(guard_metrics)
    missing = sorted(required - statistics_by_metric.keys())
    if missing:
        raise ValueError(f"required heldout metrics are missing: {missing}")
    require_all = bool(
        analysis.get("require_all_primary_replicates_nonnegative", True)
    )
    primary_pass = all(
        statistics_by_metric[metric]["mean"] > 0
        and (
            not require_all
            or all(
                value >= 0
                for value in statistics_by_metric[metric]["values"]
            )
        )
        for metric in primary_metrics
    )
    guard_threshold = float(analysis["guard_noninferiority"])
    guard_pass = all(
        statistics_by_metric[metric]["mean"] >= guard_threshold
        for metric in guard_metrics
    )
    budget_pass = max(budget_errors) <= float(analysis["max_budget_error"])
    gap_pass = gap_statistics["mean"] <= 0
    gates = {
        "primary_direction": primary_pass,
        "guard_noninferiority": guard_pass,
        "adapter_budget": budget_pass,
        "generalization_gap": gap_pass,
    }
    return {
        "schema_version": 1,
        "sweep": name,
        "model": model,
        "replicates": replicate_ids,
        "paired_results": paired,
        "metric_statistics": statistics_by_metric,
        "generalization_gap_statistics": gap_statistics,
        "maximum_realized_budget_error": max(budget_errors),
        "gates": gates,
        "decision": "promote" if all(gates.values()) else "reject",
    }


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Vision and connector LoRA interaction results",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Replicates: {len(result['replicates'])}",
        (
            "- Maximum realized adapter-budget error: "
            f"{result['maximum_realized_budget_error']:.2%}"
        ),
        "",
        "| Metric | Mean delta | 95% CI | Conclusion |",
        "| --- | ---: | ---: | --- |",
    ]
    for metric, statistics_ in result["metric_statistics"].items():
        ci95 = statistics_["ci95"]
        lines.append(
            f"| `{metric}` | {statistics_['mean']:+.4f} | "
            f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}] | "
            f"`{statistics_['conclusion']}` |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- `{name}`: `{'pass' if passed else 'fail'}`"
        for name, passed in result["gates"].items()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "lora_vision_connector_sweep.yaml",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "docs" / "results" / "ablation_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "lora_vision_connector_sweep.json"
        ),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    result = aggregate_results(config, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(args.output.with_suffix(".md"), result)
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

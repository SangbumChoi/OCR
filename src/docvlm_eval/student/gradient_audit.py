"""Aggregation and promotion gates for gradient-conflict diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

from .sweep import SweepPlan


_COSINE_PREFIX = "gradient_probe/cosine/"
_OVERLAP_PREFIX = "gradient_probe/overlap_elements/"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latest_model_hash(run_root: Path) -> str | None:
    pretrain = run_root / "artifacts" / "pretrain"
    pointer = pretrain / "latest_checkpoint.txt"
    if not pointer.is_file():
        return None
    checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    model = checkpoint / "student" / "model.pt"
    return _sha256(model) if model.is_file() else None


def _run_summary(
    *,
    run_id: str,
    arm_id: str,
    replicate_id: str,
    run_root: Path,
    require_metrics: bool,
) -> dict[str, Any]:
    metrics_path = run_root / "artifacts" / "pretrain" / "metrics.jsonl"
    if not metrics_path.is_file():
        if require_metrics:
            raise FileNotFoundError(
                f"gradient audit run {run_id!r} has no metrics: {metrics_path}"
            )
        return {
            "run_id": run_id,
            "arm_id": arm_id,
            "replicate_id": replicate_id,
            "probe_records": 0,
            "model_sha256": _latest_model_hash(run_root),
            "pairs": {},
            "metrics_path": None,
        }
    pair_values: dict[str, list[float]] = {}
    pair_overlaps: dict[str, list[float]] = {}
    probe_records = 0
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {metrics_path}:{line_number}"
                ) from error
            if record.get("kind") != "gradient_conflict":
                continue
            probe_records += 1
            for key, raw_value in record.items():
                if not key.startswith(_COSINE_PREFIX):
                    continue
                pair = key.removeprefix(_COSINE_PREFIX)
                cosine = _finite(raw_value)
                if cosine is None:
                    continue
                pair_values.setdefault(pair, []).append(cosine)
                overlap = _finite(record.get(f"{_OVERLAP_PREFIX}{pair}"))
                if overlap is not None:
                    pair_overlaps.setdefault(pair, []).append(overlap)
    if probe_records == 0:
        if not require_metrics:
            return {
                "run_id": run_id,
                "arm_id": arm_id,
                "replicate_id": replicate_id,
                "probe_records": 0,
                "model_sha256": _latest_model_hash(run_root),
                "pairs": {},
                "metrics_path": str(metrics_path),
            }
        raise ValueError(
            f"gradient audit run {run_id!r} has no gradient_conflict records"
        )
    pairs = {}
    for pair, values in sorted(pair_values.items()):
        overlaps = pair_overlaps.get(pair, [])
        pairs[pair] = {
            "measurements": len(values),
            "negative_measurements": sum(value < 0 for value in values),
            "mean_cosine": fmean(values),
            "minimum_cosine": min(values),
            "maximum_cosine": max(values),
            "negative_fraction": sum(value < 0 for value in values) / len(values),
            "mean_overlap_elements": fmean(overlaps) if overlaps else None,
        }
    return {
        "run_id": run_id,
        "arm_id": arm_id,
        "replicate_id": replicate_id,
        "probe_records": probe_records,
        "model_sha256": _latest_model_hash(run_root),
        "pairs": pairs,
        "metrics_path": str(metrics_path),
    }


def _arm_summary(
    runs: list[dict[str, Any]],
    *,
    expected_replicates: set[str],
    minimum_measurements: int,
) -> dict[str, Any]:
    pair_measurements: dict[str, int] = {}
    pair_negative_measurements: dict[str, int] = {}
    pair_weighted_sums: dict[str, float] = {}
    pair_run_means: dict[str, list[float]] = {}
    pair_replicates: dict[str, set[str]] = {}
    for run in runs:
        for pair, summary in run["pairs"].items():
            count = int(summary["measurements"])
            mean = float(summary["mean_cosine"])
            pair_measurements[pair] = pair_measurements.get(pair, 0) + count
            pair_negative_measurements[pair] = (
                pair_negative_measurements.get(pair, 0)
                + int(summary["negative_measurements"])
            )
            pair_weighted_sums[pair] = (
                pair_weighted_sums.get(pair, 0.0) + mean * count
            )
            pair_run_means.setdefault(pair, []).append(mean)
            pair_replicates.setdefault(pair, set()).add(run["replicate_id"])
    pairs = {}
    for pair, count in sorted(pair_measurements.items()):
        replicates = sorted(pair_replicates[pair])
        run_means = pair_run_means[pair]
        pairs[pair] = {
            "measurements": count,
            "replicates_observed": replicates,
            "mean_cosine": pair_weighted_sums[pair] / count,
            "minimum_run_mean_cosine": min(run_means),
            "maximum_run_mean_cosine": max(run_means),
            "negative_fraction": pair_negative_measurements[pair] / count,
            "sufficient_coverage": (
                count >= minimum_measurements
                and set(replicates) == expected_replicates
            ),
        }
    return {
        "runs": len(runs),
        "probe_records": sum(int(run["probe_records"]) for run in runs),
        "pairs": pairs,
    }


def aggregate_gradient_conflict_audit(
    plan: SweepPlan,
    *,
    minimum_measurements: int = 20,
    negative_fraction_threshold: float = 0.25,
    mean_cosine_threshold: float = -0.05,
    material_mean_delta: float = 0.05,
) -> dict[str, Any]:
    """Aggregate a three-arm replicated probe audit and make a promotion decision."""

    if minimum_measurements <= 0:
        raise ValueError("minimum_measurements must be positive")
    if not all(
        math.isfinite(value)
        for value in (
            negative_fraction_threshold,
            mean_cosine_threshold,
            material_mean_delta,
        )
    ):
        raise ValueError("gradient audit thresholds must be finite")
    if not 0 <= negative_fraction_threshold <= 1:
        raise ValueError("negative_fraction_threshold must be in [0, 1]")
    if not -1 <= mean_cosine_threshold <= 1:
        raise ValueError("mean_cosine_threshold must be in [-1, 1]")
    if not 0 <= material_mean_delta <= 2:
        raise ValueError("material_mean_delta must be in [0, 2]")
    arms = sorted({variant.arm_id for variant in plan.variants})
    required_arms = {"no_probe", "vision_anchor", "all_trunks"}
    if set(arms) != required_arms or plan.baseline != "no_probe":
        raise ValueError(
            "gradient audit requires no_probe, vision_anchor, and all_trunks "
            "with no_probe as baseline"
        )
    proxy = "vision_anchor"
    candidate = "all_trunks"
    runs = [
        _run_summary(
            run_id=variant.id,
            arm_id=variant.arm_id,
            replicate_id=variant.replicate_id,
            run_root=Path(variant.plan.root),
            require_metrics=variant.arm_id != plan.baseline,
        )
        for variant in plan.variants
    ]
    expected_replicates = set(plan.replicates)
    by_arm = {
        arm: _arm_summary(
            [run for run in runs if run["arm_id"] == arm],
            expected_replicates=expected_replicates,
            minimum_measurements=minimum_measurements,
        )
        for arm in arms
    }

    trajectory_pairs = []
    for replicate in sorted(expected_replicates):
        baseline_run = next(
            run
            for run in runs
            if run["arm_id"] == plan.baseline
            and run["replicate_id"] == replicate
        )
        baseline_hash = baseline_run["model_sha256"]
        arm_hashes = {
            arm: next(
                run["model_sha256"]
                for run in runs
                if run["arm_id"] == arm
                and run["replicate_id"] == replicate
            )
            for arm in (proxy, candidate)
        }
        trajectory_pairs.append(
            {
                "replicate_id": replicate,
                "baseline_sha256": baseline_hash,
                "arm_sha256": arm_hashes,
                "matches": {
                    arm: (
                        baseline_hash == arm_hash
                        if baseline_hash is not None and arm_hash is not None
                        else None
                    )
                    for arm, arm_hash in arm_hashes.items()
                },
            }
        )
    trajectory_values = [
        match
        for pair in trajectory_pairs
        for match in pair["matches"].values()
    ]
    trajectory_status = (
        "fail"
        if False in trajectory_values
        else "pass"
        if trajectory_values and all(value is True for value in trajectory_values)
        else "insufficient_evidence"
    )

    proxy_pairs = by_arm[proxy]["pairs"]
    candidate_pairs = by_arm[candidate]["pairs"]
    conflict_pairs = []
    material_pairs = []
    for pair, summary in candidate_pairs.items():
        if not summary["sufficient_coverage"]:
            continue
        persistent_conflict = (
            summary["negative_fraction"]
            >= negative_fraction_threshold
            or summary["mean_cosine"] <= mean_cosine_threshold
        )
        if persistent_conflict:
            conflict_pairs.append(pair)
        proxy_summary = proxy_pairs.get(pair)
        if proxy_summary is None:
            material_pairs.append(pair)
        elif (
            proxy_summary["sufficient_coverage"]
            and abs(
                summary["mean_cosine"]
                - proxy_summary["mean_cosine"]
            )
            >= material_mean_delta
        ):
            material_pairs.append(pair)

    if trajectory_status == "fail":
        decision = "invalid_probe"
    elif trajectory_status != "pass" or not any(
        pair["sufficient_coverage"] for pair in candidate_pairs.values()
    ):
        decision = "insufficient_evidence"
    elif conflict_pairs and material_pairs:
        decision = "promote_gradient_surgery"
    else:
        decision = "retain_weighted_sum"
    return {
        "schema_version": 1,
        "sweep": plan.name,
        "baseline_arm": plan.baseline,
        "proxy_arm": proxy,
        "candidate_arm": candidate,
        "thresholds": {
            "minimum_measurements": minimum_measurements,
            "negative_fraction": negative_fraction_threshold,
            "mean_cosine": mean_cosine_threshold,
            "material_mean_delta": material_mean_delta,
        },
        "runs": runs,
        "arms": by_arm,
        "trajectory_invariance": {
            "status": trajectory_status,
            "replicates": trajectory_pairs,
        },
        "evidence": {
            "persistent_conflict_pairs": conflict_pairs,
            "material_anchor_difference_pairs": material_pairs,
        },
        "decision": decision,
    }


def _audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Gradient conflict audit",
        "",
        f"- Decision: `{audit['decision']}`",
        (
            "- Trajectory invariance: "
            f"`{audit['trajectory_invariance']['status']}`"
        ),
        (
            "- Persistent conflict pairs: "
            + (
                ", ".join(
                    f"`{pair}`"
                    for pair in audit["evidence"]["persistent_conflict_pairs"]
                )
                or "none"
            )
        ),
        (
            "- Material anchor differences: "
            + (
                ", ".join(
                    f"`{pair}`"
                    for pair in audit["evidence"][
                        "material_anchor_difference_pairs"
                    ]
                )
                or "none"
            )
        ),
        "",
        "| Arm | Pair | Measurements | Mean cosine | Negative fraction | Coverage |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for arm, arm_summary in sorted(audit["arms"].items()):
        for pair, summary in sorted(arm_summary["pairs"].items()):
            lines.append(
                f"| {arm} | `{pair}` | {summary['measurements']} | "
                f"{summary['mean_cosine']:.4f} | "
                f"{summary['negative_fraction']:.4f} | "
                f"{'sufficient' if summary['sufficient_coverage'] else 'insufficient'} |"
            )
    lines.extend(
        [
            "",
            "The decision is diagnostic evidence, not a quality claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_gradient_conflict_audit(
    plan: SweepPlan,
    *,
    output_dir: str | Path | None = None,
    **thresholds: Any,
) -> dict[str, Any]:
    """Write machine-readable and Markdown audit reports."""

    audit = aggregate_gradient_conflict_audit(plan, **thresholds)
    root = Path(output_dir) if output_dir is not None else Path(plan.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "gradient_conflict_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "gradient_conflict_audit.md").write_text(
        _audit_markdown(audit),
        encoding="utf-8",
    )
    return audit

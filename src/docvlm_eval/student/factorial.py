"""Initialization-by-data-scale factorial experiments for the native student."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .sweep import (
    SweepPlan,
    SweepRunner,
    _distribution,
    _paired_conclusion,
    compile_sweep_plan,
)


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    encoded = _stable_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(payload, sort_keys=False))


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class FactorialScale:
    id: str
    synthetic_train_count: int
    public_max_rows: int | None
    sweep: SweepPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "synthetic_train_count": self.synthetic_train_count,
            "public_max_rows": self.public_max_rows,
            "sweep": self.sweep.to_dict(),
        }


@dataclass(frozen=True)
class FactorialPlan:
    name: str
    root: str
    base_sweep: str
    baseline: str
    reference_scale: str
    heldout_count: int
    public_component_index: int
    scales: tuple[FactorialScale, ...]
    fingerprint: str
    raw_spec: dict[str, Any]

    @property
    def scale_ids(self) -> tuple[str, ...]:
        return tuple(scale.id for scale in self.scales)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "root": self.root,
            "base_sweep": self.base_sweep,
            "baseline": self.baseline,
            "reference_scale": self.reference_scale,
            "heldout_count": self.heldout_count,
            "public_component_index": self.public_component_index,
            "scales": [scale.to_dict() for scale in self.scales],
            "fingerprint": self.fingerprint,
        }


def compile_factorial_plan(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    python: str,
    compile_root: str | Path | None = None,
) -> FactorialPlan:
    """Compile one matched initialization sweep at each declared data scale."""

    repo = Path(repo_root).resolve()
    source = _resolve_path(repo, config_path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("factorial config root must be a mapping")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("factorial schema_version must be 1")
    name = str(raw.get("name") or "")
    if not _NAME.fullmatch(name):
        raise ValueError("factorial name must be a safe non-empty name")
    root = _resolve_path(
        repo,
        raw.get("output_root") or f"outputs/factorials/{name}",
    )
    generated_root = (
        _resolve_path(repo, compile_root)
        if compile_root is not None
        else root / "compiled"
    )
    base_sweep_path = _resolve_path(repo, raw.get("base_sweep") or "")
    if not base_sweep_path.is_file():
        raise ValueError(f"factorial base_sweep does not exist: {base_sweep_path}")
    base_sweep = yaml.safe_load(
        base_sweep_path.read_text(encoding="utf-8")
    ) or {}
    if not isinstance(base_sweep, dict):
        raise ValueError("factorial base_sweep root must be a mapping")

    heldout_count = int(raw.get("heldout_count", 0))
    public_component_index = int(raw.get("public_component_index", 1))
    if heldout_count <= 0:
        raise ValueError("factorial heldout_count must be positive")
    if public_component_index < 0:
        raise ValueError("factorial public_component_index must be non-negative")
    scales_raw = raw.get("scales")
    if not isinstance(scales_raw, list) or len(scales_raw) < 2:
        raise ValueError("factorial scales must contain at least two entries")

    seen_ids: set[str] = set()
    seen_sizes: set[tuple[int, int | None]] = set()
    scales: list[FactorialScale] = []
    expected_arms: set[str] | None = None
    expected_replicates: tuple[str, ...] | None = None
    expected_baseline: str | None = None
    for item in scales_raw:
        if not isinstance(item, dict):
            raise ValueError("every factorial scale must be a mapping")
        scale_id = str(item.get("id") or "")
        if not _NAME.fullmatch(scale_id) or scale_id in seen_ids:
            raise ValueError("factorial scale ids must be unique safe names")
        seen_ids.add(scale_id)
        synthetic_train_count = int(item.get("synthetic_train_count", 0))
        raw_max_rows = item.get("public_max_rows")
        public_max_rows = (
            None if raw_max_rows is None else int(raw_max_rows)
        )
        if synthetic_train_count <= 0:
            raise ValueError("synthetic_train_count must be positive")
        if public_max_rows is not None and public_max_rows <= 0:
            raise ValueError("public_max_rows must be positive or null")
        size = (synthetic_train_count, public_max_rows)
        if size in seen_sizes:
            raise ValueError("factorial data scales must be unique")
        seen_sizes.add(size)

        child = copy.deepcopy(base_sweep)
        child["name"] = f"{name}-{scale_id}"
        child["output_root"] = str(root / "scales" / scale_id)
        shared = list(child.get("shared_experiment_patches") or [])
        shared.extend(
            [
                {
                    "op": "replace",
                    "path": "/synthetic/train_count",
                    "value": synthetic_train_count,
                },
                {
                    "op": "replace",
                    "path": "/synthetic/heldout_count",
                    "value": heldout_count,
                },
                {
                    "op": "replace",
                    "path": (
                        f"/data/components/{public_component_index}"
                        "/hub/max_rows"
                    ),
                    "value": public_max_rows,
                },
                {
                    "op": "add",
                    "path": "/evaluation/wandb_tags/-",
                    "value": f"data-scale:{scale_id}",
                },
            ]
        )
        child["shared_experiment_patches"] = shared
        child_path = generated_root / scale_id / "sweep.yaml"
        _write_yaml(child_path, child)
        child_plan = compile_sweep_plan(
            child_path,
            repo_root=repo,
            python=python,
            compile_root=generated_root / scale_id / "runs",
        )
        arms = {variant.arm_id for variant in child_plan.variants}
        if expected_arms is None:
            expected_arms = arms
            expected_replicates = child_plan.replicates
            expected_baseline = child_plan.baseline
        elif (
            arms != expected_arms
            or child_plan.replicates != expected_replicates
            or child_plan.baseline != expected_baseline
        ):
            raise ValueError(
                "every factorial scale must compile identical arms, "
                "replicates, and baseline"
            )
        scales.append(
            FactorialScale(
                id=scale_id,
                synthetic_train_count=synthetic_train_count,
                public_max_rows=public_max_rows,
                sweep=child_plan,
            )
        )

    reference_scale = str(raw.get("reference_scale") or scales[-1].id)
    if reference_scale not in seen_ids:
        raise ValueError("factorial reference_scale must name a configured scale")
    for previous, current in zip(scales, scales[1:]):
        previous_public = (
            math.inf
            if previous.public_max_rows is None
            else previous.public_max_rows
        )
        current_public = (
            math.inf
            if current.public_max_rows is None
            else current.public_max_rows
        )
        if (
            current.synthetic_train_count < previous.synthetic_train_count
            or current_public < previous_public
        ):
            raise ValueError(
                "factorial scales must be ordered by non-decreasing "
                "synthetic and public data size"
            )
    reference = next(
        scale for scale in scales if scale.id == reference_scale
    )
    reference_public = (
        math.inf
        if reference.public_max_rows is None
        else reference.public_max_rows
    )
    if any(
        scale.synthetic_train_count > reference.synthetic_train_count
        or (
            scale.public_max_rows
            if scale.public_max_rows is not None
            else math.inf
        )
        > reference_public
        for scale in scales
    ):
        raise ValueError(
            "factorial reference_scale must be maximal on both data axes"
        )
    fingerprint = _fingerprint(
        {
            "spec": raw,
            "base_sweep": base_sweep,
            "children": [scale.sweep.fingerprint for scale in scales],
        }
    )
    return FactorialPlan(
        name=name,
        root=str(root),
        base_sweep=str(base_sweep_path),
        baseline=str(expected_baseline),
        reference_scale=reference_scale,
        heldout_count=heldout_count,
        public_component_index=public_component_index,
        scales=tuple(scales),
        fingerprint=fingerprint,
        raw_spec=raw,
    )


def _indexed_runs(comparison: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for record in comparison["runs"].values():
        key = (str(record["arm_id"]), str(record["replicate_id"]))
        if key in indexed:
            raise ValueError(f"duplicate factorial run key {key}")
        indexed[key] = record
    return indexed


def _mixture_signature(run_root: Path) -> dict[str, Any]:
    path = run_root / "artifacts" / "data" / "mixture" / "mixture_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing mixture manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    components = []
    for component in manifest.get("components", []):
        components.append(
            {
                key: component.get(key)
                for key in (
                    "name",
                    "weight",
                    "fold",
                    "rows",
                    "upstream_manifest_fingerprint",
                )
            }
        )
    return {
        "rows": int(manifest["rows"]),
        "weights": manifest["weights"],
        "components": components,
    }


def _actual_data_by_replicate(scale: FactorialScale) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for variant in scale.sweep.variants:
        signature = _mixture_signature(Path(variant.plan.root))
        expected = signatures.get(variant.replicate_id)
        if expected is None:
            signatures[variant.replicate_id] = signature
        elif signature != expected:
            raise ValueError(
                f"scale {scale.id!r} has mismatched training data in "
                f"replicate {variant.replicate_id!r}"
            )
    return signatures


def aggregate_factorial_results(plan: FactorialPlan) -> dict[str, Any]:
    """Aggregate scale-specific initialization effects and their interactions."""

    comparisons = {}
    for scale in plan.scales:
        path = Path(scale.sweep.root) / "comparison.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"scale {scale.id!r} has no completed comparison: {path}"
            )
        comparisons[scale.id] = json.loads(path.read_text(encoding="utf-8"))
    reference = comparisons[plan.reference_scale]
    reference_arms = set(reference["variants"])
    reference_replicates = tuple(reference["replicates"])
    reference_heldout = {
        replicate: artifacts["heldout"]
        for replicate, artifacts in reference[
            "matched_evaluation_artifacts_by_replicate"
        ].items()
    }
    for scale_id, comparison in comparisons.items():
        if (
            comparison["baseline"] != plan.baseline
            or set(comparison["variants"]) != reference_arms
            or tuple(comparison["replicates"]) != reference_replicates
        ):
            raise ValueError(
                f"scale {scale_id!r} does not match the reference design"
            )
        heldout = {
            replicate: artifacts["heldout"]
            for replicate, artifacts in comparison[
                "matched_evaluation_artifacts_by_replicate"
            ].items()
        }
        if heldout != reference_heldout:
            raise ValueError(
                f"scale {scale_id!r} changed heldout evaluation artifacts"
            )

    indexed = {
        scale_id: _indexed_runs(comparison)
        for scale_id, comparison in comparisons.items()
    }
    reference_runs = indexed[plan.reference_scale]
    interactions: dict[str, dict[str, Any]] = {}
    for arm_id in sorted(reference_arms - {plan.baseline}):
        by_scale = {}
        for scale in plan.scales:
            metric_names = sorted(
                indexed[scale.id][(arm_id, reference_replicates[0])][
                    "delta_vs_baseline"
                ]
            )
            metric_statistics = {}
            for metric in metric_names:
                values = [
                    float(
                        indexed[scale.id][(arm_id, replicate)][
                            "delta_vs_baseline"
                        ][metric]
                    )
                    - float(
                        reference_runs[(arm_id, replicate)][
                            "delta_vs_baseline"
                        ][metric]
                    )
                    for replicate in reference_replicates
                ]
                metric_statistics[metric] = _distribution(
                    values,
                    key=(
                        f"{plan.fingerprint}:{arm_id}:{scale.id}:"
                        f"{metric}:interaction"
                    ),
                )
            axis_names = sorted(
                set.intersection(
                    *[
                        set(
                            indexed[scale.id][(arm_id, replicate)][
                                "heldout_axis_delta_vs_baseline"
                            ]
                        )
                        & set(
                            reference_runs[(arm_id, replicate)][
                                "heldout_axis_delta_vs_baseline"
                            ]
                        )
                        for replicate in reference_replicates
                    ]
                )
            )
            axis_statistics = {}
            for axis in axis_names:
                values = [
                    float(
                        indexed[scale.id][(arm_id, replicate)][
                            "heldout_axis_delta_vs_baseline"
                        ][axis]
                    )
                    - float(
                        reference_runs[(arm_id, replicate)][
                            "heldout_axis_delta_vs_baseline"
                        ][axis]
                    )
                    for replicate in reference_replicates
                ]
                axis_statistics[axis] = _distribution(
                    values,
                    key=(
                        f"{plan.fingerprint}:{arm_id}:{scale.id}:"
                        f"{axis}:axis-interaction"
                    ),
                )
            by_scale[scale.id] = {
                "metric_statistics": metric_statistics,
                "heldout_axis_statistics": axis_statistics,
                "heldout_score_conclusion": (
                    "reference"
                    if scale.id == plan.reference_scale
                    else _paired_conclusion(
                        metric_statistics["heldout_score"]
                    )
                ),
            }
        interactions[arm_id] = by_scale

    scales = {}
    for scale in plan.scales:
        comparison = comparisons[scale.id]
        scales[scale.id] = {
            "sweep_fingerprint": scale.sweep.fingerprint,
            "comparison": str(
                Path(scale.sweep.root) / "comparison.json"
            ),
            "configured": {
                "synthetic_train_count_per_case": (
                    scale.synthetic_train_count
                ),
                "synthetic_heldout_count_per_case": plan.heldout_count,
                "public_max_rows": scale.public_max_rows,
            },
            "actual_training_data_by_replicate": (
                _actual_data_by_replicate(scale)
            ),
            "variants": {
                arm_id: {
                    "metrics": record["metrics"],
                    "metric_statistics": record["metric_statistics"],
                    "delta_vs_baseline": record["delta_vs_baseline"],
                    "paired_delta_statistics": record[
                        "paired_delta_statistics"
                    ],
                    "heldout_axis_delta_statistics": record[
                        "heldout_axis_delta_statistics"
                    ],
                    "heldout_score_conclusion": record[
                        "heldout_score_conclusion"
                    ],
                }
                for arm_id, record in comparison["variants"].items()
            },
        }
    result = {
        "schema_version": 1,
        "factorial": plan.name,
        "factorial_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "reference_scale": plan.reference_scale,
        "replicates": list(reference_replicates),
        "replicate_count": len(reference_replicates),
        "heldout_artifacts_by_replicate": reference_heldout,
        "confidence_interval": {
            "method": "paired_percentile_bootstrap",
            "level": 0.95,
            "resamples": 10_000,
            "interaction": (
                f"(arm-{plan.baseline} at scale) - "
                f"(arm-{plan.baseline} at reference scale)"
            ),
        },
        "scales": scales,
        "interactions": interactions,
    }
    root = Path(plan.root)
    _write_json(root / "factorial_comparison.json", result)
    _write_factorial_markdown(root / "factorial_comparison.md", result)
    return result


def _format_interval(statistics: dict[str, Any]) -> str:
    ci95 = statistics["ci95"]
    if ci95 is None:
        return "unavailable"
    return f"{ci95[0]:+.6f}, {ci95[1]:+.6f}"


def _write_factorial_markdown(
    path: Path,
    result: dict[str, Any],
) -> None:
    lines = [
        f"# {result['factorial']} initialization-by-data-scale factorial",
        "",
        f"Baseline: `{result['baseline']}`",
        "",
        f"Interaction reference scale: `{result['reference_scale']}`",
        "",
        f"Paired replicates: **{result['replicate_count']}**",
        "",
        "| Scale | Initialization | Heldout mean | Effect vs baseline [95% CI] | "
        "Interaction vs reference [95% CI] | Interaction conclusion |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for scale_id, scale in result["scales"].items():
        for arm_id, record in scale["variants"].items():
            heldout = record["metric_statistics"]["heldout_score"]
            effect = record["paired_delta_statistics"]["heldout_score"]
            if arm_id == result["baseline"]:
                interaction = "reference"
                conclusion = "reference"
            else:
                interaction_record = result["interactions"][arm_id][scale_id]
                interaction_stats = interaction_record["metric_statistics"][
                    "heldout_score"
                ]
                interaction = (
                    f"{interaction_stats['mean']:+.6f} "
                    f"[{_format_interval(interaction_stats)}]"
                )
                conclusion = interaction_record[
                    "heldout_score_conclusion"
                ]
            lines.append(
                f"| `{scale_id}` | `{arm_id}` | {heldout['mean']:.6f} | "
                f"{effect['mean']:+.6f} [{_format_interval(effect)}] | "
                f"{interaction} | `{conclusion}` |"
            )
    lines.extend(["", "## Data scale", ""])
    for scale_id, scale in result["scales"].items():
        configured = scale["configured"]
        totals = sorted(
            {
                int(values["rows"])
                for values in scale[
                    "actual_training_data_by_replicate"
                ].values()
            }
        )
        lines.append(
            f"- `{scale_id}`: synthetic count "
            f"{configured['synthetic_train_count_per_case']} per case, "
            f"public cap {configured['public_max_rows']}, "
            f"actual mixed rows {totals}."
        )
    _atomic_write(path, "\n".join(lines) + "\n")


class FactorialRunner:
    """Run each scale-specific matched sweep and publish interaction effects."""

    def __init__(self, plan: FactorialPlan, *, repo_root: str | Path):
        self.plan = plan
        self.repo_root = Path(repo_root).resolve()

    def run(
        self,
        *,
        dry_run: bool = False,
        resume: bool = True,
        scale_ids: set[str] | None = None,
        variant_ids: set[str] | None = None,
        replicate_ids: set[str] | None = None,
        start: str | None = None,
        stop: str | None = None,
    ) -> dict[str, Any]:
        known_scales = set(self.plan.scale_ids)
        selected_scales = (
            known_scales if scale_ids is None else set(scale_ids)
        )
        unknown = selected_scales - known_scales
        if unknown:
            raise ValueError(f"unknown factorial scales: {sorted(unknown)}")
        response: dict[str, Any] = {
            "dry_run": dry_run,
            "factorial": self.plan.name,
            "fingerprint": self.plan.fingerprint,
            "scales": [],
        }
        root = Path(self.plan.root)
        summary_path = root / "factorial_run_summary.json"
        if not dry_run:
            root.mkdir(parents=True, exist_ok=True)
            _write_json(root / "factorial_plan.json", self.plan.to_dict())
            _write_json(root / "factorial_spec.json", self.plan.raw_spec)
            response["status"] = "running"
            _write_json(summary_path, response)
        for scale in self.plan.scales:
            if scale.id not in selected_scales:
                continue
            try:
                child_result = SweepRunner(
                    scale.sweep,
                    repo_root=self.repo_root,
                ).run(
                    dry_run=dry_run,
                    resume=resume,
                    variant_ids=variant_ids,
                    replicate_ids=replicate_ids,
                    start=start,
                    stop=stop,
                )
            except Exception as error:
                response["scales"].append(
                    {
                        "scale": scale.id,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                if not dry_run:
                    response["status"] = "failed"
                    _write_json(summary_path, response)
                raise
            response["scales"].append(
                {
                    "scale": scale.id,
                    "status": "dry_run" if dry_run else "completed",
                    "result": child_result,
                }
            )
            if not dry_run:
                _write_json(summary_path, response)
        complete = (
            not dry_run
            and selected_scales == known_scales
            and variant_ids is None
            and replicate_ids is None
            and stop in {None, "evaluate"}
        )
        if complete:
            aggregate_factorial_results(self.plan)
            response["comparison"] = str(
                root / "factorial_comparison.json"
            )
        if not dry_run:
            response["status"] = "completed"
            _write_json(summary_path, response)
        return response

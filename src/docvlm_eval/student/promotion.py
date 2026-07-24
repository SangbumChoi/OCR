"""Materialize a statistically promoted sweep arm as a canonical recipe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..architecture import load_blueprint, validate_blueprint
from .architecture_sweep import (
    apply_compute_budget_gate,
    compile_architecture_sweep,
    compute_budget_report,
)
from .experiment import build_experiment_plan
from .sweep import (
    _write_comparison_markdown,
    aggregate_sweep_results,
    apply_json_patch,
    compile_sweep_plan,
)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    digest = hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_yaml(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, yaml.safe_dump(value, sort_keys=False))


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _patches(spec: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = spec.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _selected_variant(
    comparison: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if int(comparison.get("schema_version", 0)) < 4:
        raise ValueError("comparison schema does not contain promotion evidence")
    promotion = comparison.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError("comparison promotion evidence is missing")
    if promotion.get("status") != "promote":
        raise ValueError("comparison does not authorize recipe promotion")
    selected = promotion.get("selected_variants")
    if not isinstance(selected, list) or len(selected) != 1:
        raise ValueError(
            "recipe materialization requires exactly one promoted variant"
        )
    variant_id = str(selected[0])
    candidates = promotion.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("comparison promotion candidates are missing")
    candidate = candidates.get(variant_id)
    if (
        not isinstance(candidate, dict)
        or candidate.get("decision") != "promote"
    ):
        raise ValueError("selected variant lacks a promote decision")
    return variant_id, promotion


def materialize_promoted_recipe(
    sweep_path: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    python: str,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a canonical recipe from one fingerprint-matched promoted arm."""

    repo = Path(repo_root).resolve()
    source_sweep = _resolve(repo, sweep_path)
    if not source_sweep.is_file():
        raise FileNotFoundError(source_sweep)
    source_spec = _load_yaml(source_sweep)
    architecture_plan = None
    with tempfile.TemporaryDirectory(
        prefix="docvlm-promote-compile-"
    ) as compile_root:
        if "base_sweep" in source_spec:
            architecture_plan = compile_architecture_sweep(
                source_sweep,
                repo_root=repo,
                python=python,
                compile_root=compile_root,
            )
            plan = architecture_plan.sweep
            sweep_spec = plan.raw_spec
        else:
            plan = compile_sweep_plan(
                source_sweep,
                repo_root=repo,
                python=python,
                compile_root=compile_root,
            )
            sweep_spec = source_spec
    comparison_file = (
        _resolve(repo, comparison_path)
        if comparison_path is not None
        else Path(plan.root) / "comparison.json"
    )
    if not comparison_file.is_file():
        raise FileNotFoundError(comparison_file)
    comparison = json.loads(comparison_file.read_text(encoding="utf-8"))
    if not isinstance(comparison, dict):
        raise ValueError("comparison root must be a mapping")
    if comparison.get("sweep") != plan.name:
        raise ValueError("comparison names a different sweep")
    if comparison.get("sweep_fingerprint") != plan.fingerprint:
        raise ValueError(
            "comparison fingerprint does not match the current sweep"
        )
    recomputed_comparison = aggregate_sweep_results(plan)
    if architecture_plan is not None:
        recomputed_comparison = apply_compute_budget_gate(
            recomputed_comparison,
            compute_budget_report(architecture_plan),
        )
        canonical_comparison = Path(plan.root) / "comparison.json"
        _atomic_write_json(canonical_comparison, recomputed_comparison)
        _write_comparison_markdown(
            canonical_comparison.with_suffix(".md"),
            recomputed_comparison,
        )
    if _stable_json(comparison) != _stable_json(recomputed_comparison):
        raise ValueError(
            "comparison does not match evidence recomputed from run artifacts"
        )
    comparison = recomputed_comparison
    variant_id, promotion = _selected_variant(comparison)
    expected_contract = (
        None if plan.promotion is None else plan.promotion.to_dict()
    )
    if promotion.get("contract") != expected_contract:
        raise ValueError(
            "comparison promotion contract does not match the current sweep"
        )
    variants = sweep_spec.get("variants")
    if not isinstance(variants, list):
        raise ValueError("sweep variants are missing")
    variant = next(
        (
            item
            for item in variants
            if isinstance(item, dict) and str(item.get("id")) == variant_id
        ),
        None,
    )
    if variant is None or variant_id == plan.baseline:
        raise ValueError("promoted variant is not a valid non-baseline arm")

    base_experiment_path = _resolve(
        repo,
        str(sweep_spec.get("base_experiment") or ""),
    )
    base_experiment = _load_yaml(base_experiment_path)
    base_blueprint_path = _resolve(
        repo,
        str(base_experiment.get("blueprint") or ""),
    )
    base_blueprint = load_blueprint(base_blueprint_path)
    shared_experiment_patches = _patches(
        sweep_spec,
        "shared_experiment_patches",
    )
    shared_blueprint_patches = _patches(
        sweep_spec,
        "shared_blueprint_patches",
    )
    arm_experiment_patches = _patches(variant, "experiment_patches")
    arm_blueprint_patches = _patches(variant, "blueprint_patches")
    experiment = apply_json_patch(
        base_experiment,
        [*shared_experiment_patches, *arm_experiment_patches],
    )
    blueprint = apply_json_patch(
        base_blueprint,
        [*shared_blueprint_patches, *arm_blueprint_patches],
    )
    output = _resolve(repo, output_dir)
    recipe_name = f"{plan.name}-promoted-{variant_id}"
    experiment["name"] = recipe_name
    experiment["output_root"] = str(output / "run")
    experiment["blueprint"] = str(output / "blueprint.yaml")
    evaluation = experiment.setdefault("evaluation", {})
    tags = [str(tag) for tag in evaluation.get("wandb_tags") or []]
    evaluation["wandb_group"] = recipe_name
    evaluation["wandb_run"] = recipe_name
    evaluation["wandb_tags"] = list(
        dict.fromkeys(
            [
                *tags,
                "promoted-recipe",
                f"source-sweep:{plan.name}",
                f"source-variant:{variant_id}",
            ]
        )
    )
    estimates, blueprint_errors = validate_blueprint(blueprint)
    if blueprint_errors:
        raise ValueError(
            "promoted blueprint is invalid: "
            + "; ".join(blueprint_errors)
        )
    source_sweep_sha256 = _file_sha256(source_sweep)
    comparison_sha256 = _file_sha256(comparison_file)
    recipe_fingerprint = _fingerprint(
        {
            "source_sweep_fingerprint": plan.fingerprint,
            "source_sweep_sha256": source_sweep_sha256,
            "comparison_sha256": comparison_sha256,
            "selected_variant": variant_id,
            "experiment": experiment,
            "blueprint": blueprint,
        }
    )
    manifest_path = output / "promotion_manifest.json"
    if output.exists() and any(output.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(
                f"promotion output is non-empty and unmanaged: {output}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("recipe_fingerprint") != recipe_fingerprint:
            raise FileExistsError(
                "promotion output contains a different recipe"
            )
        for name in ("experiment.yaml", "blueprint.yaml"):
            path = output / name
            expected = existing.get("artifacts", {}).get(name)
            if not path.is_file() or _file_sha256(path) != expected:
                raise ValueError(
                    "existing promoted recipe artifacts fail integrity checks"
                )
        return existing

    created_output = not output.exists()
    output.mkdir(parents=True, exist_ok=True)
    experiment_path = output / "experiment.yaml"
    blueprint_path = output / "blueprint.yaml"
    try:
        _atomic_write_yaml(blueprint_path, blueprint)
        _atomic_write_yaml(experiment_path, experiment)
        experiment_plan = build_experiment_plan(
            experiment_path,
            repo_root=repo,
            python=python,
        )
        manifest = {
            "schema_version": 1,
            "recipe_fingerprint": recipe_fingerprint,
            "source": {
                "sweep": str(source_sweep),
                "sweep_kind": (
                    "architecture"
                    if architecture_plan is not None
                    else "generic"
                ),
                "sweep_sha256": source_sweep_sha256,
                "sweep_fingerprint": plan.fingerprint,
                "comparison": str(comparison_file),
                "comparison_sha256": comparison_sha256,
                "comparison_schema_version": comparison["schema_version"],
                "baseline": plan.baseline,
                "selected_variant": variant_id,
            },
            "promotion": promotion,
            "patches": {
                "shared_experiment": shared_experiment_patches,
                "shared_blueprint": shared_blueprint_patches,
                "arm_experiment": arm_experiment_patches,
                "arm_blueprint": arm_blueprint_patches,
                "replicate_patches_included": False,
            },
            "artifacts": {
                "experiment.yaml": _file_sha256(experiment_path),
                "blueprint.yaml": _file_sha256(blueprint_path),
            },
            "validated": {
                "experiment_fingerprint": experiment_plan.fingerprint,
                "parameter_estimates": estimates,
            },
        }
        _atomic_write_json(manifest_path, manifest)
        return manifest
    except Exception:
        if created_output:
            shutil.rmtree(output)
        else:
            experiment_path.unlink(missing_ok=True)
            blueprint_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
        raise

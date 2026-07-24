import json
import sys
from pathlib import Path

import pytest
import yaml

import docvlm_eval.student.promotion as promotion_module
from docvlm_eval.student.architecture_sweep import (
    apply_compute_budget_gate,
    compile_architecture_sweep,
)
from docvlm_eval.student.promotion import materialize_promoted_recipe
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def _promotion_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sweep = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sweep_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    sweep["output_root"] = str(tmp_path / "sweep-output")
    sweep["replicates"][0]["experiment_patches"][0]["value"] = 999
    sweep["replicates"][1]["experiment_patches"][0]["value"] = 1999
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(sweep, sort_keys=False),
        encoding="utf-8",
    )
    plan = compile_sweep_plan(
        sweep_path,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )
    selected = "no_sequence_distillation"
    comparison = {
        "schema_version": 4,
        "sweep": plan.name,
        "sweep_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "promotion": {
            "status": "promote",
            "selected_variants": [selected],
            "baseline_retained": False,
            "contract": plan.promotion.to_dict(),
            "multiple_comparisons": {
                "method": "bonferroni_one_sided_percentile_bootstrap",
            },
            "candidates": {
                selected: {
                    "decision": "promote",
                    "evidence": {
                        "simultaneous_lower_bound": 0.1,
                    },
                }
            },
        },
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    return sweep_path, comparison_path, plan


def test_materialize_promoted_recipe_excludes_replicate_seed_patches(
    tmp_path,
    monkeypatch,
):
    sweep_path, comparison_path, plan = _promotion_fixture(tmp_path)
    output = tmp_path / "promoted"
    monkeypatch.setattr(
        promotion_module,
        "aggregate_sweep_results",
        lambda _: json.loads(comparison_path.read_text(encoding="utf-8")),
    )

    manifest = materialize_promoted_recipe(
        sweep_path,
        output,
        repo_root=ROOT,
        python=sys.executable,
        comparison_path=comparison_path,
    )

    experiment = yaml.safe_load(
        (output / "experiment.yaml").read_text(encoding="utf-8")
    )
    assert experiment["sequence_teacher"]["enabled"] is False
    assert experiment["initialization"]["seed"] == 5
    assert experiment["initialization"]["seed"] not in {999, 1999}
    assert experiment["blueprint"] == str(output / "blueprint.yaml")
    assert experiment["output_root"] == str(output / "run")
    assert "promoted-recipe" in experiment["evaluation"]["wandb_tags"]
    assert manifest["source"]["sweep_fingerprint"] == plan.fingerprint
    assert manifest["source"]["selected_variant"] == (
        "no_sequence_distillation"
    )
    assert manifest["patches"]["replicate_patches_included"] is False
    assert manifest["validated"]["parameter_estimates"]["total"] < 1_000_000_000

    repeated = materialize_promoted_recipe(
        sweep_path,
        output,
        repo_root=ROOT,
        python=sys.executable,
        comparison_path=comparison_path,
    )
    assert repeated["recipe_fingerprint"] == manifest["recipe_fingerprint"]


def test_materialize_promoted_architecture_profile(tmp_path, monkeypatch):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_architecture_compute_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "architecture-output")
    sweep_path = tmp_path / "architecture-sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    architecture_plan = compile_architecture_sweep(
        sweep_path,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )
    plan = architecture_plan.sweep
    selected = "r448_l32"
    comparison = {
        "schema_version": 4,
        "sweep": plan.name,
        "sweep_fingerprint": plan.fingerprint,
        "baseline": plan.baseline,
        "promotion": {
            "status": "promote",
            "selected_variants": [selected],
            "baseline_retained": False,
            "contract": plan.promotion.to_dict(),
            "multiple_comparisons": {
                "method": "bonferroni_one_sided_percentile_bootstrap",
            },
            "pareto_frontier": [selected],
            "candidates": {
                selected: {
                    "decision": "promote",
                    "evidence": {},
                }
            },
        },
    }
    comparison = apply_compute_budget_gate(
        comparison,
        {"status": "pass"},
    )
    comparison_path = tmp_path / "architecture-comparison.json"
    comparison_path.write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        promotion_module,
        "aggregate_sweep_results",
        lambda _: {
            key: value
            for key, value in comparison.items()
            if key != "promotion"
        }
        | {
            "promotion": {
                key: value
                for key, value in comparison["promotion"].items()
                if key != "external_gates"
            }
        },
    )
    monkeypatch.setattr(
        promotion_module,
        "compute_budget_report",
        lambda _: {"status": "pass"},
    )
    monkeypatch.setattr(
        promotion_module,
        "_write_comparison_markdown",
        lambda *_: None,
    )

    output = tmp_path / "promoted-architecture"
    manifest = materialize_promoted_recipe(
        sweep_path,
        output,
        repo_root=ROOT,
        python=sys.executable,
        comparison_path=comparison_path,
    )

    blueprint = yaml.safe_load(
        (output / "blueprint.yaml").read_text(encoding="utf-8")
    )
    assert blueprint["student"]["vision"]["image_size"] == 448
    assert blueprint["student"]["connector"]["latent_tokens"] == 32
    assert blueprint["training"]["pretraining"]["input_pipeline"][
        "visual_canvas_mode"
    ] == "fixed_square"
    assert manifest["source"]["sweep_kind"] == "architecture"
    assert manifest["source"]["selected_variant"] == selected
    assert manifest["promotion"]["external_gates"] == {
        "architecture_compute_budget": "pass"
    }


def test_materialize_promoted_recipe_rejects_stale_or_tampered_evidence(
    tmp_path,
    monkeypatch,
):
    sweep_path, comparison_path, _ = _promotion_fixture(tmp_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["sweep_fingerprint"] = "sha256:stale"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        materialize_promoted_recipe(
            sweep_path,
            tmp_path / "stale",
            repo_root=ROOT,
            python=sys.executable,
            comparison_path=comparison_path,
        )

    _, valid_comparison, _ = _promotion_fixture(tmp_path / "valid")
    monkeypatch.setattr(
        promotion_module,
        "aggregate_sweep_results",
        lambda _: json.loads(valid_comparison.read_text(encoding="utf-8")),
    )
    output = tmp_path / "promoted"
    materialize_promoted_recipe(
        tmp_path / "valid" / "sweep.yaml",
        output,
        repo_root=ROOT,
        python=sys.executable,
        comparison_path=valid_comparison,
    )
    (output / "experiment.yaml").write_text(
        "tampered: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integrity"):
        materialize_promoted_recipe(
            tmp_path / "valid" / "sweep.yaml",
            output,
            repo_root=ROOT,
            python=sys.executable,
            comparison_path=valid_comparison,
        )


def test_materialize_promoted_recipe_requires_one_authorized_arm(
    tmp_path,
    monkeypatch,
):
    sweep_path, comparison_path, _ = _promotion_fixture(tmp_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["promotion"]["status"] = "retain_baseline"
    comparison["promotion"]["selected_variants"] = []
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    monkeypatch.setattr(
        promotion_module,
        "aggregate_sweep_results",
        lambda _: json.loads(comparison_path.read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="does not authorize"):
        materialize_promoted_recipe(
            sweep_path,
            tmp_path / "not-promoted",
            repo_root=ROOT,
            python=sys.executable,
            comparison_path=comparison_path,
        )


def test_materialize_promoted_recipe_rejects_edited_comparison(
    tmp_path,
    monkeypatch,
):
    sweep_path, comparison_path, _ = _promotion_fixture(tmp_path)
    recomputed = json.loads(comparison_path.read_text(encoding="utf-8"))
    supplied = json.loads(comparison_path.read_text(encoding="utf-8"))
    supplied["promotion"]["candidates"]["no_sequence_distillation"][
        "evidence"
    ]["simultaneous_lower_bound"] = 0.9
    comparison_path.write_text(json.dumps(supplied), encoding="utf-8")
    monkeypatch.setattr(
        promotion_module,
        "aggregate_sweep_results",
        lambda _: recomputed,
    )

    with pytest.raises(ValueError, match="recomputed from run artifacts"):
        materialize_promoted_recipe(
            sweep_path,
            tmp_path / "edited",
            repo_root=ROOT,
            python=sys.executable,
            comparison_path=comparison_path,
        )

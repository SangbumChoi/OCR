import json
import sys
from pathlib import Path

import pytest
import yaml

from docvlm_eval.student.architecture_sweep import (
    apply_compute_budget_gate,
    compile_architecture_sweep,
    compute_budget_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _compile(tmp_path: Path):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_architecture_compute_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "architecture-sweep.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return compile_architecture_sweep(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def _compile_mixer(tmp_path: Path):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_language_mixer_compute_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "mixer-output")
    config = tmp_path / "language-mixer-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return compile_architecture_sweep(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "mixer-compiled",
    )


def test_architecture_sweep_compiles_paired_compute_matched_profiles(tmp_path):
    plan = _compile(tmp_path)

    assert plan.baseline == "r896_l64"
    assert plan.sweep.promotion is not None
    assert plan.sweep.promotion.to_dict()["mode"] == "pareto"
    assert len(plan.profiles) == 5
    assert len(plan.sweep.variants) == 15
    assert plan.budgets.pretrain > plan.budgets.sft > plan.budgets.rlvr
    assert (
        plan.sweep.control_values_by_replicate["seed_0"][
            "experiment:/runtime/visual_backend_benchmark/enabled"
        ]
        is False
    )
    for variant in plan.sweep.variants:
        blueprint = variant.plan.resolved_blueprint
        pretrain = blueprint["training"]["pretraining"]
        sft = blueprint["training"]["posttraining"]["sft"]["optimizer"]
        rlvr = blueprint["training"]["posttraining"]["rlvr"]["optimizer"]
        assert pretrain["optimizer"]["total_student_flops"] == (
            plan.budgets.pretrain
        )
        assert pretrain["optimizer"]["schedule_unit"] == "student_flops"
        assert pretrain["optimizer"]["stop_at_student_flops"] is True
        assert pretrain["curriculum"]["unit"] == (
            "training_compute_fraction"
        )
        assert pretrain["input_pipeline"]["visual_canvas_mode"] == (
            "fixed_square"
        )
        assert pretrain["input_pipeline"]["visual_sequence_mode"] == "dense"
        assert pretrain["input_pipeline"]["aspect_ratio_bucketing"] is False
        assert sft["total_student_flops"] == plan.budgets.sft
        assert sft["schedule_unit"] == "student_flops"
        assert rlvr["total_student_flops"] == plan.budgets.rlvr
        assert rlvr["max_steps"] is None
        tags = variant.plan.raw_spec["evaluation"]["wandb_tags"]
        assert "compute-matched-architecture" in tags
        assert (
            variant.plan.raw_spec["runtime"]["visual_backend_benchmark"][
                "enabled"
            ]
            is False
        )
        assert "visual_backend_benchmark" not in variant.plan.stage_names
        assert variant.decision_metrics[
            "training_flops_per_sample"
        ] > 0


def test_architecture_sweep_profiles_have_expected_compute_order(tmp_path):
    plan = _compile(tmp_path)
    profiles = {profile.id: profile for profile in plan.profiles}

    assert (
        profiles["r896_l64"].compute["training_flops_per_sample"]
        > profiles["r672_l48"].compute["training_flops_per_sample"]
        > profiles["r448_l32"].compute["training_flops_per_sample"]
    )
    assert (
        profiles["r896_l64"].compute["training_flops_per_sample"]
        > profiles["r896_l32"].compute["training_flops_per_sample"]
    )


def test_language_mixer_sweep_compiles_hybrid_profiles(tmp_path):
    plan = _compile_mixer(tmp_path)
    profiles = {profile.id: profile for profile in plan.profiles}

    assert plan.baseline == "all_attention"
    assert plan.sweep.promotion is not None
    assert plan.sweep.promotion.to_dict()["selection_order"][1] == (
        "decision.rlvr_peak_kv_cache_bytes_bfloat16"
    )
    assert len(plan.profiles) == 4
    assert len(plan.sweep.variants) == 12
    assert profiles["all_attention"].blueprint_patches == ()
    assert (
        profiles["lfm_ratio_k3"].compute["parameter_counts"]["total"]
        < 1_000_000_000
    )
    assert (
        profiles["lfm_ratio_k3"].compute[
            "training_flops_per_sample"
        ]
        != profiles["all_attention"].compute[
            "training_flops_per_sample"
        ]
    )
    for variant in plan.sweep.variants:
        layer_indices = variant.plan.resolved_blueprint["student"][
            "language"
        ]["full_attention_layers"]
        if variant.id.startswith("all_attention"):
            assert layer_indices is None
        elif variant.id.startswith("alternating_k3"):
            assert len(layer_indices) == 12
        else:
            assert len(layer_indices) == 8
        assert (
            variant.plan.resolved_blueprint["training"]["pretraining"][
                "optimizer"
            ]["total_student_flops"]
            == plan.budgets.pretrain
        )
        assert "architecture-profile:" in " ".join(
            variant.plan.raw_spec["evaluation"]["wandb_tags"]
        )
        assert variant.decision_metrics[
            "rlvr_peak_kv_cache_bytes_bfloat16"
        ] > 0


def test_architecture_sweep_does_not_inherit_base_scalar_promotion(tmp_path):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_architecture_compute_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw.pop("promotion")
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "no-promotion-architecture-sweep.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    plan = compile_architecture_sweep(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert plan.sweep.promotion is None


def test_architecture_profile_patches_cannot_override_compute_contract(
    tmp_path,
):
    raw = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "sub1b_language_mixer_compute_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["profiles"][1]["blueprint_patches"] = [
        {
            "op": "replace",
            "path": (
                "/training/pretraining/optimizer/"
                "total_student_flops"
            ),
            "value": 1,
        }
    ]
    config = tmp_path / "invalid-profile-patch.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must target /student"):
        compile_architecture_sweep(
            config,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "invalid-compiled",
        )


def test_compute_budget_report_rejects_excessive_overshoot(tmp_path):
    plan = _compile(tmp_path)
    for variant in plan.sweep.variants:
        for stage, budget in plan.budgets.to_dict().items():
            if stage.startswith("warmup_"):
                continue
            output = Path(variant.plan.root) / "artifacts" / stage
            checkpoint = output / "checkpoints" / "step-00000001"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "student_flops_seen": int(
                            budget
                            * (
                                1.03
                                if variant.id.startswith("r448_l32")
                                and stage == "rlvr"
                                else 1.001
                            )
                        )
                    }
                ),
                encoding="utf-8",
            )
            (output / "latest_checkpoint.txt").write_text(
                str(checkpoint) + "\n",
                encoding="utf-8",
            )

    report = compute_budget_report(plan)

    assert report["status"] == "fail"
    assert report["runs"]["r448_l32--seed_0"]["rlvr"]["status"] == "fail"
    assert report["runs"]["r896_l64--seed_0"]["pretrain"]["status"] == "pass"


def test_compute_budget_gate_revokes_a_selected_architecture():
    comparison = {
        "promotion": {
            "status": "promote",
            "selected_variants": ["smaller"],
            "baseline_retained": False,
            "candidates": {
                "smaller": {"decision": "promote"},
            },
        }
    }

    gated = apply_compute_budget_gate(
        comparison,
        {"status": "fail"},
    )

    assert gated["promotion"]["status"] == "retain_baseline"
    assert gated["promotion"]["selected_variants"] == []
    assert gated["promotion"]["baseline_retained"] is True
    assert gated["promotion"]["candidates"]["smaller"]["decision"] == (
        "reject_external_gate"
    )
    assert gated["promotion"]["external_gates"] == {
        "architecture_compute_budget": "fail"
    }
    assert comparison["promotion"]["status"] == "promote"

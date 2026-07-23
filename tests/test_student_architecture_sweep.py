import json
import sys
from pathlib import Path

import yaml

from docvlm_eval.student.architecture_sweep import (
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


def test_architecture_sweep_compiles_paired_compute_matched_profiles(tmp_path):
    plan = _compile(tmp_path)

    assert plan.baseline == "r896_l64"
    assert len(plan.profiles) == 5
    assert len(plan.sweep.variants) == 15
    assert plan.budgets.pretrain > plan.budgets.sft > plan.budgets.rlvr
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

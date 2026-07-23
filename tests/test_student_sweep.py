import json
import sys
from pathlib import Path

import pytest
import yaml

from docvlm_eval.student.sweep import (
    SweepRunner,
    aggregate_sweep_results,
    apply_json_patch,
    compile_sweep_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_sweep(tmp_path: Path, *, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sweep_tiny.yaml").read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _compile_tiny(tmp_path: Path):
    return compile_sweep_plan(
        _tiny_sweep(tmp_path),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def test_json_patch_supports_mapping_and_list_operations_without_mutation():
    source = {"model": {"layers": [1, 2], "width": 64}}

    patched = apply_json_patch(
        source,
        [
            {"op": "replace", "path": "/model/width", "value": 96},
            {"op": "add", "path": "/model/layers/-", "value": 3},
            {"op": "remove", "path": "/model/layers/0"},
        ],
    )

    assert patched == {"model": {"layers": [2, 3], "width": 96}}
    assert source == {"model": {"layers": [1, 2], "width": 64}}
    with pytest.raises(ValueError, match="does not exist"):
        apply_json_patch(source, [{"op": "replace", "path": "/missing", "value": 1}])


def test_tiny_sweep_compiles_matched_independent_experiments(tmp_path):
    plan = _compile_tiny(tmp_path)

    assert plan.baseline == "baseline"
    assert plan.replicates == ("seed_0", "seed_1")
    assert [(variant.arm_id, variant.replicate_id) for variant in plan.variants] == [
        ("baseline", "seed_0"),
        ("no_sequence_distillation", "seed_0"),
        ("baseline", "seed_1"),
        ("no_sequence_distillation", "seed_1"),
    ]
    baseline = plan.variants[0]
    ablation = plan.variants[1]
    assert baseline.parameters["total"] == 587_017
    assert ablation.parameters["total"] == 587_017
    assert "generate_teacher_predictions" in baseline.plan.stage_names
    assert "generate_teacher_predictions" not in ablation.plan.stage_names
    assert Path(baseline.plan.root) != Path(ablation.plan.root)
    evaluation = baseline.plan.raw_spec["evaluation"]
    assert evaluation["wandb_group"] == "docvlm-tiny-sweep"
    assert evaluation["wandb_run"] == "docvlm-tiny-sweep--baseline--seed_0"
    assert "variant:baseline" in evaluation["wandb_tags"]
    assert "replicate:seed_0" in evaluation["wandb_tags"]
    assert (
        plan.control_values_by_replicate["seed_0"][
            "experiment:/pretraining/max_steps"
        ]
        == 1
    )
    assert (
        plan.control_values_by_replicate["seed_1"]["experiment:/initialization/seed"]
        == 105
    )
    initialize = next(
        stage for stage in plan.variants[2].plan.stages if stage.name == "initialize_student"
    )
    assert initialize.command[initialize.command.index("--seed") + 1] == "105"

    other = compile_sweep_plan(
        _tiny_sweep(tmp_path / "second"),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "other-compiled",
    )
    assert [variant.fingerprint for variant in plan.variants] != [
        variant.fingerprint for variant in other.variants
    ]


def test_initialization_sweep_compiles_pinned_transfer_arms(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_initialization_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "initialization-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 15
    assert plan.baseline == "random"
    selective = next(
        variant
        for variant in plan.variants
        if variant.arm_id == "selective" and variant.replicate_id == "seed_0"
    )
    assert selective.plan.raw_spec["initialization"]["arm"] == "I4_selective"
    assert selective.plan.stage_names[:2] == [
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
    ]
    initialize = next(
        stage
        for stage in selective.plan.stages
        if stage.name == "initialize_student"
    )
    assert initialize.dependencies == (
        "train_tokenizer",
        "acquire_vision_checkpoint",
        "acquire_language_checkpoint",
    )


def test_pretraining_loss_sweep_compiles_active_leave_one_out_arms(tmp_path):
    from docvlm_eval.student.pretrain import (
        PretrainConfig,
        pretraining_supervision_contract,
    )

    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_pretraining_loss_sweep.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "loss-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 15
    assert plan.baseline == "full_objective"
    expected_removed = {
        "no_autoregressive": "autoregressive",
        "no_region_text_contrastive": "region_text_contrastive",
        "no_box_regression": "box_regression",
        "no_orientation": "orientation",
    }
    for variant in plan.variants:
        training = PretrainConfig.from_blueprint(
            variant.plan.resolved_blueprint,
            tmp_path / variant.id,
        )
        contract = pretraining_supervision_contract(
            training,
            has_online_teacher=False,
        )
        assert contract["online_teacher_losses"] == []
        assert all(stage["active_losses"] for stage in contract["stages"])
        if variant.arm_id in expected_removed:
            removed = expected_removed[variant.arm_id]
            assert all(
                removed not in stage["active_losses"]
                for stage in contract["stages"]
            )
        assert "loss-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_sft_target_sweep_compiles_three_sft_only_targets(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sft_target_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "sft-target-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "evidence_linked"
    expected_modes = {
        "evidence_linked": "evidence_linked",
        "free_rationale": "free_rationale",
        "answer_only": "answer_only",
    }
    for variant in plan.variants:
        assert (
            variant.plan.raw_spec["posttraining"]["sft"]["target_mode"]
            == expected_modes[variant.arm_id]
        )
        assert "rlvr" not in variant.plan.stage_names
        evaluate = next(
            stage for stage in variant.plan.stages if stage.name == "evaluate"
        )
        assert "@student:sft" in evaluate.command
        assert "sft-target-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_rlvr_reward_sweep_compiles_sft_and_reward_controls(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_rlvr_reward_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "rlvr-reward-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "full_reward"
    for variant in plan.variants:
        evaluate = next(
            stage for stage in variant.plan.stages if stage.name == "evaluate"
        )
        if variant.arm_id == "sft_only":
            assert "rlvr" not in variant.plan.stage_names
            assert "@student:sft" in evaluate.command
        else:
            assert "rlvr" in variant.plan.stage_names
            assert "@student:rlvr" in evaluate.command
        reward_mix = variant.plan.resolved_blueprint["training"][
            "posttraining"
        ]["rlvr"]["reward_mix"]
        if variant.arm_id == "correctness_only":
            assert reward_mix["answer_correctness"] == 1.0
            assert all(
                weight == 0.0
                for name, weight in reward_mix.items()
                if name != "answer_correctness"
            )
        assert "rlvr-reward-ablation" in variant.plan.raw_spec["evaluation"][
            "wandb_tags"
        ]


def test_sequence_teacher_sweep_compiles_pinned_fixed_dose_arms(tmp_path):
    raw = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sequence_teacher_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["output_root"] = str(tmp_path / "output")
    config = tmp_path / "sequence-teacher-sweep.yaml"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert len(plan.variants) == 9
    assert plan.baseline == "gold_only"
    expected = {
        "lfm": (
            "lfm2_5-vl-1.6b",
            "919fde3d022e3f90a4716006f993938ee8c2eb97",
        ),
        "qwen": (
            "qwen3_5-0.8b",
            "2fc06364715b967f1860aea9cf38778875588b17",
        ),
    }
    for variant in plan.variants:
        teacher = variant.plan.raw_spec["sequence_teacher"]
        assert teacher["max_requests"] == 4096
        assert teacher["accepted_target_count"] == 400
        assert (
            variant.plan.raw_spec["tokenizer"]["include_teacher_targets"]
            is False
        )
        if variant.arm_id == "gold_only":
            assert "generate_teacher_predictions" not in variant.plan.stage_names
            continue
        assert (teacher["model"], teacher["revision"]) == expected[
            variant.arm_id
        ]
        generate = next(
            stage
            for stage in variant.plan.stages
            if stage.name == "generate_teacher_predictions"
        )
        assert generate.command[generate.command.index("--model-revision") + 1] == (
            teacher["revision"]
        )
        apply = next(
            stage
            for stage in variant.plan.stages
            if stage.name == "apply_teacher_targets"
        )
        assert (
            apply.command[apply.command.index("--accepted-target-count") + 1]
            == "400"
        )


def test_sweep_fingerprint_ignores_only_the_temporary_compile_location(tmp_path):
    config = _tiny_sweep(tmp_path)
    first = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-a",
    )
    second = compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-b",
    )

    assert first.fingerprint == second.fingerprint
    assert [variant.fingerprint for variant in first.variants] == [
        variant.fingerprint for variant in second.variants
    ]


def test_sweep_without_replicates_keeps_single_run_per_arm_contract(tmp_path):
    def mutate(raw):
        raw.pop("replicates")
        raw.pop("replicate_controls")

    plan = compile_sweep_plan(
        _tiny_sweep(tmp_path, mutate=mutate),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )

    assert plan.replicates == ("default",)
    assert [variant.id for variant in plan.variants] == [
        "baseline",
        "no_sequence_distillation",
    ]
    assert plan.variants[0].plan.raw_spec["evaluation"]["wandb_run"] == (
        "docvlm-tiny-sweep--baseline"
    )


def test_sweep_rejects_a_variant_that_changes_a_matched_control(tmp_path):
    def mutate(raw):
        raw["variants"][1]["experiment_patches"].append(
            {"op": "replace", "path": "/evaluation/seed", "value": 99}
        )

    with pytest.raises(ValueError, match="violates matched controls"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path, mutate=mutate),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled",
        )


def test_sweep_rejects_undeclared_or_incomplete_replicate_dimensions(tmp_path):
    def undeclared(raw):
        raw["replicates"][0]["experiment_patches"].append(
            {"op": "replace", "path": "/synthetic/count", "value": 2}
        )

    with pytest.raises(ValueError, match="undeclared replicate controls"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "undeclared", mutate=undeclared),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled-undeclared",
        )

    def incomplete(raw):
        raw["replicates"][0]["experiment_patches"].pop()

    with pytest.raises(ValueError, match="does not set every replicate control"):
        compile_sweep_plan(
            _tiny_sweep(tmp_path / "incomplete", mutate=incomplete),
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled-incomplete",
        )


def test_full_sweep_compiles_loss_sft_and_reward_ablation_contracts(tmp_path):
    plan = compile_sweep_plan(
        ROOT / "configs" / "sub1b_sweep.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )
    variants = {
        variant.arm_id: variant
        for variant in plan.variants
        if variant.replicate_id == "seed_0"
    }

    assert set(variants) == {
        "baseline",
        "no_sequence_distillation",
        "no_spatial_auxiliary",
        "sft_answer_only",
        "correctness_only_rlvr",
        "no_supervised_replay",
    }
    assert len(plan.variants) == 18
    no_spatial = variants["no_spatial_auxiliary"].plan.resolved_blueprint
    assert no_spatial["training"]["pretraining"]["losses"]["box_regression"] == 0.0
    assert (
        no_spatial["training"]["pretraining"]["curriculum"]["stages"][0][
            "loss_weights"
        ]["box_regression"]
        == 0.0
    )
    assert (
        variants["sft_answer_only"].plan.raw_spec["posttraining"]["sft"]["target_mode"]
        == "answer_only"
    )
    reward_mix = variants["correctness_only_rlvr"].plan.resolved_blueprint["training"][
        "posttraining"
    ]["rlvr"]["reward_mix"]
    assert reward_mix["answer_correctness"] == 1.0
    assert sum(reward_mix.values()) == pytest.approx(1.0)
    no_replay = variants["no_supervised_replay"].plan.resolved_blueprint[
        "training"
    ]["posttraining"]["rlvr"]["supervised_replay"]
    assert no_replay == {"every_steps": 0, "loss_coefficient": 0.0}


def _comparison(score: float, milliseconds: float) -> dict:
    def summary(split: str, split_score: float):
        return {
            "split": split,
            "dataset_size": 2,
            "n_samples": 2,
            "score": split_score,
            "reward": split_score / 2,
            "valid_structure_fraction": 0.75,
            "answer_rate": 1.0,
            "elapsed_seconds": milliseconds * 2 / 1000,
            "milliseconds_per_sample": milliseconds,
            "by_answer_type": {
                "chart": {
                    "n": 2,
                    "score": split_score,
                    "reward": split_score / 2,
                    "valid_structure_fraction": 0.75,
                    "answer_rate": 1.0,
                }
            },
            "by_source": {},
            "by_language": {},
            "by_robustness_axis": {
                "document_family": {
                    "chart": {
                        "n": 2,
                        "score": split_score,
                        "reward": split_score / 2,
                        "valid_structure_fraction": 0.75,
                        "answer_rate": 1.0,
                    }
                }
            },
            "reward_components": {},
        }

    heldout = summary("heldout", score)
    train = summary("train", score + 0.1)
    return {
        "splits": {"train": train, "heldout": heldout},
        "train_minus_heldout": {
            "headline": {
                "score": 0.1,
                "reward": 0.05,
                "valid_structure_fraction": 0.0,
                "answer_rate": 0.0,
            },
            "by_answer_type": {},
            "by_robustness_axis": {
                "document_family": {
                    "chart": {
                        "score": 0.1,
                        "reward": 0.05,
                        "valid_structure_fraction": 0.0,
                        "answer_rate": 0.0,
                    }
                }
            },
        },
    }


def test_sweep_aggregates_baseline_deltas_ranking_and_markdown(tmp_path):
    plan = _compile_tiny(tmp_path)
    scores = {
        ("baseline", "seed_0"): (0.2, 20.0),
        ("baseline", "seed_1"): (0.3, 22.0),
        ("no_sequence_distillation", "seed_0"): (0.35, 24.0),
        ("no_sequence_distillation", "seed_1"): (0.4, 25.0),
    }
    for variant in plan.variants:
        samples = Path(variant.plan.root) / "artifacts" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        image = samples / "document.png"
        image.write_bytes(f"image-{variant.replicate_id}".encode())
        row = {
            "sample_id": "sample-1",
            "image_path": str(image),
            "question": "Read the chart.",
            "answers": ["42"],
            "answer_type": "chart",
            "metric": "relaxed_acc",
            "meta": {"language": "en"},
        }
        for split in ("train", "heldout"):
            (samples / f"{split}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
        path = Path(variant.plan.root) / "artifacts" / "evaluation" / "comparison.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        heldout_score, milliseconds = scores[
            (variant.arm_id, variant.replicate_id)
        ]
        path.write_text(
            json.dumps(_comparison(heldout_score, milliseconds)),
            encoding="utf-8",
        )
        for split, split_score in (
            ("train", heldout_score + 0.1),
            ("heldout", heldout_score),
        ):
            split_root = path.parent / split
            split_root.mkdir()
            eval_row = {
                "sample_id": "sample-1",
                "score": split_score,
                "answer": "42",
                "answer_type": "chart",
                "confidence": 0.8,
                "meta": {},
                "reward_components": {},
                "applicable_rewards": [],
            }
            (split_root / "per_sample.jsonl").write_text(
                json.dumps(eval_row) + "\n",
                encoding="utf-8",
            )

    result = aggregate_sweep_results(plan)

    assert result["ranking"] == ["no_sequence_distillation", "baseline"]
    assert result["variants"]["no_sequence_distillation"]["delta_vs_baseline"][
        "heldout_score"
    ] == pytest.approx(0.125)
    paired = result["variants"]["no_sequence_distillation"][
        "paired_delta_statistics"
    ]["heldout_score"]
    assert paired["n"] == 2
    assert paired["ci95"] == pytest.approx([0.1, 0.15])
    assert (
        result["variants"]["no_sequence_distillation"]["heldout_score_conclusion"]
        == "improved"
    )
    assert set(result["matched_evaluation_artifacts_by_replicate"]) == {
        "seed_0",
        "seed_1",
    }
    assert result["replicate_count"] == 2
    candidate = result["variants"]["no_sequence_distillation"]
    assert candidate["gate_status"] == "insufficient_evidence"
    assert candidate["gate_statuses"]["parameter_budget"] == "pass"
    assert candidate["gate_statuses"]["generalization"] == "pass"
    assert (
        result["variants"]["baseline"]["gate_statuses"]["generalization"]
        == "insufficient_evidence"
    )
    robustness = candidate["heldout_robustness_delta_statistics"][
        "document_family"
    ]["chart"]
    assert robustness["n"] == 2
    assert robustness["mean"] == pytest.approx(0.125)
    assert robustness["ci95"] == pytest.approx([0.1, 0.15])
    assert Path(candidate["gate_report"]).is_file()
    assert all(Path(record["gate_report"]).is_file() for record in result["runs"].values())
    assert (Path(plan.root) / "comparison.json").is_file()
    markdown = (Path(plan.root) / "comparison.md").read_text(encoding="utf-8")
    assert "Paired delta [95% CI]" in markdown
    assert "Gates" in markdown
    assert "`no_sequence_distillation`" in markdown


def test_sweep_rejects_mismatched_evaluation_artifacts(tmp_path):
    plan = _compile_tiny(tmp_path)
    for index, variant in enumerate(plan.variants):
        samples = Path(variant.plan.root) / "artifacts" / "samples"
        evaluation = Path(variant.plan.root) / "artifacts" / "evaluation"
        samples.mkdir(parents=True, exist_ok=True)
        evaluation.mkdir(parents=True, exist_ok=True)
        image = samples / "document.png"
        image.write_bytes(f"image-{index}".encode())
        row = {
            "sample_id": "sample-1",
            "image_path": str(image),
            "question": "Read.",
            "answers": ["42"],
            "answer_type": "chart",
            "metric": "relaxed_acc",
            "meta": {},
        }
        for split in ("train", "heldout"):
            (samples / f"{split}.jsonl").write_text(
                json.dumps(row) + "\n",
                encoding="utf-8",
            )
        (evaluation / "comparison.json").write_text(
            json.dumps(_comparison(0.1, 1.0)),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="mismatched evaluation artifacts"):
        aggregate_sweep_results(plan)


def test_sweep_runner_dry_run_does_not_create_the_output_root(tmp_path):
    plan = _compile_tiny(tmp_path)
    output = Path(plan.root)
    assert not output.exists()

    result = SweepRunner(plan, repo_root=ROOT).run(dry_run=True)

    assert result["dry_run"] is True
    assert len(result["variants"]) == 4
    assert not output.exists()


def test_sweep_runner_records_each_variant_once_and_updates_suite_state(
    tmp_path,
    monkeypatch,
):
    import docvlm_eval.student.sweep as sweep_module

    plan = _compile_tiny(tmp_path)
    calls = []

    class FakeExperimentRunner:
        def __init__(self, experiment_plan, *, repo_root):
            del repo_root
            self.plan = experiment_plan

        def run(self, **kwargs):
            calls.append((self.plan.name, kwargs))
            return {"dry_run": False, "outcomes": []}

    monkeypatch.setattr(sweep_module, "ExperimentRunner", FakeExperimentRunner)
    result = SweepRunner(plan, repo_root=ROOT).run(stop="pretrain")

    assert len(calls) == 4
    assert result["status"] == "completed"
    assert [item["status"] for item in result["variants"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]
    persisted = json.loads(
        (Path(plan.root) / "sweep_run_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"


def test_sweep_runner_filters_by_arm_and_replicate(tmp_path, monkeypatch):
    import docvlm_eval.student.sweep as sweep_module

    plan = _compile_tiny(tmp_path)
    calls = []

    class FakeExperimentRunner:
        def __init__(self, experiment_plan, *, repo_root):
            del repo_root
            self.plan = experiment_plan

        def run(self, **kwargs):
            calls.append((self.plan.name, kwargs))
            return {"dry_run": True, "outcomes": []}

    monkeypatch.setattr(sweep_module, "ExperimentRunner", FakeExperimentRunner)
    result = SweepRunner(plan, repo_root=ROOT).run(
        dry_run=True,
        variant_ids={"baseline"},
        replicate_ids={"seed_1"},
    )

    assert len(calls) == 1
    assert result["variants"][0]["run"] == "baseline--seed_1"

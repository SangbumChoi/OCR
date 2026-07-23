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
        path.write_text(
            json.dumps(_comparison(*scores[(variant.arm_id, variant.replicate_id)])),
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
    assert (Path(plan.root) / "comparison.json").is_file()
    markdown = (Path(plan.root) / "comparison.md").read_text(encoding="utf-8")
    assert "Paired delta [95% CI]" in markdown
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

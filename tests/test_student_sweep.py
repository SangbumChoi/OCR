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
    assert [variant.id for variant in plan.variants] == [
        "baseline",
        "no_sequence_distillation",
    ]
    baseline, ablation = plan.variants
    assert baseline.parameters["total"] == 587_017
    assert ablation.parameters["total"] == 587_017
    assert "generate_teacher_predictions" in baseline.plan.stage_names
    assert "generate_teacher_predictions" not in ablation.plan.stage_names
    assert Path(baseline.plan.root) != Path(ablation.plan.root)
    evaluation = baseline.plan.raw_spec["evaluation"]
    assert evaluation["wandb_group"] == "docvlm-tiny-sweep"
    assert evaluation["wandb_run"] == "docvlm-tiny-sweep--baseline"
    assert "variant:baseline" in evaluation["wandb_tags"]
    assert plan.control_values["experiment:/pretraining/max_steps"] == 1

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


def test_full_sweep_compiles_loss_sft_and_reward_ablation_contracts(tmp_path):
    plan = compile_sweep_plan(
        ROOT / "configs" / "sub1b_sweep.yaml",
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )
    variants = {variant.id: variant for variant in plan.variants}

    assert set(variants) == {
        "baseline",
        "no_sequence_distillation",
        "no_spatial_auxiliary",
        "sft_answer_only",
        "correctness_only_rlvr",
    }
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
    scores = {"baseline": (0.2, 20.0), "no_sequence_distillation": (0.35, 24.0)}
    for variant in plan.variants:
        samples = Path(variant.plan.root) / "artifacts" / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        image = samples / "document.png"
        image.write_bytes(b"identical-image")
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
            json.dumps(_comparison(*scores[variant.id])),
            encoding="utf-8",
        )

    result = aggregate_sweep_results(plan)

    assert result["ranking"] == ["no_sequence_distillation", "baseline"]
    assert result["variants"]["no_sequence_distillation"]["delta_vs_baseline"][
        "heldout_score"
    ] == pytest.approx(0.15)
    assert set(result["matched_evaluation_artifacts"]) == {"train", "heldout"}
    assert (Path(plan.root) / "comparison.json").is_file()
    markdown = (Path(plan.root) / "comparison.md").read_text(encoding="utf-8")
    assert "Train-heldout gap" in markdown
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
    assert len(result["variants"]) == 2
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

    assert len(calls) == 2
    assert result["status"] == "completed"
    assert [item["status"] for item in result["variants"]] == [
        "completed",
        "completed",
    ]
    persisted = json.loads(
        (Path(plan.root) / "sweep_run_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "completed"

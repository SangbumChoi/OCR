import json
import sys
from pathlib import Path

import pytest
import yaml

from docvlm_eval.student.factorial import (
    FactorialRunner,
    aggregate_factorial_results,
    compile_factorial_plan,
)
from docvlm_eval.student.sweep import aggregate_sweep_results


ROOT = Path(__file__).resolve().parents[1]
REVISION = "f5eb52104627d20ddd1eab2130ad78f87cb0d7c9"


def _factorial_config(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    experiment = yaml.safe_load(
        (ROOT / "configs" / "sub1b_experiment_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    experiment["blueprint"] = str(
        ROOT / "configs" / "sub1b_architecture.yaml"
    )
    experiment["synthetic"]["config"] = str(
        ROOT / "configs" / "synth_data.yaml"
    )
    experiment["data"]["components"][0]["weight"] = 0.5
    experiment["data"]["components"].append(
        {
            "name": "public_udd",
            "weight": 0.5,
            "hub": {
                "repo_id": "danelcsb/UDD",
                "revision": REVISION,
                "split": "train",
                "fold": "train",
                "sources": [],
                "tasks": [],
                "languages": [],
                "max_rows": None,
                "seed": 7,
                "decode_checks": 1,
            },
        }
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(experiment, sort_keys=False),
        encoding="utf-8",
    )

    sweep = yaml.safe_load(
        (ROOT / "configs" / "sub1b_sweep_tiny.yaml").read_text(
            encoding="utf-8"
        )
    )
    sweep["base_experiment"] = str(experiment_path)
    sweep["output_root"] = str(tmp_path / "unused-sweep")
    sweep_path = tmp_path / "base-sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(sweep, sort_keys=False),
        encoding="utf-8",
    )

    factorial = {
        "schema_version": 1,
        "name": "tiny-initialization-data-scale",
        "base_sweep": str(sweep_path),
        "output_root": str(tmp_path / "factorial-output"),
        "reference_scale": "full",
        "heldout_count": 3,
        "public_component_index": 1,
        "scales": [
            {
                "id": "low",
                "synthetic_train_count": 1,
                "public_max_rows": 2,
            },
            {
                "id": "full",
                "synthetic_train_count": 4,
                "public_max_rows": None,
            },
        ],
    }
    path = tmp_path / "factorial.yaml"
    path.write_text(
        yaml.safe_dump(factorial, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _compile(tmp_path: Path):
    return compile_factorial_plan(
        _factorial_config(tmp_path),
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def test_factorial_compiles_scale_specific_training_and_fixed_heldout(
    tmp_path,
):
    plan = _compile(tmp_path)

    assert plan.scale_ids == ("low", "full")
    assert plan.reference_scale == "full"
    assert sum(len(scale.sweep.variants) for scale in plan.scales) == 8
    low = plan.scales[0]
    full = plan.scales[1]
    low_variant = low.sweep.variants[0]
    full_variant = full.sweep.variants[0]
    low_train = next(
        stage
        for stage in low_variant.plan.stages
        if stage.name == "synthetic_train"
    )
    low_heldout = next(
        stage
        for stage in low_variant.plan.stages
        if stage.name == "synthetic_heldout"
    )
    full_train = next(
        stage
        for stage in full_variant.plan.stages
        if stage.name == "synthetic_train"
    )
    low_public = next(
        stage
        for stage in low_variant.plan.stages
        if stage.name == "acquire_component_public_udd"
    )
    full_public = next(
        stage
        for stage in full_variant.plan.stages
        if stage.name == "acquire_component_public_udd"
    )

    assert low_train.command[low_train.command.index("--count") + 1] == "1"
    assert full_train.command[full_train.command.index("--count") + 1] == "4"
    assert low_heldout.command[low_heldout.command.index("--count") + 1] == "3"
    assert low_public.command[low_public.command.index("--max-rows") + 1] == "2"
    assert "--max-rows" not in full_public.command
    assert "data-scale:low" in low_variant.plan.raw_spec["evaluation"][
        "wandb_tags"
    ]
    assert (
        low_variant.plan.resolved_blueprint["training"]["pretraining"][
            "optimizer"
        ]["total_tokens"]
        == full_variant.plan.resolved_blueprint["training"]["pretraining"][
            "optimizer"
        ]["total_tokens"]
    )


def test_factorial_rejects_non_monotonic_data_scales(tmp_path):
    config = _factorial_config(tmp_path)
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["scales"][1]["public_max_rows"] = 1
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-decreasing"):
        compile_factorial_plan(
            config,
            repo_root=ROOT,
            python=sys.executable,
            compile_root=tmp_path / "compiled",
        )


def test_factorial_fingerprint_ignores_temporary_compile_location(tmp_path):
    config = _factorial_config(tmp_path)
    first = compile_factorial_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-a",
    )
    second = compile_factorial_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled-b",
    )

    assert first.fingerprint == second.fingerprint
    assert [
        scale.sweep.fingerprint for scale in first.scales
    ] == [
        scale.sweep.fingerprint for scale in second.scales
    ]


def _comparison(score: float) -> dict:
    def split(name: str, value: float) -> dict:
        return {
            "split": name,
            "dataset_size": 1,
            "n_samples": 1,
            "score": value,
            "reward": value / 2,
            "valid_structure_fraction": 1.0,
            "answer_rate": 1.0,
            "elapsed_seconds": 0.01,
            "milliseconds_per_sample": 10.0,
            "by_answer_type": {
                "chart": {
                    "n": 1,
                    "score": value,
                    "reward": value / 2,
                    "valid_structure_fraction": 1.0,
                    "answer_rate": 1.0,
                }
            },
            "by_source": {},
            "by_language": {},
            "reward_components": {},
        }

    return {
        "splits": {
            "train": split("train", score + 0.1),
            "heldout": split("heldout", score),
        },
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


def _write_fake_run(
    variant,
    *,
    scale_id: str,
    score: float,
    mixture_rows: int,
) -> None:
    root = Path(variant.plan.root)
    samples = root / "artifacts" / "samples"
    evaluation = root / "artifacts" / "evaluation"
    mixture = root / "artifacts" / "data" / "mixture"
    samples.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    mixture.mkdir(parents=True)

    for split in ("train", "heldout"):
        image = samples / f"{split}.png"
        image.write_bytes(
            (
                f"{split}-{variant.replicate_id}"
                if split == "heldout"
                else f"{split}-{scale_id}-{variant.replicate_id}"
            ).encode()
        )
        row = {
            "sample_id": "sample-1",
            "image_path": str(image),
            "question": (
                "Read the heldout chart."
                if split == "heldout"
                else f"Read the {scale_id} training chart."
            ),
            "answers": ["42"],
            "answer_type": "chart",
            "metric": "relaxed_acc",
            "meta": {"language": "en"},
        }
        (samples / f"{split}.jsonl").write_text(
            json.dumps(row) + "\n",
            encoding="utf-8",
        )
        split_root = evaluation / split
        split_root.mkdir()
        eval_row = {
            "sample_id": "sample-1",
            "score": score + (0.1 if split == "train" else 0.0),
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
    (evaluation / "comparison.json").write_text(
        json.dumps(_comparison(score)),
        encoding="utf-8",
    )
    (mixture / "mixture_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": mixture_rows,
                "weights": {
                    "synthetic_documents": 0.5,
                    "public_udd": 0.5,
                },
                "dataset_fingerprint": (
                    f"{scale_id}-{variant.replicate_id}"
                ),
                "components": [
                    {
                        "name": "synthetic_documents",
                        "path": str(root / "synthetic"),
                        "weight": 0.5,
                        "rows": mixture_rows // 2,
                    },
                    {
                        "name": "public_udd",
                        "path": str(root / "public"),
                        "weight": 0.5,
                        "rows": mixture_rows - mixture_rows // 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _materialize_factorial(plan) -> None:
    scores = {
        ("low", "baseline", "seed_0"): 0.20,
        ("low", "baseline", "seed_1"): 0.30,
        ("low", "no_sequence_distillation", "seed_0"): 0.50,
        ("low", "no_sequence_distillation", "seed_1"): 0.55,
        ("full", "baseline", "seed_0"): 0.40,
        ("full", "baseline", "seed_1"): 0.45,
        ("full", "no_sequence_distillation", "seed_0"): 0.50,
        ("full", "no_sequence_distillation", "seed_1"): 0.55,
    }
    for scale in plan.scales:
        for variant in scale.sweep.variants:
            _write_fake_run(
                variant,
                scale_id=scale.id,
                score=scores[
                    (scale.id, variant.arm_id, variant.replicate_id)
                ],
                mixture_rows=4 if scale.id == "low" else 12,
            )
        aggregate_sweep_results(scale.sweep)


def test_factorial_aggregates_paired_interactions_and_actual_data(tmp_path):
    plan = _compile(tmp_path)
    _materialize_factorial(plan)

    result = aggregate_factorial_results(plan)

    interaction = result["interactions"]["no_sequence_distillation"][
        "low"
    ]["metric_statistics"]["heldout_score"]
    assert interaction["mean"] == pytest.approx(0.175)
    assert interaction["ci95"] == pytest.approx([0.15, 0.2])
    assert (
        result["interactions"]["no_sequence_distillation"]["low"][
            "heldout_score_conclusion"
        ]
        == "improved"
    )
    assert (
        result["interactions"]["no_sequence_distillation"]["full"][
            "heldout_score_conclusion"
        ]
        == "reference"
    )
    assert result["scales"]["low"]["configured"][
        "synthetic_heldout_count_per_case"
    ] == 3
    assert {
        value["rows"]
        for value in result["scales"]["low"][
            "actual_training_data_by_replicate"
        ].values()
    } == {4}
    assert (
        Path(plan.root) / "factorial_comparison.json"
    ).is_file()
    markdown = (
        Path(plan.root) / "factorial_comparison.md"
    ).read_text(encoding="utf-8")
    assert "Interaction vs reference" in markdown


def test_factorial_rejects_changed_heldout_artifacts(tmp_path):
    plan = _compile(tmp_path)
    _materialize_factorial(plan)
    comparison_path = (
        Path(plan.scales[0].sweep.root) / "comparison.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    comparison["matched_evaluation_artifacts_by_replicate"]["seed_0"][
        "heldout"
    ] = "sha256:changed"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    with pytest.raises(ValueError, match="changed heldout"):
        aggregate_factorial_results(plan)


def test_factorial_runner_filters_scale_variant_and_replicate(
    tmp_path,
    monkeypatch,
):
    import docvlm_eval.student.factorial as factorial_module

    plan = _compile(tmp_path)
    calls = []

    class FakeSweepRunner:
        def __init__(self, sweep_plan, *, repo_root):
            del repo_root
            self.plan = sweep_plan

        def run(self, **kwargs):
            calls.append((self.plan.name, kwargs))
            return {"dry_run": True}

    monkeypatch.setattr(factorial_module, "SweepRunner", FakeSweepRunner)
    result = FactorialRunner(plan, repo_root=ROOT).run(
        dry_run=True,
        scale_ids={"low"},
        variant_ids={"baseline"},
        replicate_ids={"seed_1"},
    )

    assert len(calls) == 1
    assert calls[0][0].endswith("-low")
    assert calls[0][1]["variant_ids"] == {"baseline"}
    assert result["scales"][0]["scale"] == "low"

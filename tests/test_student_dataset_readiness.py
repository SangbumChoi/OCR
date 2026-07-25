from __future__ import annotations

from copy import deepcopy

from docvlm_eval.student.dataset_readiness import (
    REQUIRED_UDD_COLUMNS,
    build_public_dataset_readiness,
    validate_public_dataset_readiness,
)


REVISION = "a" * 40
TASK_COUNTS = {
    "vqa": 15,
    "recognition": 9,
    "reasoning": 8,
    "localization": 3,
    "kie": 2,
    "table": 2,
    "classification": 1,
}
SOURCES = [
    "chartqa",
    "cord",
    "docmatix",
    "docvqa",
    "dvqa",
    "funsd",
    "infovqa",
    "mtvqa",
    "plotqa",
    "pubtabnet",
    "sroie",
    "synthdog_ko",
    "tatqa",
    "visualmrc",
]


def _card() -> str:
    features = "\n".join(
        f"  - name: {column}\n    dtype: string"
        for column in sorted(REQUIRED_UDD_COLUMNS)
    )
    task_text = ", ".join(
        f"{task} {count}" for task, count in TASK_COUNTS.items()
    )
    source_text = ", ".join(f"`{source}`" for source in SOURCES)
    return f"""---
dataset_info:
  features:
{features}
  splits:
  - name: train
    num_bytes: 5000
    num_examples: 40
---
> **Current release:** **40 image-rows / 77 QAs** from **14 source datasets** /
> **7 tasks** - {task_text} image-rows;

### Derived columns
current distribution (image-rows): en 30, ko 4, zh 3, ja 2, und 1

### Sources (14)
{source_text}.
The task label `reasoning` is not a source.

### Load
"""


def _experiment() -> dict:
    return {
        "data": {
            "components": [
                {"name": "synthetic_documents", "path": "@synthetic", "weight": 0.45},
                {
                    "name": "public_udd",
                    "weight": 0.55,
                    "hub": {
                        "repo_id": "owner/UDD",
                        "revision": REVISION,
                        "split": "train",
                        "fold": "train",
                        "sources": [],
                        "tasks": [],
                        "languages": [],
                        "max_rows": None,
                        "decode_checks": 32,
                    },
                },
            ]
        }
    }


def _viewer() -> dict:
    return {
        "dataset_info": {
            "default": {
                "features": {
                    column: {"dtype": "string"}
                    for column in REQUIRED_UDD_COLUMNS
                },
                "splits": {"train": {"num_examples": 40}},
                "download_size": 500,
            }
        }
    }


def _files() -> list[dict]:
    return [
        {
            "path": f"data/train-{index:05d}-of-00005.parquet",
            "size": 100,
            "sha256": f"{index + 1:064x}",
        }
        for index in range(5)
    ]


def _build() -> dict:
    return build_public_dataset_readiness(
        _experiment(),
        repo_id="owner/UDD",
        requested_revision=REVISION,
        resolved_revision=REVISION,
        main_revision=REVISION,
        card=_card(),
        files=_files(),
        viewer_info=_viewer(),
    )


def test_public_dataset_readiness_binds_multitask_hub_snapshot():
    result = _build()

    assert result["overall_status"] == "pass"
    assert result["training_component_authorized"] is True
    assert result["quality_claim_authorized"] is False
    assert result["dataset"]["task_counts"] == TASK_COUNTS
    assert result["dataset"]["capability_sources"]["chart_understanding"] == [
        "chartqa",
        "dvqa",
        "plotqa",
        "tatqa",
    ]
    assert validate_public_dataset_readiness(
        result,
        repo_id="owner/UDD",
        revision=REVISION,
    ) == []


def test_public_dataset_readiness_rejects_mutation_and_moving_revision():
    result = _build()
    tampered = deepcopy(result)
    tampered["dataset"]["task_counts"].pop("kie")
    errors = validate_public_dataset_readiness(
        tampered,
        repo_id="owner/UDD",
        revision=REVISION,
    )

    assert "fingerprint mismatch" in errors
    assert "required task coverage is incomplete" in errors

    moving = build_public_dataset_readiness(
        _experiment(),
        repo_id="owner/UDD",
        requested_revision=REVISION,
        resolved_revision=REVISION,
        main_revision="b" * 40,
        card=_card(),
        files=_files(),
        viewer_info=_viewer(),
    )
    assert moving["overall_status"] == "fail"

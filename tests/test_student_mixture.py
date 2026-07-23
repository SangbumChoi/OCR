import json
from pathlib import Path

import pytest

from docvlm_eval.student.mixture import (
    MixtureComponent,
    build_weighted_mixture,
    validate_components,
)


def _write_dataset(path: Path, image_path: Path, *, source: str, count: int, fold: str = ""):
    from datasets import Dataset

    rows = [
        {
            "image": str(image_path),
            "sample_id": f"{source}-{index}",
            "source": source,
            "task": "vqa",
            "instructions": [f"Question {index}?"],
            "answers": [[f"answer-{index}"]],
            "language": "en",
            "fold": fold,
        }
        for index in range(count)
    ]
    Dataset.from_list(rows).save_to_disk(str(path))


def test_component_validation_normalizes_weights():
    components = validate_components(
        [
            MixtureComponent("synthetic", "/tmp/a", 3.0),
            MixtureComponent("public", "/tmp/b", 1.0),
        ]
    )
    assert components[0].weight == pytest.approx(0.75)
    assert components[1].weight == pytest.approx(0.25)
    with pytest.raises(ValueError, match="unique"):
        validate_components(
            [
                MixtureComponent("same", "/tmp/a", 1.0),
                MixtureComponent("same", "/tmp/b", 1.0),
            ]
        )
    with pytest.raises(ValueError, match="must match"):
        MixtureComponent("../escape", "/tmp/a", 1.0)


def test_build_mixture_preserves_rows_and_runtime_weights(tmp_path):
    from datasets import load_from_disk
    from PIL import Image

    from docvlm_eval.student.data import BalancedGroupBatchSampler, UDDStudentDataset

    image_path = tmp_path / "page.png"
    Image.new("RGB", (16, 16), "white").save(image_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, image_path, source="source-a", count=2)
    _write_dataset(second, image_path, source="source-b", count=3)
    (first / "component_manifest.json").write_text(
        json.dumps({"revision": "a" * 40}),
        encoding="utf-8",
    )

    output = tmp_path / "mixture"
    manifest = build_weighted_mixture(
        [
            MixtureComponent("synthetic_documents", str(first), 0.7),
            MixtureComponent("public_document_data", str(second), 0.3),
        ],
        output,
    )
    assert manifest["rows"] == 5
    assert manifest["weights"] == {
        "synthetic_documents": pytest.approx(0.7),
        "public_document_data": pytest.approx(0.3),
    }
    assert manifest["components"][0]["upstream_manifest_fingerprint"].startswith(
        "sha256:"
    )
    assert manifest["components"][1]["upstream_manifest_fingerprint"] is None

    mixed = load_from_disk(str(output))
    assert mixed["fold"] == ["train"] * 5
    assert mixed["mixture_component"] == [
        "synthetic_documents",
        "synthetic_documents",
        "public_document_data",
        "public_document_data",
        "public_document_data",
    ]
    expanded = UDDStudentDataset(mixed, include_grounding=False)
    assert expanded.groups("component") == mixed["mixture_component"]
    sampler = BalancedGroupBatchSampler(
        expanded.groups("component"),
        batch_size=8,
        group_weights={
            "synthetic_documents": 1.0,
            "public_document_data": 0.0,
        },
        num_batches=1,
        seed=3,
    )
    assert all(index < 2 for index in next(iter(sampler)))


def test_mixture_fold_filter_fails_closed(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    source = tmp_path / "source"
    _write_dataset(source, image_path, source="a", count=1, fold="train")
    with pytest.raises(ValueError, match="no selected rows"):
        build_weighted_mixture(
            [MixtureComponent("a", str(source), 1.0, fold="heldout")],
            tmp_path / "out",
        )

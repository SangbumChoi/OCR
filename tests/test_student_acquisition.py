from pathlib import Path

import pytest

from docvlm_eval.student.acquisition import (
    HubComponentSpec,
    acquire_hub_component,
    materialize_component,
)
from docvlm_eval.student.mixture import MixtureComponent, build_weighted_mixture


REVISION = "a" * 40


def _udd_dataset(tmp_path: Path, *, duplicate_phash: bool = False):
    from datasets import Dataset, Image as HFImage
    from PIL import Image

    rows = []
    for index in range(5):
        width = 20 if duplicate_phash and index < 2 else 20 + index
        image_path = tmp_path / f"page-{index}.png"
        Image.new("RGB", (width, 12), (index * 20, 0, 0)).save(image_path)
        phash = "same" if duplicate_phash and index < 2 else f"hash-{index}"
        rows.append(
            {
                "image": str(image_path),
                "sample_id": f"sample-{index}",
                "source": "cord" if index % 2 == 0 else "chartqa",
                "task": "kie" if index % 2 == 0 else "reasoning",
                "instructions": [f"Question {index}?"],
                "answers": [[f"answer-{index}"]],
                "elements_json": "[]",
                "full_text": "",
                "table_html": "",
                "language": "en",
                "metric": "anls",
                "fold": "heldout" if index == 4 else "train",
                "phash": phash,
                "image_width": width,
                "image_height": 12,
                "license": "apache-2.0",
            }
        )
    return Dataset.from_list(rows).cast_column("image", HFImage())


def test_hub_component_requires_immutable_revision():
    with pytest.raises(ValueError, match="immutable"):
        HubComponentSpec(repo_id="owner/dataset", revision="main")


def test_materialize_component_filters_and_hash_samples_deterministically(tmp_path):
    from datasets import load_from_disk

    dataset = _udd_dataset(tmp_path)
    spec = HubComponentSpec(
        repo_id="owner/dataset",
        revision=REVISION,
        fold="train",
        max_rows=2,
        seed=19,
        decode_checks=2,
    )
    first_output = tmp_path / "first"
    manifest = materialize_component(
        dataset,
        first_output,
        spec,
        resolved_revision=REVISION,
    )
    assert manifest["source_rows"] == 5
    assert manifest["validation"]["rows"] == 2
    assert manifest["validation"]["decoded_images_checked"] == 2
    selected = load_from_disk(str(first_output))
    assert selected["fold"] == ["train", "train"]
    mixture = build_weighted_mixture(
        [MixtureComponent("public_udd", str(first_output), 1.0)],
        tmp_path / "mixture",
    )
    assert mixture["rows"] == 2
    assert mixture["components"][0]["upstream_manifest_fingerprint"].startswith(
        "sha256:"
    )

    second_output = tmp_path / "second"
    second = materialize_component(
        dataset,
        second_output,
        spec,
        resolved_revision=REVISION,
    )
    assert second["selected_indices_fingerprint"] == manifest["selected_indices_fingerprint"]
    assert load_from_disk(str(second_output))["sample_id"] == selected["sample_id"]


def test_task_stratified_sampling_guarantees_floor_and_is_deterministic(tmp_path):
    from datasets import load_from_disk

    dataset = _udd_dataset(tmp_path)
    spec = HubComponentSpec(
        repo_id="owner/dataset",
        revision=REVISION,
        fold="train",
        max_rows=2,
        seed=19,
        decode_checks=2,
        sampling_strategy="task_stratified",
        min_rows_per_task=1,
    )
    first = materialize_component(
        dataset,
        tmp_path / "stratified-first",
        spec,
        resolved_revision=REVISION,
    )
    second = materialize_component(
        dataset,
        tmp_path / "stratified-second",
        spec,
        resolved_revision=REVISION,
    )

    assert first["selection"] == {
        "applied_strategy": "task_stratified",
        "eligible_rows": 4,
        "eligible_task_counts": {"kie": 2, "reasoning": 2},
        "task_quotas": {"kie": 1, "reasoning": 1},
        "selected_rows": 2,
        "selected_task_counts": {"kie": 1, "reasoning": 1},
        "task_floor_satisfied": True,
    }
    assert (
        second["selected_indices_fingerprint"]
        == first["selected_indices_fingerprint"]
    )
    selected = load_from_disk(str(tmp_path / "stratified-first"))
    assert sorted(selected["task"]) == ["kie", "reasoning"]


def test_task_stratified_sampling_rejects_impossible_floor(tmp_path):
    dataset = _udd_dataset(tmp_path)
    spec = HubComponentSpec(
        repo_id="owner/dataset",
        revision=REVISION,
        fold="train",
        max_rows=3,
        decode_checks=1,
        sampling_strategy="task_stratified",
        min_rows_per_task=2,
    )

    with pytest.raises(ValueError, match="cannot satisfy"):
        materialize_component(
            dataset,
            tmp_path / "impossible",
            spec,
            resolved_revision=REVISION,
        )


def test_sampling_contract_rejects_inconsistent_controls():
    with pytest.raises(ValueError, match="min_rows_per_task requires"):
        HubComponentSpec(
            repo_id="owner/dataset",
            revision=REVISION,
            min_rows_per_task=1,
        )
    with pytest.raises(ValueError, match="requires a positive"):
        HubComponentSpec(
            repo_id="owner/dataset",
            revision=REVISION,
            sampling_strategy="task_stratified",
        )


def test_materialize_component_rejects_duplicate_image_identity(tmp_path):
    dataset = _udd_dataset(tmp_path, duplicate_phash=True)
    spec = HubComponentSpec(
        repo_id="owner/dataset",
        revision=REVISION,
        fold="train",
        decode_checks=1,
    )
    with pytest.raises(ValueError, match="duplicate phash"):
        materialize_component(
            dataset,
            tmp_path / "output",
            spec,
            resolved_revision=REVISION,
        )


def test_hub_acquisition_verifies_resolved_revision_and_loader_contract(tmp_path):
    dataset = _udd_dataset(tmp_path)
    calls = {}

    def loader(**kwargs):
        calls.update(kwargs)
        return dataset

    class Api:
        def dataset_info(self, repo_id, revision):
            assert repo_id == "owner/dataset"
            assert revision == REVISION
            return type("Info", (), {"sha": REVISION})()

    spec = HubComponentSpec(
        repo_id="owner/dataset",
        revision=REVISION,
        fold="train",
        sources=("cord",),
        max_rows=1,
        decode_checks=1,
    )
    manifest = acquire_hub_component(
        spec,
        tmp_path / "output",
        token="secret",
        dataset_loader=loader,
        hub_api=Api(),
    )
    assert calls == {
        "path": "owner/dataset",
        "split": "train",
        "revision": REVISION,
        "token": "secret",
    }
    assert manifest["validation"]["sources"] == {"cord": 1}

    class WrongApi:
        def dataset_info(self, repo_id, revision):
            return type("Info", (), {"sha": "b" * 40})()

    with pytest.raises(ValueError, match="does not match pinned"):
        acquire_hub_component(
            spec,
            tmp_path / "wrong",
            dataset_loader=loader,
            hub_api=WrongApi(),
        )

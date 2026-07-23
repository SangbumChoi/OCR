import json
from pathlib import Path

import pytest

from docvlm_eval.student.checkpoint_acquisition import (
    HubCheckpointSpec,
    acquire_hub_checkpoint,
    checkpoint_manifest_valid,
    checkpoint_path_from_manifest,
    validate_checkpoint_snapshot,
)


REVISION = "a" * 40


def _snapshot(tmp_path: Path, *, model_type: str = "siglip") -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": model_type}),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"safe checkpoint")
    return snapshot


def test_hub_checkpoint_requires_immutable_revision():
    with pytest.raises(ValueError, match="immutable"):
        HubCheckpointSpec(
            repo_id="owner/model",
            revision="main",
            family="siglip",
        )


def test_checkpoint_snapshot_validates_family_and_content(tmp_path):
    snapshot = _snapshot(tmp_path)
    spec = HubCheckpointSpec(
        repo_id="owner/model",
        revision=REVISION,
        family="siglip",
    )

    manifest = validate_checkpoint_snapshot(
        snapshot,
        spec,
        resolved_revision=REVISION,
    )

    assert manifest["model_type"] == "siglip"
    assert manifest["total_bytes"] > 0
    assert manifest["content_fingerprint"].startswith("sha256:")
    with pytest.raises(ValueError, match="incompatible"):
        validate_checkpoint_snapshot(
            snapshot,
            HubCheckpointSpec(
                repo_id="owner/model",
                revision=REVISION,
                family="llama",
            ),
            resolved_revision=REVISION,
        )


def test_hub_checkpoint_acquisition_writes_a_live_cache_manifest(tmp_path):
    snapshot = _snapshot(tmp_path)
    calls = {}

    def loader(**kwargs):
        calls.update(kwargs)
        return str(snapshot)

    class Api:
        def model_info(self, repo_id, revision):
            assert repo_id == "owner/model"
            assert revision == REVISION
            return type("Info", (), {"sha": REVISION})()

    manifest_path = tmp_path / "checkpoint.json"
    spec = HubCheckpointSpec(
        repo_id="owner/model",
        revision=REVISION,
        family="siglip",
    )
    acquire_hub_checkpoint(
        spec,
        manifest_path,
        token="secret",
        snapshot_loader=loader,
        hub_api=Api(),
    )

    assert calls["repo_id"] == "owner/model"
    assert calls["revision"] == REVISION
    assert calls["token"] == "secret"
    assert checkpoint_manifest_valid(manifest_path)
    assert checkpoint_path_from_manifest(manifest_path) == snapshot.resolve()

    (snapshot / "model.safetensors").unlink()
    assert not checkpoint_manifest_valid(manifest_path)


def test_checkpoint_snapshot_rejects_wrong_revision_and_missing_shard(tmp_path):
    snapshot = _snapshot(tmp_path)
    spec = HubCheckpointSpec(
        repo_id="owner/model",
        revision=REVISION,
        family="siglip",
    )
    with pytest.raises(ValueError, match="does not match pinned"):
        validate_checkpoint_snapshot(
            snapshot,
            spec,
            resolved_revision="b" * 40,
        )

    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing shards"):
        validate_checkpoint_snapshot(
            snapshot,
            spec,
            resolved_revision=REVISION,
        )

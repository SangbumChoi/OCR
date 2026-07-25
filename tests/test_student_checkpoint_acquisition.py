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
    assert manifest["schema_version"] == 2
    assert manifest["weight_format"] == "safetensors"
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


@pytest.mark.parametrize("model_type", ["lfm2", "lfm2_vl"])
def test_checkpoint_snapshot_accepts_lfm2_family(tmp_path, model_type):
    snapshot = _snapshot(tmp_path, model_type=model_type)
    manifest = validate_checkpoint_snapshot(
        snapshot,
        HubCheckpointSpec(
            repo_id="LiquidAI/LFM2-compatible",
            revision=REVISION,
            family="lfm2",
        ),
        resolved_revision=REVISION,
    )

    assert manifest["model_type"] == model_type
    assert manifest["spec"]["family"] == "lfm2"


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
            siblings = [
                type("Sibling", (), {"rfilename": "config.json"})(),
                type(
                    "Sibling",
                    (),
                    {"rfilename": "model.safetensors"},
                )(),
                type(
                    "Sibling",
                    (),
                    {"rfilename": "pytorch_model.bin"},
                )(),
            ]
            return type(
                "Info",
                (),
                {"sha": REVISION, "siblings": siblings},
            )()

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
    assert calls["allow_patterns"] == [
        "config.json",
        "model.safetensors",
    ]
    assert checkpoint_manifest_valid(manifest_path)
    assert checkpoint_path_from_manifest(manifest_path) == snapshot.resolve()

    original = (snapshot / "model.safetensors").read_bytes()
    (snapshot / "model.safetensors").write_bytes(b"x" * len(original))
    assert checkpoint_manifest_valid(manifest_path)
    assert not checkpoint_manifest_valid(
        manifest_path,
        verify_hashes=True,
    )
    with pytest.raises(ValueError, match="invalid or incomplete"):
        checkpoint_path_from_manifest(manifest_path)
    (snapshot / "model.safetensors").write_bytes(original)

    (snapshot / "model.safetensors").unlink()
    assert not checkpoint_manifest_valid(manifest_path)


def test_checkpoint_acquisition_prefers_one_safetensors_shard_set(
    tmp_path,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen2"}),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"b")
    (snapshot / "pytorch_model.bin").write_bytes(b"duplicate")
    calls = {}

    def loader(**kwargs):
        calls.update(kwargs)
        return str(snapshot)

    class Api:
        def model_info(self, repo_id, revision):
            del repo_id, revision
            names = [
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "pytorch_model.bin",
            ]
            siblings = [type("Sibling", (), {"rfilename": name})() for name in names]
            return type(
                "Info",
                (),
                {"sha": REVISION, "siblings": siblings},
            )()

    manifest = acquire_hub_checkpoint(
        HubCheckpointSpec(
            repo_id="owner/model",
            revision=REVISION,
            family="llama",
        ),
        tmp_path / "manifest.json",
        snapshot_loader=loader,
        hub_api=Api(),
    )

    assert calls["allow_patterns"] == [
        "config.json",
        "model.safetensors.index.json",
        "model-*.safetensors",
    ]
    assert manifest["weight_format"] == "safetensors"
    assert manifest["selected_allow_patterns"] == calls["allow_patterns"]
    assert {record["path"] for record in manifest["files"]} == {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


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

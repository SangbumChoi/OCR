from __future__ import annotations

import json
import struct
from types import SimpleNamespace

from safetensors import safe_open

from docvlm_eval.student.checkpoint_acquisition import (
    checkpoint_manifest_valid,
    checkpoint_path_from_manifest,
)
from docvlm_eval.student.selective_checkpoint import (
    materialize_contiguous_safetensors_subset,
)


def test_materializes_only_one_contiguous_tensor_prefix(tmp_path):
    tensors = {
        "model.text.weight": SimpleNamespace(
            dtype="F32",
            shape=[1],
            data_offsets=(0, 4),
            parameter_count=1,
        ),
        "model.vision.blocks.0.weight": SimpleNamespace(
            dtype="F32",
            shape=[2],
            data_offsets=(4, 12),
            parameter_count=2,
        ),
        "model.vision.blocks.1.weight": SimpleNamespace(
            dtype="F32",
            shape=[1],
            data_offsets=(12, 16),
            parameter_count=1,
        ),
    }
    metadata = SimpleNamespace(
        files_metadata={
            "model.safetensors": SimpleNamespace(
                tensors=tensors,
                metadata={"format": "pt"},
            )
        }
    )
    source_header_length = 16
    source_data = (
        struct.pack("<f", 9.0)
        + struct.pack("<ff", 1.5, 2.5)
        + struct.pack("<f", 3.5)
    )

    def range_get(_url, start, end):
        if (start, end) == (0, 7):
            return struct.pack("<Q", source_header_length)
        data_base = 8 + source_header_length
        return source_data[start - data_base : end - data_base + 1]

    output = tmp_path / "subset"
    manifest_path = tmp_path / "checkpoint.json"
    manifest = materialize_contiguous_safetensors_subset(
        repo_id="test/model",
        revision="a" * 40,
        tensor_prefixes=["model.vision.blocks."],
        output=output,
        manifest_path=manifest_path,
        metadata_loader=lambda *_args, **_kwargs: metadata,
        revision_resolver=lambda *_args: "a" * 40,
        file_url=lambda *_args: "https://example.test/model.safetensors",
        config_loader=lambda *_args: json.dumps(
            {"model_type": "test"}
        ).encode(),
        range_get=range_get,
    )

    assert manifest["selection"]["tensor_count"] == 2
    assert manifest["selection"]["parameter_count"] == 3
    assert manifest["source"]["payload_bytes"] == 12
    with safe_open(
        output / "model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == {
            "model.vision.blocks.0.weight",
            "model.vision.blocks.1.weight",
        }
        assert handle.get_tensor(
            "model.vision.blocks.0.weight"
        ).tolist() == [1.5, 2.5]
    assert checkpoint_manifest_valid(manifest_path, verify_hashes=True)
    assert checkpoint_path_from_manifest(manifest_path) == output.resolve()


def test_rejects_noncontiguous_selection(tmp_path):
    tensors = {
        "keep.0": SimpleNamespace(
            dtype="F32",
            shape=[1],
            data_offsets=(0, 4),
            parameter_count=1,
        ),
        "drop": SimpleNamespace(
            dtype="F32",
            shape=[1],
            data_offsets=(4, 8),
            parameter_count=1,
        ),
        "keep.1": SimpleNamespace(
            dtype="F32",
            shape=[1],
            data_offsets=(8, 12),
            parameter_count=1,
        ),
    }
    metadata = SimpleNamespace(
        files_metadata={
            "model.safetensors": SimpleNamespace(
                tensors=tensors,
                metadata={},
            )
        }
    )

    try:
        materialize_contiguous_safetensors_subset(
            repo_id="test/model",
            revision="b" * 40,
            tensor_prefixes=["keep."],
            output=tmp_path / "subset",
            metadata_loader=lambda *_args, **_kwargs: metadata,
            revision_resolver=lambda *_args: "b" * 40,
            file_url=lambda *_args: "unused",
            config_loader=lambda *_args: b"{}",
            range_get=lambda *_args: b"",
        )
    except ValueError as error:
        assert "not one contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous selection was accepted")

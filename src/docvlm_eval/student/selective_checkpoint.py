"""Materialize a contiguous tensor subset from a pinned Hub safetensors file."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence


_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def _default_range_get(url: str, start: int, end: int) -> bytes:
    import requests

    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=(30, 300),
    )
    response.raise_for_status()
    payload = response.content
    expected = end - start + 1
    if response.status_code != 206 or len(payload) != expected:
        raise RuntimeError(
            "Hub server did not honor the exact safetensors range request"
        )
    return payload


def _safetensors_bytes(
    tensors: Sequence[tuple[str, Any]],
    payload: bytes,
    *,
    metadata: dict[str, str] | None,
) -> bytes:
    cursor = 0
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    for name, info in tensors:
        length = int(info.data_offsets[1]) - int(info.data_offsets[0])
        header[name] = {
            "dtype": str(info.dtype),
            "shape": list(info.shape),
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length
    if cursor != len(payload):
        raise ValueError("selected tensor bytes do not match metadata offsets")
    encoded = json.dumps(
        header,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    encoded += b" " * padding
    return struct.pack("<Q", len(encoded)) + encoded + payload


def materialize_contiguous_safetensors_subset(
    *,
    repo_id: str,
    revision: str,
    tensor_prefixes: Sequence[str],
    output: str | Path,
    manifest_path: str | Path | None = None,
    metadata_loader: Callable[..., Any] | None = None,
    revision_resolver: Callable[[str, str], str] | None = None,
    file_url: Callable[[str, str, str], str] | None = None,
    config_loader: Callable[[str, str], bytes] | None = None,
    range_get: Callable[[str, int, int], bytes] | None = None,
) -> dict[str, Any]:
    """Write a valid checkpoint containing one contiguous selected tensor range."""

    if not repo_id or "/" not in repo_id:
        raise ValueError("repo_id must use namespace/name")
    if not _COMMIT.fullmatch(revision):
        raise ValueError("revision must be an immutable 40-character SHA")
    prefixes = tuple(sorted(set(str(item) for item in tensor_prefixes)))
    if not prefixes or any(not item for item in prefixes):
        raise ValueError("tensor_prefixes must be unique non-empty strings")
    if metadata_loader is None:
        from huggingface_hub import get_safetensors_metadata

        metadata_loader = get_safetensors_metadata
    if revision_resolver is None:
        from huggingface_hub import HfApi

        def resolve_revision(model: str, rev: str) -> str:
            return str(HfApi().model_info(model, revision=rev).sha)

        revision_resolver = resolve_revision
    if file_url is None:
        from huggingface_hub import hf_hub_url

        def resolve_file_url(
            model: str,
            filename: str,
            rev: str,
        ) -> str:
            return hf_hub_url(model, filename, revision=rev)

        file_url = resolve_file_url
    if config_loader is None:
        from huggingface_hub import hf_hub_download

        def load_config(model: str, rev: str) -> bytes:
            return Path(
                hf_hub_download(model, "config.json", revision=rev)
            ).read_bytes()

        config_loader = load_config
    if range_get is None:
        range_get = _default_range_get

    resolved = revision_resolver(repo_id, revision)
    if resolved != revision:
        raise ValueError(
            f"resolved revision {resolved!r} does not match {revision!r}"
        )
    metadata = metadata_loader(repo_id, revision=revision)
    selected_by_file: dict[str, list[tuple[str, Any]]] = {}
    for filename, file_metadata in metadata.files_metadata.items():
        selected = [
            (name, info)
            for name, info in file_metadata.tensors.items()
            if name.startswith(prefixes)
        ]
        if selected:
            selected_by_file[str(filename)] = selected
    if len(selected_by_file) != 1:
        raise ValueError(
            "selected tensors must exist in exactly one safetensors file"
        )
    filename, selected = next(iter(selected_by_file.items()))
    selected.sort(key=lambda item: int(item[1].data_offsets[0]))
    if not selected:
        raise ValueError("tensor selector matched no tensors")
    if any(
        int(left[1].data_offsets[1])
        != int(right[1].data_offsets[0])
        for left, right in zip(selected, selected[1:])
    ):
        raise ValueError(
            "selected tensor payload is not one contiguous byte interval"
        )

    url = file_url(repo_id, filename, revision)
    raw_header_length = range_get(url, 0, 7)
    if len(raw_header_length) != 8:
        raise ValueError("invalid safetensors header-length response")
    header_length = struct.unpack("<Q", raw_header_length)[0]
    data_base = 8 + header_length
    data_start = int(selected[0][1].data_offsets[0])
    data_end = int(selected[-1][1].data_offsets[1])
    payload = range_get(
        url,
        data_base + data_start,
        data_base + data_end - 1,
    )
    expected_bytes = data_end - data_start
    if len(payload) != expected_bytes:
        raise ValueError("selected safetensors payload is truncated")

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_bytes = config_loader(repo_id, revision)
    subset_bytes = _safetensors_bytes(
        selected,
        payload,
        metadata={"format": "pt"},
    )
    for name, content in (
        ("config.json", config_bytes),
        ("model.safetensors", subset_bytes),
    ):
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=root,
            prefix=f".{name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, root / name)

    topology = [
        {
            "name": name,
            "dtype": str(info.dtype),
            "shape": list(info.shape),
            "source_offsets": list(info.data_offsets),
        }
        for name, info in selected
    ]
    files = [
        _file_record(root, root / "config.json"),
        _file_record(root, root / "model.safetensors"),
    ]
    manifest = {
        "schema_version": 1,
        "kind": "selective_hub_safetensors_subset",
        "source": {
            "repo_id": repo_id,
            "revision": revision,
            "resolved_revision": resolved,
            "filename": filename,
            "data_interval": [data_start, data_end],
            "payload_bytes": expected_bytes,
            "payload_sha256": (
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            ),
        },
        "selection": {
            "tensor_prefixes": list(prefixes),
            "tensor_count": len(selected),
            "parameter_count": sum(
                int(info.parameter_count) for _, info in selected
            ),
            "topology_fingerprint": _fingerprint(topology),
            "contiguous": True,
        },
        "output": {
            "path": str(root),
            "files": files,
            "total_bytes": sum(item["bytes"] for item in files),
            "content_fingerprint": _fingerprint(files),
        },
    }
    manifest["manifest_fingerprint"] = _fingerprint(manifest)
    destination = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "selective_checkpoint_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return manifest

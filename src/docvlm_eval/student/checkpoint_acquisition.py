"""Immutable Hugging Face checkpoint acquisition for selective initialization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FAMILY_MODEL_TYPES = {
    "siglip": {"siglip"},
    "llama": {"llama", "mistral", "qwen2", "qwen3"},
    "lfm2": {"lfm2", "lfm2_vl"},
    "student": {"docvlm_student"},
}
_DEFAULT_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)
_WEIGHT_FORMATS = {"safetensors", "pytorch_bin"}


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _repo_filenames(info: Any) -> set[str]:
    filenames = set()
    for sibling in getattr(info, "siblings", ()) or ():
        name = sibling if isinstance(sibling, str) else getattr(sibling, "rfilename", None)
        if isinstance(name, str) and name:
            filenames.add(name)
    return filenames


def _select_weight_download(
    spec: HubCheckpointSpec,
    info: Any,
) -> tuple[tuple[str, ...], str | None]:
    filenames = _repo_filenames(info)
    if not filenames:
        return spec.allow_patterns, None
    allowed = set(spec.allow_patterns)
    candidates = (
        (
            "safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ),
        ("safetensors", "model.safetensors", None),
        (
            "pytorch_bin",
            "pytorch_model.bin.index.json",
            "pytorch_model-*.bin",
        ),
        ("pytorch_bin", "pytorch_model.bin", None),
    )
    for weight_format, primary, shards in candidates:
        if primary not in filenames or primary not in allowed:
            continue
        patterns = ["config.json", primary]
        if shards is not None:
            if shards not in allowed:
                continue
            patterns.append(shards)
        return tuple(patterns), weight_format
    return spec.allow_patterns, None


def _checkpoint_weight_paths(
    snapshot: Path,
    *,
    weight_format: str,
) -> set[Path]:
    if weight_format not in _WEIGHT_FORMATS:
        raise ValueError(f"unsupported checkpoint weight format: {weight_format}")
    if weight_format == "safetensors":
        index_path = snapshot / "model.safetensors.index.json"
        single_path = snapshot / "model.safetensors"
    else:
        index_path = snapshot / "pytorch_model.bin.index.json"
        single_path = snapshot / "pytorch_model.bin"
    weight_paths: set[Path] = set()
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = {snapshot / str(value) for value in (index.get("weight_map") or {}).values()}
        if not shards:
            raise ValueError(f"checkpoint index {index_path.name} has no weight_map")
        missing = sorted(path.name for path in shards if not path.is_file())
        if missing:
            raise ValueError(f"checkpoint index references missing shards: {missing[:3]}")
        weight_paths.update(shards)
        weight_paths.add(index_path)
    elif single_path.is_file():
        weight_paths.add(single_path)
    return weight_paths


@dataclass(frozen=True)
class HubCheckpointSpec:
    repo_id: str
    revision: str
    family: str
    allow_patterns: tuple[str, ...] = _DEFAULT_ALLOW_PATTERNS
    tensor_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError("Hub model repo_id must use namespace/name")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError("Hub model revision must be an immutable 40-character commit SHA")
        if self.family not in _FAMILY_MODEL_TYPES:
            raise ValueError("checkpoint family must be student, siglip, llama, or lfm2")
        if not self.allow_patterns:
            raise ValueError("checkpoint allow_patterns cannot be empty")
        if any(not prefix for prefix in self.tensor_prefixes):
            raise ValueError("checkpoint tensor_prefixes must be non-empty strings")


def validate_checkpoint_snapshot(
    snapshot_path: str | Path,
    spec: HubCheckpointSpec,
    *,
    resolved_revision: str,
    weight_format: str | None = None,
    selected_allow_patterns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    snapshot = Path(snapshot_path).resolve()
    if resolved_revision != spec.revision:
        raise ValueError(
            f"Hub resolved revision {resolved_revision!r} does not match pinned {spec.revision!r}"
        )
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise ValueError("checkpoint snapshot has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "")
    if model_type not in _FAMILY_MODEL_TYPES[spec.family]:
        raise ValueError(
            f"checkpoint model_type={model_type!r} is incompatible with family={spec.family!r}"
        )

    selected_format = weight_format
    if selected_format is None:
        selected_format = (
            "safetensors"
            if (
                (snapshot / "model.safetensors").is_file()
                or (snapshot / "model.safetensors.index.json").is_file()
            )
            else "pytorch_bin"
        )
    weight_paths = _checkpoint_weight_paths(
        snapshot,
        weight_format=selected_format,
    )
    if not weight_paths:
        raise ValueError("checkpoint snapshot has no supported model weights")

    provenance_paths = {config_path, *weight_paths}
    files = [_file_record(snapshot, path) for path in sorted(provenance_paths)]
    return {
        "schema_version": 2,
        "kind": "huggingface_model_checkpoint",
        "spec": asdict(spec),
        "weight_format": selected_format,
        "selected_allow_patterns": list(selected_allow_patterns or spec.allow_patterns),
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "model_type": model_type,
        "files": files,
        "total_bytes": sum(record["bytes"] for record in files),
        "content_fingerprint": _fingerprint(files),
    }


def acquire_hub_checkpoint(
    spec: HubCheckpointSpec,
    manifest_path: str | Path,
    *,
    token: str | None = None,
    snapshot_loader: Callable[..., str] | None = None,
    hub_api: Any | None = None,
) -> dict[str, Any]:
    """Acquire a pinned snapshot in the shared Hub cache and write a run manifest."""

    if spec.tensor_prefixes:
        raise ValueError(
            "tensor-prefix checkpoint specs require selective acquisition"
        )
    if snapshot_loader is None:
        from huggingface_hub import snapshot_download

        snapshot_loader = snapshot_download
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi(token=token)
    info = hub_api.model_info(spec.repo_id, revision=spec.revision)
    resolved_revision = str(info.sha)
    selected_patterns, weight_format = _select_weight_download(spec, info)
    snapshot = snapshot_loader(
        repo_id=spec.repo_id,
        revision=spec.revision,
        token=token,
        allow_patterns=list(selected_patterns),
    )
    manifest = validate_checkpoint_snapshot(
        snapshot,
        spec,
        resolved_revision=resolved_revision,
        weight_format=weight_format,
        selected_allow_patterns=selected_patterns,
    )
    _atomic_write_json(Path(manifest_path), manifest)
    return manifest


def checkpoint_manifest_valid(
    path: str | Path,
    *,
    verify_hashes: bool = False,
) -> bool:
    try:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kind = manifest.get("kind")
        if kind == "huggingface_model_checkpoint":
            snapshot = Path(manifest["snapshot_path"])
            files = manifest.get("files", [])
        elif kind == "selective_hub_safetensors_subset":
            output = manifest["output"]
            snapshot = Path(output["path"])
            files = output.get("files", [])
        else:
            return False
        if not snapshot.is_dir():
            return False
        for record in files:
            candidate = snapshot / str(record["path"])
            if not candidate.is_file() or candidate.stat().st_size != int(record["bytes"]):
                return False
            if verify_hashes and _file_record(snapshot, candidate)["sha256"] != record.get(
                "sha256"
            ):
                return False
        return bool(files)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def checkpoint_path_from_manifest(path: str | Path) -> Path:
    if not checkpoint_manifest_valid(path, verify_hashes=True):
        raise ValueError(f"checkpoint manifest is invalid or incomplete: {path}")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["kind"] == "selective_hub_safetensors_subset":
        return Path(manifest["output"]["path"])
    return Path(manifest["snapshot_path"])

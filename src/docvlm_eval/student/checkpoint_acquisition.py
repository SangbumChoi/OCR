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


@dataclass(frozen=True)
class HubCheckpointSpec:
    repo_id: str
    revision: str
    family: str
    allow_patterns: tuple[str, ...] = _DEFAULT_ALLOW_PATTERNS

    def __post_init__(self) -> None:
        if not self.repo_id or "/" not in self.repo_id:
            raise ValueError("Hub model repo_id must use namespace/name")
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ValueError(
                "Hub model revision must be an immutable 40-character commit SHA"
            )
        if self.family not in _FAMILY_MODEL_TYPES:
            raise ValueError(
                "checkpoint family must be student, siglip, or llama"
            )
        if not self.allow_patterns:
            raise ValueError("checkpoint allow_patterns cannot be empty")


def validate_checkpoint_snapshot(
    snapshot_path: str | Path,
    spec: HubCheckpointSpec,
    *,
    resolved_revision: str,
) -> dict[str, Any]:
    snapshot = Path(snapshot_path).resolve()
    if resolved_revision != spec.revision:
        raise ValueError(
            f"Hub resolved revision {resolved_revision!r} does not match pinned "
            f"{spec.revision!r}"
        )
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise ValueError("checkpoint snapshot has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = str(config.get("model_type") or "")
    if model_type not in _FAMILY_MODEL_TYPES[spec.family]:
        raise ValueError(
            f"checkpoint model_type={model_type!r} is incompatible with "
            f"family={spec.family!r}"
        )

    index_paths = [
        snapshot / "model.safetensors.index.json",
        snapshot / "pytorch_model.bin.index.json",
    ]
    weight_paths: set[Path] = set()
    for index_path in index_paths:
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = {
            snapshot / str(value)
            for value in (index.get("weight_map") or {}).values()
        }
        if not shards:
            raise ValueError(f"checkpoint index {index_path.name} has no weight_map")
        missing = sorted(path.name for path in shards if not path.is_file())
        if missing:
            raise ValueError(
                f"checkpoint index references missing shards: {missing[:3]}"
            )
        weight_paths.update(shards)
        weight_paths.add(index_path)
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = snapshot / name
        if candidate.is_file():
            weight_paths.add(candidate)
    if not any(
        path.suffix in {".safetensors", ".bin"}
        for path in weight_paths
    ):
        raise ValueError("checkpoint snapshot has no supported model weights")

    provenance_paths = {config_path, *weight_paths}
    files = [
        _file_record(snapshot, path)
        for path in sorted(provenance_paths)
    ]
    return {
        "schema_version": 1,
        "kind": "huggingface_model_checkpoint",
        "spec": asdict(spec),
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

    if snapshot_loader is None:
        from huggingface_hub import snapshot_download

        snapshot_loader = snapshot_download
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi(token=token)
    info = hub_api.model_info(spec.repo_id, revision=spec.revision)
    resolved_revision = str(info.sha)
    snapshot = snapshot_loader(
        repo_id=spec.repo_id,
        revision=spec.revision,
        token=token,
        allow_patterns=list(spec.allow_patterns),
    )
    manifest = validate_checkpoint_snapshot(
        snapshot,
        spec,
        resolved_revision=resolved_revision,
    )
    _atomic_write_json(Path(manifest_path), manifest)
    return manifest


def checkpoint_manifest_valid(path: str | Path) -> bool:
    try:
        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = Path(manifest["snapshot_path"])
        if (
            manifest.get("kind") != "huggingface_model_checkpoint"
            or not snapshot.is_dir()
        ):
            return False
        for record in manifest.get("files", []):
            candidate = snapshot / str(record["path"])
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(record["bytes"])
            ):
                return False
        return bool(manifest.get("files"))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def checkpoint_path_from_manifest(path: str | Path) -> Path:
    if not checkpoint_manifest_valid(path):
        raise ValueError(f"checkpoint manifest is invalid or incomplete: {path}")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return Path(manifest["snapshot_path"])

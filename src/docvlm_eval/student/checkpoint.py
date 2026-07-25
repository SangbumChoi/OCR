"""Checkpoint loading helpers for native, PyTorch, and Hugging Face state dictionaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _language_config(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ("language", "text_config", "language_config", "llm_config"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return raw


def load_checkpoint_attention_geometry(
    path: str | Path,
    *,
    family: str,
) -> dict[str, Any] | None:
    """Read language attention geometry without loading checkpoint tensors."""
    path = Path(path)
    root = path.parent if path.is_file() else path
    config_names = (
        ("student_config.json", "config.json")
        if family == "student"
        else ("config.json",)
    )
    config_path = next(
        (root / name for name in config_names if (root / name).is_file()),
        None,
    )
    if config_path is None:
        return None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    language = _language_config(raw)
    hidden = language.get("width", language.get("hidden_size"))
    heads = language.get(
        "attention_heads",
        language.get("num_attention_heads"),
    )
    kv_heads = language.get(
        "kv_heads",
        language.get("num_key_value_heads"),
    )
    rope_base = language.get(
        "rope_base",
        language.get("rope_theta"),
    )
    if rope_base is None:
        rope_parameters = language.get("rope_parameters")
        if isinstance(rope_parameters, dict):
            rope_base = rope_parameters.get("rope_theta")
    if any(value is None for value in (hidden, heads, kv_heads, rope_base)):
        return None
    hidden = int(hidden)
    heads = int(heads)
    kv_heads = int(kv_heads)
    if hidden <= 0 or heads <= 0 or kv_heads <= 0 or hidden % heads:
        return None
    model_type = str(language.get("model_type") or raw.get("model_type") or "")
    known_bias_free = model_type in {
        "lfm2",
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
    }
    return {
        "hidden_width": hidden,
        "attention_heads": heads,
        "kv_heads": kv_heads,
        "head_dim": int(language.get("head_dim") or hidden // heads),
        "rope_base": float(rope_base),
        "rope_layout": str(
            language.get("rope_layout")
            or ("half_split" if known_bias_free else "interleaved")
        ),
        "norm_eps": float(
            language.get(
                "norm_eps",
                language.get("rms_norm_eps", 1e-6),
            )
        ),
        "qk_norm": bool(
            language.get("qk_norm", model_type == "lfm2")
        ),
        "attention_bias": bool(
            language.get("attention_bias", not known_bias_free)
        ),
        "mlp_bias": bool(
            language.get("mlp_bias", not known_bias_free)
        ),
        "conv_kernel_size": int(
            language.get(
                "conv_kernel_size",
                language.get("conv_L_cache", 3),
            )
        ),
        "conv_bias": bool(language.get("conv_bias", False)),
    }


def _load_one(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "loading safetensors requires `pip install safetensors`"
            ) from exc
        return load_file(str(path), device="cpu")

    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
        return loaded["state_dict"]
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a state dictionary")
    return loaded


def _checkpoint_files(path: Path) -> tuple[Path, list[Path]]:
    if path.is_file():
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    configs = [
        path / name
        for name in ("student_config.json", "config.json")
        if (path / name).is_file()
    ]
    for name in ("model.pt", "model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return path, [*configs, candidate]
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = path / index_name
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(
            {
                path / str(value)
                for value in (index.get("weight_map") or {}).values()
            }
        )
        if not shards:
            raise ValueError(f"{index_path} has no weight_map shards")
        missing = [str(shard) for shard in shards if not shard.is_file()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint index references missing shards: {missing[:3]}"
            )
        return path, [*configs, index_path, *shards]
    raise FileNotFoundError(f"no supported checkpoint weights found under {path}")


def checkpoint_content_identity(
    path: str | Path,
) -> dict[str, Any]:
    """Hash exactly the config and weight files consumed by checkpoint loading."""

    checkpoint = Path(path)
    root, files = _checkpoint_files(checkpoint)
    records = []
    for candidate in sorted(set(files)):
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": (
                    candidate.name
                    if checkpoint.is_file()
                    else str(candidate.relative_to(root))
                ),
                "bytes": candidate.stat().st_size,
                "sha256": f"sha256:{digest.hexdigest()}",
            }
        )
    payload = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "kind": "checkpoint_content",
        "files": records,
        "total_bytes": sum(record["bytes"] for record in records),
        "content_fingerprint": (
            f"sha256:{hashlib.sha256(payload).hexdigest()}"
        ),
    }


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    """Load one state dictionary from a native or Hugging Face checkpoint path."""
    path = Path(path)
    if path.is_file():
        return _load_one(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    for name in ("model.pt", "model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.exists():
            return _load_one(candidate)

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            raise ValueError(f"{index_path} has no weight_map shards")
        merged: dict[str, Any] = {}
        for shard in shards:
            shard_state = _load_one(path / shard)
            overlap = merged.keys() & shard_state.keys()
            if overlap:
                raise ValueError(f"duplicate tensors across checkpoint shards: {sorted(overlap)[:3]}")
            merged.update(shard_state)
        return merged
    raise FileNotFoundError(f"no supported checkpoint weights found under {path}")

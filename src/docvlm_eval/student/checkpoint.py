"""Checkpoint loading helpers for native, PyTorch, and Hugging Face state dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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

#!/usr/bin/env python3
"""Acquire a contiguous tensor subset from an immutable Hub safetensors file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock

from docvlm_eval.student.checkpoint_acquisition import (
    checkpoint_manifest_valid,
)
from docvlm_eval.student.selective_checkpoint import (
    materialize_contiguous_safetensors_subset,
)


def _cached_manifest(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    tensor_prefixes: list[str],
) -> dict | None:
    if not checkpoint_manifest_valid(path, verify_hashes=True):
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    source = manifest.get("source") or {}
    selection = manifest.get("selection") or {}
    if (
        manifest.get("kind") != "selective_hub_safetensors_subset"
        or source.get("repo_id") != repo_id
        or source.get("revision") != revision
        or selection.get("tensor_prefixes")
        != sorted(set(tensor_prefixes))
    ):
        return None
    return manifest


def _atomic_manifest_copy(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tensor-prefix", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir = args.output_dir.resolve()
    cache_manifest = args.output_dir / "selective_checkpoint_manifest.json"
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{args.output_dir}.lock"):
        manifest = _cached_manifest(
            cache_manifest,
            repo_id=args.repo_id,
            revision=args.revision,
            tensor_prefixes=args.tensor_prefix,
        )
        reused = manifest is not None
        if manifest is None:
            manifest = materialize_contiguous_safetensors_subset(
                repo_id=args.repo_id,
                revision=args.revision,
                tensor_prefixes=args.tensor_prefix,
                output=args.output_dir,
                manifest_path=cache_manifest,
            )
        _atomic_manifest_copy(args.manifest.resolve(), manifest)
    summary = {
        "kind": manifest["kind"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "output": manifest["output"]["path"],
        "payload_bytes": manifest["source"]["payload_bytes"],
        "reused": reused,
        "tensor_count": manifest["selection"]["tensor_count"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Acquire a contiguous tensor subset from an immutable Hub safetensors file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.student.selective_checkpoint import (
    materialize_contiguous_safetensors_subset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tensor-prefix", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = materialize_contiguous_safetensors_subset(
        repo_id=args.repo_id,
        revision=args.revision,
        tensor_prefixes=args.tensor_prefix,
        output=args.output_dir,
        manifest_path=args.manifest,
    )
    summary = {
        "kind": manifest["kind"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "output": manifest["output"]["path"],
        "payload_bytes": manifest["source"]["payload_bytes"],
        "tensor_count": manifest["selection"]["tensor_count"],
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

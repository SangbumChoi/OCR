#!/usr/bin/env python3
"""Materialize and verify the pinned SmolVLM2 vision-block transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from docvlm_eval.architecture import load_blueprint
from docvlm_eval.student.checkpoint import (
    checkpoint_content_identity,
    load_checkpoint_state,
)
from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.model import VisionTower
from docvlm_eval.student.selective_checkpoint import (
    materialize_contiguous_safetensors_subset,
)
from docvlm_eval.student.transfer import selective_transfer


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
PREFIX = "model.vision_model.encoder.layers."


def _fingerprint(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class _VisionTarget(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision = VisionTower(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "sources"
            / "smolvlm2-500m-vision-blocks"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_smol_vision_real_source_preflight.json"
        ),
    )
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    manifest_path = (
        args.checkpoint / "selective_checkpoint_manifest.json"
    )
    if args.reuse and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = materialize_contiguous_safetensors_subset(
            repo_id=REPO_ID,
            revision=REVISION,
            tensor_prefixes=[PREFIX],
            output=args.checkpoint,
        )
    config = StudentConfig.from_blueprint(
        load_blueprint(ROOT / "configs" / "sub1b_architecture.yaml")
    )
    torch.manual_seed(0)
    target = _VisionTarget(config.vision)
    source_identity = checkpoint_content_identity(args.checkpoint)
    report = selective_transfer(
        target,
        load_checkpoint_state(args.checkpoint),
        {"vision": 1.0},
        family="siglip",
        vision_scope="transformer_blocks",
        shape_policy="exact",
        source_identity=source_identity,
        require_healthy_source_weights=True,
    ).to_dict()
    target_parameters = sum(
        parameter.numel() for parameter in target.parameters()
    )
    copied_keys = report["copied_keys"]
    result = {
        "schema_version": 1,
        "claim_scope": "real_source_vision_initialization_contract_only",
        "executed_on": "cpu",
        "source": {
            "model": REPO_ID,
            "revision": REVISION,
            "selective_manifest_fingerprint": manifest[
                "manifest_fingerprint"
            ],
            "source_payload_bytes": manifest["source"]["payload_bytes"],
            "selected_content_fingerprint": source_identity[
                "content_fingerprint"
            ],
            "selected_tensor_count": manifest["selection"][
                "tensor_count"
            ],
            "selected_parameter_count": manifest["selection"][
                "parameter_count"
            ],
        },
        "target": {
            "component": "vision",
            "parameters": target_parameters,
            "architecture": "docvlm-production-vision",
        },
        "transfer": {
            "family": report["family"],
            "vision_scope": report["vision_scope"],
            "shape_policy": report["shape_policy"],
            "copied_tensors": report["copied_tensors"],
            "copied_parameters": report["copied_parameters"],
            "realized_component_parameter_fraction": (
                report["copied_parameters"] / target_parameters
            ),
            "all_copied_keys_within_scope": bool(copied_keys)
            and all(
                key.startswith("vision.blocks.") for key in copied_keys
            ),
            "shape_skips": len(report["skipped_shape"]),
            "semantic_skips": len(report["skipped_semantic"]),
            "missing_source": len(report["missing_source"]),
            "unhealthy_source_weight_roles": report[
                "unhealthy_source_weight_roles"
            ],
            "mapping_fingerprint": report["mapping_fingerprint"],
            "copied_values_fingerprint": report[
                "copied_values_fingerprint"
            ],
            "value_verified": report["value_verified"],
        },
        "quality_claim_authorized": False,
        "promotion_claim_authorized": False,
        "limitations": [
            "This verifies selective source acquisition and initialization values only.",
            "Downstream benefit requires a matched language-only versus dual-source training experiment.",
        ],
    }
    result["report_fingerprint"] = _fingerprint(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "copied_parameters": report["copied_parameters"],
                "copied_tensors": report["copied_tensors"],
                "output": str(args.output.resolve()),
                "payload_bytes": manifest["source"]["payload_bytes"],
                "scope_valid": result["transfer"][
                    "all_copied_keys_within_scope"
                ],
                "value_verified": report["value_verified"],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

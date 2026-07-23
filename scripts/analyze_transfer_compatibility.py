#!/usr/bin/env python3
"""Compare a pinned Hub checkpoint's tensor shapes with the native student."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.checkpoint_acquisition import HubCheckpointSpec
from docvlm_eval.student.config import StudentConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--family",
        required=True,
        choices=["student", "siglip", "llama"],
    )
    parser.add_argument(
        "--component",
        required=True,
        choices=["vision", "language"],
    )
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from huggingface_hub import HfApi, get_safetensors_metadata

    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.transfer import selective_transfer

    spec = HubCheckpointSpec(
        repo_id=args.repo_id,
        revision=args.revision,
        family=args.family,
    )
    if not 0.0 <= args.fraction <= 1.0:
        raise SystemExit("--fraction must be within [0, 1]")
    resolved_revision = str(
        HfApi().model_info(spec.repo_id, revision=spec.revision).sha
    )
    if resolved_revision != spec.revision:
        raise SystemExit(
            f"resolved revision {resolved_revision} does not match {spec.revision}"
        )
    metadata = get_safetensors_metadata(
        spec.repo_id,
        revision=spec.revision,
    )
    source_shapes = {
        key: tuple(tensor.shape)
        for file_metadata in metadata.files_metadata.values()
        for key, tensor in file_metadata.tensors.items()
    }
    shape_records = sorted(
        (key, list(shape))
        for key, shape in source_shapes.items()
    )
    blueprint = load_blueprint(args.config)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(errors))
    with torch.device("meta"):
        student = DocumentVLMStudent(
            StudentConfig.from_blueprint(blueprint)
        )
        source = {
            key: torch.empty(shape)
            for key, shape in source_shapes.items()
        }
    report = selective_transfer(
        student,
        source,
        {args.component: args.fraction},
        family=spec.family,
    ).to_dict()
    student_parameters = sum(
        parameter.numel()
        for parameter in student.parameters()
    )
    result = {
        "schema_version": 1,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "resolved_revision": resolved_revision,
        "family": spec.family,
        "component": args.component,
        "fraction": args.fraction,
        "source_tensors": len(source_shapes),
        "source_shape_fingerprint": (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    shape_records,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ),
        "student_parameters": student_parameters,
        "compatible_tensors": report["copied_tensors"],
        "compatible_parameters": report["copied_parameters"],
        "student_fraction": report["copied_parameters"] / student_parameters,
        "shape_mismatches": len(report["skipped_shape"]),
        "missing_source": len(report["missing_source"]),
        "report": report,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()

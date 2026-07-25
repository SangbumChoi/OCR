#!/usr/bin/env python3
"""Build or validate bounded real-weight evidence across pinned small VLMs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from docvlm_eval.student.architecture_commonality import (
    load_architecture_catalog,
)
from docvlm_eval.student.weight_commonality import (
    build_weight_commonality_report,
    load_weight_commonality_report,
    refresh_weight_commonality_report,
    validate_weight_commonality_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _markdown(report: dict) -> str:
    commonality = report["commonality"]
    lines = [
        "# Cross-architecture weight commonality",
        "",
        "This report samples real weights from immutable public checkpoints without "
        "downloading full model files. Each tensor contributes at most three evenly "
        "spaced byte windows. Raw values are discarded; only aggregate statistics and "
        "content fingerprints are retained.",
        "",
        "These statistics compare operator distributions, not neuron coordinates. "
        "A similar scale never establishes basis alignment and cannot by itself authorize "
        "a direct copy.",
        "",
        "## Evidence budget",
        "",
        "| Model | Roles | Tensors | Values | Bytes read |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in report["models"]:
        lines.append(
            f"| `{model['model_id']}` | {model['sampled_roles']} | "
            f"{model['sampled_tensors']} | {model['sampled_values']} | "
            f"{model['bytes_read']:,} |"
        )
    lines.extend(
        [
            "",
            "## Recurrent weight characteristics",
            "",
            "| Semantic role | Models | Scaled RMS ratio | Median zeros | Stable | Transfer rule |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in commonality["common_roles"]:
        lines.append(
            f"| `{item['role']}` | {item['model_count']} | "
            f"{item['scaled_rms_ratio']:.3f} | "
            f"{item['median_zero_fraction']:.3g} | "
            f"{'yes' if item['stable_across_models'] else 'no'} | "
            f"`{item['transfer_rule']}` |"
        )
    lines.extend(
        [
            "",
            "## Selective-transfer contract",
            "",
            "- Exact copy requires a stable sampled source plus matching semantic role, shape, "
            "normalization, attention geometry, and position convention.",
            "- Wider SwiGLU transfer uses one joint channel selection across gate, up, and down "
            "matrices; independent cropping is forbidden.",
            "- Token embeddings require an explicit tokenizer identity map.",
            "- Cross-model scale instability removes the population prior but does not veto a "
            "healthy pairwise transfer that passes the full semantic and geometry preflight.",
            "- A source role with non-finite, degenerate, sparse, or extreme sampled weights is "
            "distillation-only.",
            "- Topology mismatches remain feature or relation distillation targets even when "
            "their aggregate weight scales look similar.",
            "",
            f"Report fingerprint: `{report['report_fingerprint']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "small_vlm_architectures.yaml",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "small_vlm_weight_commonality.json"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "report"
            / "small_vlm_weight_commonality.md"
        ),
    )
    parser.add_argument("--max-tensors-per-role", type=int, default=3)
    parser.add_argument("--max-values-per-tensor", type=int, default=2048)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-derived", action="store_true")
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()

    profiles = load_architecture_catalog(args.catalog)
    if args.validate_only and args.refresh_derived:
        raise SystemExit("--validate-only and --refresh-derived are exclusive")
    if args.validate_only:
        report = load_weight_commonality_report(args.json_output)
    elif args.refresh_derived:
        report = refresh_weight_commonality_report(
            load_weight_commonality_report(args.json_output)
        )
        _atomic_write(
            args.json_output,
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
        )
        _atomic_write(args.report_output, _markdown(report))
    else:
        report = build_weight_commonality_report(
            profiles,
            max_tensors_per_role=args.max_tensors_per_role,
            max_values_per_tensor=args.max_values_per_tensor,
            max_workers=args.max_workers,
        )
        _atomic_write(
            args.json_output,
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
        )
        _atomic_write(args.report_output, _markdown(report))

    audit = validate_weight_commonality_report(report, profiles)
    if args.audit_output is not None:
        _atomic_write(
            args.audit_output,
            json.dumps(audit, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
        )
    print(
        "weight commonality audit: "
        f"{audit['status']} ({audit['source_count']} sources, "
        f"{audit['stable_role_count']} stable roles)"
    )
    if audit["status"] != "pass":
        raise SystemExit("\n".join(audit["errors"]))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compose architecture and real-weight evidence into transfer decisions."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from docvlm_eval.student.architecture_commonality import (
    load_architecture_catalog,
)
from docvlm_eval.student.source_selection import (
    build_source_selection_matrix,
    validate_source_selection_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write(path: Path, content: str) -> None:
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


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Selective-transfer source matrix",
        "",
        "This matrix composes pinned architecture compatibility, bounded real-weight "
        "health sketches, and available real-payload execution evidence. It never "
        "treats similar weight distributions as neuron-basis alignment.",
        "",
        "## Decisions",
        "",
    ]
    for target in report["targets"]:
        lines.extend(
            [
                f"### `{target['target']}`",
                "",
                "| Source | Direct | Structured | Token map | Payload preflight | "
                "Distill | Real payload |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for source in target["sources"]:
            counts = source["action_counts"]
            lines.append(
                f"| `{source['source']}` | "
                f"{counts.get('direct_copy_candidate', 0)} | "
                f"{counts.get('structured_transfer_candidate', 0)} | "
                f"{counts.get('token_identity_map_required', 0)} | "
                f"{counts.get('pairwise_payload_preflight_required', 0)} | "
                f"{counts.get('feature_or_relation_distillation', 0)} | "
                f"`{source['real_payload_evidence']['status']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Research decision",
            "",
            "- The native 800M target has no sampled-and-topology-qualified language "
            "copy source in this five-model set. Use language feature, relation, or "
            "sequence distillation unless a separate pairwise source preflight passes.",
            "- The LFM-aligned 814M target makes LFM attention and short convolution "
            "direct-copy candidates and its reduced SwiGLU a structured-transfer "
            "candidate. The recorded real-payload run verifies this language transfer "
            "at 80.49% coverage with zero shape, semantic, or missing-source skips.",
            "- SmolVLM2 is a vision-block candidate for both targets, but it remains "
            "unexecuted pairwise evidence and must not be combined with the LFM result "
            "as though a dual-source checkpoint had been tested.",
            "- Position weights without sampled semantic-role evidence require a "
            "pairwise payload preflight even when the config convention matches.",
            "- Token embeddings remain identity-map gated. Vocabulary width equality "
            "does not establish token identity.",
            "",
            "## Claim boundary",
            "",
            "This artifact selects experiments; it does not establish downstream "
            "quality or authorize promotion. Direct and structured candidates still "
            "require pairwise payload checks, and empirical benefit requires matched "
            "random-initialized controls.",
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
        "--architecture-report",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "small_vlm_architecture_commonality.json"
        ),
    )
    parser.add_argument(
        "--weight-report",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "small_vlm_weight_commonality.json"
        ),
    )
    parser.add_argument(
        "--real-payload-preflight",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_lfm_real_source_preflight.json"
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "selective_transfer_source_matrix.json"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "report"
            / "selective_transfer_source_matrix.md"
        ),
    )
    args = parser.parse_args()

    architecture_report = _read(args.architecture_report)
    weight_report = _read(args.weight_report)
    profiles = load_architecture_catalog(args.catalog)
    real_payload_preflight = _read(args.real_payload_preflight)
    report = build_source_selection_matrix(
        architecture_report,
        weight_report,
        profiles,
        real_payload_preflight=real_payload_preflight,
    )
    audit = validate_source_selection_matrix(
        report,
        architecture_report=architecture_report,
        weight_report=weight_report,
        profiles=profiles,
        real_payload_preflight=real_payload_preflight,
    )
    if audit["status"] != "pass":
        raise SystemExit("\n".join(audit["errors"]))
    _write(
        args.json_output,
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
    )
    _write(args.report_output, _markdown(report))
    print(
        json.dumps(
            {
                "status": audit["status"],
                "targets": len(report["targets"]),
                "report_fingerprint": report["report_fingerprint"],
                "json_output": str(args.json_output.resolve()),
                "report_output": str(args.report_output.resolve()),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

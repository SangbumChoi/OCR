"""Schema validation and report rendering for the frontier-method catalog."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


EXPECTED_CATEGORY_COUNTS = {
    "vision_encoder": 10,
    "connector_resolution": 10,
    "document_model": 15,
    "small_llm": 10,
    "transfer_distillation": 15,
    "data_construction": 15,
    "posttraining_rl": 15,
    "objective_reliability": 10,
}
DECISIONS = {"adopt", "ablate", "reference", "reject"}
REQUIRED_FIELDS = {
    "id",
    "name",
    "year",
    "organization",
    "category",
    "stage",
    "source",
    "benefit",
    "limitation",
    "decision",
    "knobs",
}

CATEGORY_TITLES = {
    "vision_encoder": "Vision encoders and visual pretraining",
    "connector_resolution": "Connectors, compression, and resolution",
    "document_model": "Document-specialized modeling",
    "small_llm": "Small language-model architecture and systems",
    "transfer_distillation": "Selective transfer, distillation, and compression",
    "data_construction": "Data construction, curation, and hard domains",
    "posttraining_rl": "Post-training, preferences, and reinforcement learning",
    "objective_reliability": "Objectives, multitask balance, and reliability",
}


def load_method_catalog(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            rows.append(row)
    return rows


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _validate_evidence_locator(
    method_id: str,
    kind: str,
    locator: Any,
    *,
    repo_root: Path,
) -> tuple[list[str], set[str], dict[str, Any] | None]:
    errors = []
    if not isinstance(locator, dict):
        return [f"{method_id}: {kind} locator must be a mapping"], set(), None
    raw_path = locator.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [f"{method_id}: {kind} locator requires a path"], set(), None
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{method_id}: {kind} path must stay within the repository")
        return errors, set(), None
    if kind == "implementation" and path.parts[0] not in {
        "configs",
        "scripts",
        "src",
    }:
        errors.append(
            f"{method_id}: implementation path must be under configs, scripts, or src"
        )
    if kind == "verification" and path.parts[0] != "tests":
        errors.append(f"{method_id}: verification path must be under tests")
    resolved = repo_root / path
    if not resolved.is_file():
        errors.append(f"{method_id}: evidence file does not exist: {raw_path}")
        return errors, set(), None
    anchors = locator.get("anchors")
    if (
        not isinstance(anchors, list)
        or not anchors
        or any(not isinstance(anchor, str) or not anchor for anchor in anchors)
    ):
        errors.append(f"{method_id}: {kind} locator requires non-empty anchors")
        return errors, set(), None
    content = resolved.read_text(encoding="utf-8")
    missing = [anchor for anchor in anchors if anchor not in content]
    if missing:
        errors.append(
            f"{method_id}: stale {kind} anchors in {raw_path}: {missing}"
        )
    knobs = locator.get("knobs", [])
    if kind == "implementation" and (
        not isinstance(knobs, list)
        or any(not isinstance(knob, str) or not knob for knob in knobs)
    ):
        errors.append(f"{method_id}: implementation knobs must be a list")
        knobs = []
    if kind == "verification" and "knobs" in locator:
        errors.append(f"{method_id}: verification locators cannot bind knobs")
    record = {
        "path": raw_path,
        "anchors": len(anchors),
        "sha256": f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}",
    }
    return errors, set(knobs), record


def audit_adopted_method_evidence(
    rows: Iterable[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Prove that every adopted method has live code and verification anchors."""

    rows = list(rows)
    catalog_errors = validate_method_catalog(rows)
    if catalog_errors:
        raise ValueError(
            "invalid method catalog:\n"
            + "\n".join(f"- {error}" for error in catalog_errors)
        )
    if not isinstance(evidence, dict) or evidence.get("schema_version") != 1:
        raise ValueError("method evidence schema_version must be 1")
    records = evidence.get("methods")
    if not isinstance(records, dict):
        raise ValueError("method evidence methods must be a mapping")
    by_id = {str(row["id"]): row for row in rows}
    adopted = {
        method_id: row
        for method_id, row in by_id.items()
        if row["decision"] == "adopt"
    }
    errors = []
    missing = sorted(set(adopted) - set(records))
    unknown = sorted(set(records) - set(by_id))
    non_adopted = sorted(set(records) - set(adopted))
    if missing:
        errors.append(f"adopted methods lack evidence: {missing}")
    if unknown:
        errors.append(f"evidence references unknown methods: {unknown}")
    if non_adopted:
        errors.append(f"evidence certifies non-adopted methods: {non_adopted}")

    root = Path(repo_root).resolve()
    method_reports = {}
    for method_id, row in adopted.items():
        record = records.get(method_id)
        if not isinstance(record, dict):
            continue
        claim = record.get("claim")
        if not isinstance(claim, str) or len(claim.split()) < 6:
            errors.append(f"{method_id}: evidence claim is too short")
        implementation = record.get("implementation")
        verification = record.get("verification")
        if not isinstance(implementation, list) or not implementation:
            errors.append(f"{method_id}: implementation evidence is required")
            implementation = []
        if not isinstance(verification, list) or not verification:
            errors.append(f"{method_id}: verification evidence is required")
            verification = []
        covered_knobs: set[str] = set()
        implementation_records = []
        verification_records = []
        for locator in implementation:
            locator_errors, knobs, locator_record = _validate_evidence_locator(
                method_id,
                "implementation",
                locator,
                repo_root=root,
            )
            errors.extend(locator_errors)
            covered_knobs.update(knobs)
            if locator_record is not None:
                implementation_records.append(locator_record)
        for locator in verification:
            locator_errors, _, locator_record = _validate_evidence_locator(
                method_id,
                "verification",
                locator,
                repo_root=root,
            )
            errors.extend(locator_errors)
            if locator_record is not None:
                verification_records.append(locator_record)
        expected_knobs = set(str(knob) for knob in row["knobs"])
        if covered_knobs != expected_knobs:
            errors.append(
                f"{method_id}: knob evidence mismatch; "
                f"missing={sorted(expected_knobs - covered_knobs)}, "
                f"extra={sorted(covered_knobs - expected_knobs)}"
            )
        method_reports[method_id] = {
            "name": row["name"],
            "stage": row["stage"],
            "claim": claim,
            "knobs": sorted(covered_knobs),
            "implementation": implementation_records,
            "verification": verification_records,
        }
    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "catalog_methods": len(rows),
        "adopted_methods": len(adopted),
        "certified_methods": len(method_reports),
        "errors": errors,
        "methods": method_reports,
    }
    report["fingerprint"] = _fingerprint(report)
    return report


def validate_method_catalog(rows: Iterable[dict[str, Any]]) -> list[str]:
    rows = list(rows)
    errors: list[str] = []
    if len(rows) != 100:
        errors.append(f"catalog must contain exactly 100 methods, found {len(rows)}")

    ids: list[str] = []
    sources: list[str] = []
    for index, row in enumerate(rows, 1):
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"row {index} is missing: {', '.join(missing)}")
            continue
        method_id = str(row["id"])
        ids.append(method_id)
        sources.append(str(row["source"]))
        if row["category"] not in EXPECTED_CATEGORY_COUNTS:
            errors.append(f"{method_id}: unknown category {row['category']!r}")
        if row["decision"] not in DECISIONS:
            errors.append(f"{method_id}: unknown decision {row['decision']!r}")
        if not str(row["source"]).startswith("https://arxiv.org/abs/"):
            errors.append(f"{method_id}: source must be a primary arXiv abstract URL")
        if not 2014 <= int(row["year"]) <= 2026:
            errors.append(f"{method_id}: year is outside the catalog scope")
        if len(str(row["benefit"]).split()) < 8:
            errors.append(f"{method_id}: benefit is too short to be useful")
        if len(str(row["limitation"]).split()) < 8:
            errors.append(f"{method_id}: limitation is too short to be useful")
        if not isinstance(row["knobs"], list) or not row["knobs"]:
            errors.append(f"{method_id}: knobs must be a non-empty list")

    duplicate_ids = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate method ids: {', '.join(duplicate_ids)}")
    duplicate_sources = sorted(key for key, count in Counter(sources).items() if count > 1)
    if duplicate_sources:
        errors.append(f"duplicate primary sources: {', '.join(duplicate_sources)}")

    organizations = {str(row.get("organization", "")) for row in rows}
    if len(organizations) < 25:
        errors.append(f"catalog must span at least 25 organizations, found {len(organizations)}")
    knobs = {
        str(knob)
        for row in rows
        if isinstance(row.get("knobs"), list)
        for knob in row["knobs"]
    }
    if len(knobs) < 100:
        errors.append(f"catalog must expose at least 100 distinct knobs, found {len(knobs)}")

    actual_counts = Counter(row.get("category") for row in rows)
    for category, expected in EXPECTED_CATEGORY_COUNTS.items():
        if actual_counts[category] != expected:
            errors.append(
                f"{category}: expected {expected} methods, found {actual_counts[category]}"
            )
    return errors


def _escape_table(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_method_survey(rows: Iterable[dict[str, Any]]) -> str:
    rows = list(rows)
    errors = validate_method_catalog(rows)
    if errors:
        raise ValueError("invalid method catalog:\n" + "\n".join(f"- {error}" for error in errors))

    decisions = Counter(row["decision"] for row in rows)
    organization_count = len({row["organization"] for row in rows})
    knob_count = len({knob for row in rows for knob in row["knobs"]})
    years = [int(row["year"]) for row in rows]
    lines = [
        "# Frontier method catalog for a sub-1B document VLM",
        "",
        "_Generated from "
        "[`configs/frontier_method_catalog.jsonl`](../../configs/frontier_method_catalog.jsonl) "
        "by `scripts/build_method_survey.py`._",
        "",
        "## Scope and decision rule",
        "",
        "This catalog evaluates exactly **100 primary methods** as design choices for the adjustable",
        "approximately 790M document VLM. It is not a leaderboard and does not assume that a method",
        "validated at 7B or on natural images transfers to this regime. Each row records the useful",
        "mechanism, the likely failure mode below one billion parameters, and one of four decisions:",
        "",
        "- **adopt**: part of the default blueprint or its required training machinery;",
        "- **ablate**: plausible, but must beat a matched control before entering the default;",
        "- **reference**: useful baseline or evidence, not currently worth default complexity;",
        "- **reject**: conflicts with measured results, deployment budget, or trustworthy supervision.",
        "",
        f"Decision totals: **{decisions['adopt']} adopt**, **{decisions['ablate']} ablate**, "
        f"**{decisions['reference']} reference**, **{decisions['reject']} reject**.",
        f"The sources span **{organization_count} organizations**, **{min(years)}-{max(years)}**, "
        f"and expose **{knob_count} distinct adjustable knobs**.",
        "",
        "Adoption is separately fail-closed by "
        "[`configs/frontier_method_evidence.yaml`](../../configs/frontier_method_evidence.yaml):",
        "every adopted method must bind all catalog knobs to live implementation anchors and at",
        "least one live verification anchor. The production experiment runs this audit before any",
        "model-allocating stage.",
        "",
        "## Recommended end-to-end stack",
        "",
        "1. **Visual input:** ViT with native-resolution packing, SigLIP-style contrastive transfer,",
        "   and a fixed visual-token budget. Compare full SigLIP2 transfer with partial and random arms.",
        "2. **Alignment:** a small Perceiver-like gated resampler; sweep latent-token count against",
        "   small-text OCR, cross-region reasoning, latency, and memory.",
        "3. **Decoder:** RMSNorm, SwiGLU, RoPE, and grouped-query attention as the stable control;",
        "   ablate LFM-style gated short convolution at alternating and sparse attention ratios.",
        "4. **Initialization:** alternating-layer and structured-pruning transfer controls plus logit,",
        "   hidden-state, and bounded token-relation distillation. Incompatible widths use distillation,",
        "   never silent tensor surgery.",
        "5. **Data:** authored document graphs, multilingual typography, public-data replay, and",
        "   fixed-compute domain reweighting. Financial programs, hybrid table-text evidence, charts,",
        "   and scientific figure context are explicit families, not generic VQA.",
        "6. **Post-training:** grounded SFT followed by GRPO/Visual-RFT with decomposed verifiable",
        "   rewards. Free rationales and self-judging are excluded unless evidence gates overturn the",
        "   current negative result.",
        "7. **Reliability:** GIoU-based grounding, calibrated confidence, selective-risk evaluation,",
        "   and gradient-conflict ablations across task and language slices.",
        "",
        "## Critical interpretation",
        "",
        "- A method is not accepted because its source reports a high score. The relevant proof is a",
        "  matched run at the same student size, data tokens, visual tokens, and deployment budget.",
        "- Generic semantic vision pretraining can hurt exact OCR. Every transfer arm is therefore",
        "  evaluated on character error, rare scripts, box IoU, and counterfactual evidence edits.",
        "- The measured UDD experiment found rationale supervision harmful in its present form.",
        "  Evidence-linked rationales remain an ablation; free rationale distillation is rejected.",
        "- Learned or self-generated rewards never override authored strings, boxes, table trees,",
        "  chart values, formulas, or executable financial programs.",
        "",
    ]

    for category in EXPECTED_CATEGORY_COUNTS:
        category_rows = [row for row in rows if row["category"] == category]
        lines.extend(
            [
                f"## {CATEGORY_TITLES[category]} ({len(category_rows)})",
                "",
                "| ID | Method | Source | Useful mechanism | Sub-1B criticism | Decision | Adjustable knobs |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in category_rows:
            source = f"[{row['organization']}, {row['year']}]({row['source']})"
            knobs = ", ".join(f"`{knob}`" for knob in row["knobs"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_table(row["id"]),
                        _escape_table(row["name"]),
                        source,
                        _escape_table(row["benefit"]),
                        _escape_table(row["limitation"]),
                        f"**{row['decision']}**",
                        knobs,
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## How to use the catalog",
            "",
            "Change one knob family at a time in `configs/sub1b_architecture.yaml`, retain the same",
            "train/held-out split and compute budget, and record the catalog IDs in the run config.",
            "Promote an `ablate` method to `adopt` only after it improves its target axis without",
            "violating grounding, multilingual, reliability, latency, or parameter gates. Record a",
            "negative result by changing the decision and limitation in the catalog, then regenerate",
            "this report; do not delete the method.",
            "",
        ]
    )
    return "\n".join(lines)

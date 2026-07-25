"""Compact readiness evidence for the pinned public UDD training component."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_UDD_COLUMNS = {
    "image",
    "sample_id",
    "source",
    "task",
    "instructions",
    "answers",
    "full_text",
    "table_html",
    "language",
    "metric",
    "hf_id",
    "split",
    "hf_config",
    "n_fields",
    "n_regions",
    "image_width",
    "image_height",
    "phash",
    "license",
    "fold",
    "elements_json",
}
REQUIRED_UDD_TASKS = {
    "recognition",
    "kie",
    "vqa",
    "localization",
    "table",
    "reasoning",
    "classification",
}
REQUIRED_UDD_LANGUAGES = {"en", "ko", "zh", "ja"}
REQUIRED_CAPABILITY_SOURCES = {
    "chart_understanding": {"chartqa", "dvqa", "plotqa", "tatqa"},
    "document_understanding": {"docvqa", "infovqa", "visualmrc", "docmatix"},
    "key_information_extraction": {"cord", "funsd", "sroie"},
    "multilingual_documents": {"mtvqa", "synthdog_ko"},
    "table_understanding": {"pubtabnet"},
}
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_COUNT = r"([0-9][0-9,]*)"


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()}"


def _text_fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _integer(value: str) -> int:
    return int(value.replace(",", ""))


def _front_matter(card: str) -> dict[str, Any]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", card, flags=re.DOTALL)
    if match is None:
        raise ValueError("dataset card is missing YAML front matter")
    payload = yaml.safe_load(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError("dataset card front matter must be a mapping")
    return payload


def _release_summary(card: str) -> dict[str, Any]:
    release = re.search(
        rf"Current release:\*\*\s*\*\*{_COUNT} image-rows / {_COUNT} QAs\*\*"
        rf".*?from \*\*{_COUNT} source.*?/\s*(?:>\s*)?\*\*{_COUNT} tasks\*\*",
        card,
        flags=re.DOTALL,
    )
    if release is None:
        raise ValueError("dataset card is missing the structured release summary")
    task_counts: dict[str, int] = {}
    for task in sorted(REQUIRED_UDD_TASKS):
        match = re.search(rf"\b{re.escape(task)}\s+{_COUNT}\b", card)
        if match is not None:
            task_counts[task] = _integer(match.group(1))
    return {
        "rows": _integer(release.group(1)),
        "qas": _integer(release.group(2)),
        "source_count": _integer(release.group(3)),
        "task_count": _integer(release.group(4)),
        "task_counts": task_counts,
    }


def _source_summary(card: str) -> tuple[int, list[str]]:
    section = re.search(
        r"### Sources \(([0-9]+)\)\s*\n(.*?)(?=\n### |\Z)",
        card,
        flags=re.DOTALL,
    )
    if section is None:
        raise ValueError("dataset card is missing the source inventory")
    inventory = section.group(2).split(".\n", 1)[0]
    sources = sorted(
        {
            value.strip()
            for value in re.findall(r"`([^`]+)`", inventory)
            if value.strip() and "/" not in value and " " not in value
        }
    )
    return int(section.group(1)), sources


def _language_summary(card: str) -> dict[str, int]:
    section = re.search(
        r"current distribution \(image-rows\):\s*(.*?)(?:\n|$)",
        card,
        flags=re.DOTALL,
    )
    if section is None:
        raise ValueError("dataset card is missing the language distribution")
    return {
        language: _integer(count)
        for language, count in re.findall(
            rf"\b([a-z]{{2,3}})\s+{_COUNT}\b",
            section.group(1),
        )
    }


def _card_schema(card: str) -> tuple[set[str], int, int]:
    metadata = _front_matter(card)
    info = metadata.get("dataset_info") or {}
    features = info.get("features") or []
    columns = {
        str(feature.get("name") or "")
        for feature in features
        if isinstance(feature, dict)
    }
    splits = info.get("splits") or []
    train = next(
        (
            split
            for split in splits
            if isinstance(split, dict) and split.get("name") == "train"
        ),
        {},
    )
    return columns, int(train.get("num_examples") or 0), int(
        train.get("num_bytes") or 0
    )


def _viewer_schema(viewer_info: Mapping[str, Any]) -> tuple[set[str], int]:
    default = (viewer_info.get("dataset_info") or {}).get("default") or {}
    features = default.get("features") or {}
    split = (default.get("splits") or {}).get("train") or {}
    return set(features), int(split.get("num_examples") or 0)


def _public_component(experiment: Mapping[str, Any]) -> dict[str, Any]:
    components = (experiment.get("data") or {}).get("components") or []
    matches = [
        item
        for item in components
        if isinstance(item, dict)
        and isinstance(item.get("hub"), dict)
        and str(item["hub"].get("repo_id") or "").endswith("/UDD")
    ]
    if len(matches) != 1:
        raise ValueError("experiment must contain exactly one Hub UDD component")
    return matches[0]


def build_public_dataset_readiness(
    experiment: Mapping[str, Any],
    *,
    repo_id: str,
    requested_revision: str,
    resolved_revision: str,
    main_revision: str,
    card: str,
    files: list[Mapping[str, Any]],
    viewer_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one immutable Hub snapshot to the executable training mixture."""

    component = _public_component(experiment)
    hub = component["hub"]
    release = _release_summary(card)
    card_columns, card_rows, card_bytes = _card_schema(card)
    viewer_columns, viewer_rows = _viewer_schema(viewer_info)
    declared_source_count, sources = _source_summary(card)
    languages = _language_summary(card)
    parquet = sorted(
        (
            {
                "path": str(item.get("path") or ""),
                "size": int(item.get("size") or 0),
                "sha256": str(item.get("sha256") or ""),
            }
            for item in files
            if str(item.get("path") or "").endswith(".parquet")
        ),
        key=lambda item: item["path"],
    )
    capability_sources = {
        capability: sorted(required & set(sources))
        for capability, required in REQUIRED_CAPABILITY_SOURCES.items()
    }

    checks = {
        "immutable_revision": (
            bool(_COMMIT_SHA.fullmatch(requested_revision))
            and requested_revision == resolved_revision == main_revision
        ),
        "training_component_binding": (
            str(component.get("name") or "") == "public_udd"
            and str(hub.get("repo_id") or "") == repo_id
            and str(hub.get("revision") or "") == requested_revision
            and str(hub.get("split") or "") == "train"
            and str(hub.get("fold") or "") == "train"
            and not (hub.get("sources") or [])
            and not (hub.get("tasks") or [])
            and not (hub.get("languages") or [])
            and hub.get("max_rows") is None
            and int(hub.get("decode_checks") or 0) >= 32
            and float(component.get("weight") or 0.0) == 0.55
        ),
        "schema": (
            REQUIRED_UDD_COLUMNS.issubset(card_columns)
            and REQUIRED_UDD_COLUMNS.issubset(viewer_columns)
        ),
        "scale": (
            release["rows"] == card_rows == viewer_rows
            and release["qas"] >= release["rows"]
            and release["source_count"] == declared_source_count == len(sources)
            and release["task_count"] == len(REQUIRED_UDD_TASKS)
            and sum(release["task_counts"].values()) == release["rows"]
        ),
        "multitask_coverage": (
            set(release["task_counts"]) == REQUIRED_UDD_TASKS
            and all(release["task_counts"][task] > 0 for task in REQUIRED_UDD_TASKS)
        ),
        "capability_source_coverage": all(
            set(observed) == required
            for observed, required in (
                (capability_sources[name], expected)
                for name, expected in REQUIRED_CAPABILITY_SOURCES.items()
            )
        ),
        "multilingual_coverage": (
            REQUIRED_UDD_LANGUAGES.issubset(languages)
            and all(languages[language] > 0 for language in REQUIRED_UDD_LANGUAGES)
        ),
        "shard_integrity": (
            len(parquet) == 5
            and all(
                item["path"]
                and item["size"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                for item in parquet
            )
            and sum(item["size"] for item in parquet)
            == int(
                ((viewer_info.get("dataset_info") or {}).get("default") or {}).get(
                    "download_size"
                )
                or 0
            )
        ),
    }
    payload = {
        "schema_version": 1,
        "claim_scope": "public_udd_training_input_readiness_only",
        "overall_status": "pass" if all(checks.values()) else "fail",
        "training_component_authorized": all(checks.values()),
        "quality_claim_authorized": False,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "main_revision_at_capture": main_revision,
        "component": {
            "name": component.get("name"),
            "weight": component.get("weight"),
            "split": hub.get("split"),
            "fold": hub.get("fold"),
            "sources": list(hub.get("sources") or []),
            "tasks": list(hub.get("tasks") or []),
            "languages": list(hub.get("languages") or []),
            "max_rows": hub.get("max_rows"),
            "decode_checks": hub.get("decode_checks"),
        },
        "dataset": {
            **release,
            "columns": sorted(card_columns),
            "sources": sources,
            "languages": dict(sorted(languages.items())),
            "capability_sources": capability_sources,
            "card_num_bytes": card_bytes,
        },
        "parquet_shards": parquet,
        "checks": checks,
        "source_fingerprints": {
            "dataset_card": _text_fingerprint(card),
            "viewer_info": _fingerprint(viewer_info),
        },
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def validate_public_dataset_readiness(
    payload: Mapping[str, Any],
    *,
    repo_id: str,
    revision: str,
) -> list[str]:
    """Revalidate compact evidence before it can satisfy goal readiness."""

    errors = []
    unsigned = dict(payload)
    fingerprint = str(unsigned.pop("fingerprint", ""))
    if fingerprint != _fingerprint(unsigned):
        errors.append("fingerprint mismatch")
    if payload.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if payload.get("claim_scope") != "public_udd_training_input_readiness_only":
        errors.append("invalid claim_scope")
    if payload.get("repo_id") != repo_id:
        errors.append("repo_id mismatch")
    if payload.get("requested_revision") != revision:
        errors.append("revision mismatch")
    if payload.get("resolved_revision") != revision:
        errors.append("resolved revision mismatch")
    if payload.get("overall_status") != "pass":
        errors.append("dataset readiness did not pass")
    if payload.get("training_component_authorized") is not True:
        errors.append("training component is not authorized")
    if payload.get("quality_claim_authorized") is not False:
        errors.append("dataset evidence cannot authorize quality")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        value is True for value in checks.values()
    ):
        errors.append("dataset readiness checks are incomplete")
    dataset = payload.get("dataset") or {}
    if set((dataset.get("task_counts") or {})) != REQUIRED_UDD_TASKS:
        errors.append("required task coverage is incomplete")
    if not REQUIRED_UDD_COLUMNS.issubset(dataset.get("columns") or []):
        errors.append("required schema columns are incomplete")
    if not REQUIRED_UDD_LANGUAGES.issubset(
        (dataset.get("languages") or {}).keys()
    ):
        errors.append("required language coverage is incomplete")
    return errors


def load_experiment(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a mapping")
    return payload

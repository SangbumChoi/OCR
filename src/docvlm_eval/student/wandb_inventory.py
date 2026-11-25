"""Compact W&B run inventory for external-activity observation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_wandb_run_inventory(
    runs: Iterable[Mapping[str, Any]],
    *,
    entity: str,
    project: str,
    observed_at: str,
    extraction_surface: str = "authenticated W&B API",
) -> dict[str, Any]:
    """Normalize run identity and state without copying configs or metrics."""

    required_text = {
        "entity": str(entity).strip(),
        "project": str(project).strip(),
        "observed_at": str(observed_at).strip(),
        "extraction_surface": str(extraction_surface).strip(),
    }
    missing_context = sorted(
        key for key, value in required_text.items() if not value
    )
    if missing_context:
        raise ValueError(
            f"W&B inventory context is empty: {missing_context}"
        )
    normalized: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        record = {
            "id": _optional_text(run.get("id")),
            "name": _optional_text(run.get("name")),
            "state": _optional_text(run.get("state")),
            "created_at": _optional_text(run.get("created_at")),
            "updated_at": _optional_text(run.get("updated_at")),
        }
        missing = [
            field
            for field in ("id", "name", "state")
            if record[field] is None
        ]
        if missing:
            raise ValueError(
                f"W&B run {index} is missing required fields: {missing}"
            )
        normalized.append(record)

    duplicate_ids = sorted(
        run_id
        for run_id, count in Counter(
            str(record["id"]) for record in normalized
        ).items()
        if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate W&B run IDs: {duplicate_ids}")
    normalized.sort(
        key=lambda record: (
            str(record.get("created_at") or ""),
            str(record["id"]),
        ),
        reverse=True,
    )
    states = dict(
        sorted(Counter(str(record["state"]) for record in normalized).items())
    )
    result = {
        "schema_version": 1,
        "claim_scope": "wandb_run_inventory_only",
        "observed_at": required_text["observed_at"],
        "source": {
            "provider": "Weights & Biases",
            "entity": required_text["entity"],
            "project": required_text["project"],
            "extraction_surface": required_text["extraction_surface"],
        },
        "run_count": len(normalized),
        "states": states,
        "runs": normalized,
        "limitations": [
            (
                "Run state and name are external activity signals, not sealed "
                "training or quality evidence."
            ),
            (
                "Configs, histories, summaries, artifacts, and metric tables "
                "are intentionally excluded."
            ),
        ],
    }
    result["fingerprint"] = _fingerprint(result)
    return result


def wandb_run_inventory_valid(value: Any) -> bool:
    """Verify the compact schema, counts, unique IDs, and fingerprint."""

    if not isinstance(value, dict):
        return False
    if (
        value.get("schema_version") != 1
        or value.get("claim_scope") != "wandb_run_inventory_only"
        or not isinstance(value.get("runs"), list)
        or not isinstance(value.get("observed_at"), str)
        or not value["observed_at"]
    ):
        return False
    source = value.get("source")
    if not isinstance(source, dict) or any(
        not isinstance(source.get(field), str) or not source[field]
        for field in ("provider", "entity", "project", "extraction_surface")
    ):
        return False
    body = dict(value)
    observed_fingerprint = body.pop("fingerprint", None)
    if observed_fingerprint != _fingerprint(body):
        return False
    runs = value["runs"]
    if value.get("run_count") != len(runs):
        return False
    ids: list[str] = []
    states: Counter[str] = Counter()
    for run in runs:
        if not isinstance(run, dict):
            return False
        if any(
            not isinstance(run.get(field), str) or not run[field]
            for field in ("id", "name", "state")
        ):
            return False
        ids.append(run["id"])
        states[run["state"]] += 1
    return (
        len(ids) == len(set(ids))
        and value.get("states") == dict(sorted(states.items()))
    )

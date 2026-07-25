"""Content-addressed cross-runtime handoff for sealed Smol pilot evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .pilot_execution import attestation_is_sealed
from .sweep import SweepPlan


PILOT = "docvlm-smol-vision-transfer-pilot"
FILES = ("sweep_run_summary.json", "comparison.json")


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _expected_runs(plan: SweepPlan) -> set[str]:
    if (
        plan.name != PILOT
        or plan.baseline != "lfm_language_only"
        or tuple(plan.replicates) != ("seed_0",)
    ):
        raise ValueError("handoff requires the exact Smol pilot topology")
    runs = {variant.id for variant in plan.variants}
    expected = {
        "lfm_language_only--seed_0",
        "lfm_smol_dual--seed_0",
    }
    if runs != expected:
        raise ValueError("handoff requires exactly two matched Smol pilot runs")
    return expected


def _attestation_hashes(
    records: Any,
    *,
    expected_runs: set[str],
    source: str,
) -> dict[str, str]:
    if not isinstance(records, dict) or set(records) != expected_runs:
        raise ValueError(f"{source} does not contain the exact pilot runs")
    hashes: dict[str, str] = {}
    for run_id, attestation in records.items():
        if not attestation_is_sealed(attestation):
            raise ValueError(f"{source} run {run_id!r} is not sealed")
        hashes[run_id] = str(attestation["attestation_sha256"])
    return dict(sorted(hashes.items()))


def _validate_sources(
    plan: SweepPlan,
    summary: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, str]:
    expected_runs = _expected_runs(plan)
    if summary.get("status") != "completed":
        raise ValueError("pilot sweep summary is not completed")
    variants = summary.get("variants")
    if not isinstance(variants, list):
        raise ValueError("pilot sweep summary variants must be a list")
    summary_records: dict[str, Any] = {}
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("pilot sweep summary variant must be a mapping")
        run_id = str(item.get("run") or "")
        if not run_id or run_id in summary_records:
            raise ValueError("pilot sweep summary has duplicate or empty run IDs")
        if item.get("status") != "completed":
            raise ValueError(f"pilot run {run_id!r} is not completed")
        summary_records[run_id] = item.get("execution_attestation")
    summary_hashes = _attestation_hashes(
        summary_records,
        expected_runs=expected_runs,
        source="pilot sweep summary",
    )

    if (
        comparison.get("schema_version") != 6
        or comparison.get("sweep") != PILOT
        or comparison.get("sweep_fingerprint") != plan.fingerprint
        or comparison.get("baseline") != plan.baseline
        or comparison.get("replicates") != ["seed_0"]
    ):
        raise ValueError("pilot comparison identity does not match the plan")
    comparison_hashes = _attestation_hashes(
        comparison.get("execution_attestations"),
        expected_runs=expected_runs,
        source="pilot comparison",
    )
    if summary_hashes != comparison_hashes:
        raise ValueError(
            "pilot summary and comparison attestations do not match"
        )
    return summary_hashes


def build_smol_pilot_handoff(
    plan: SweepPlan,
    *,
    sweep_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build one immutable handoff directory and return its manifest."""

    source_root = Path(sweep_root).resolve()
    sources = {name: source_root / name for name in FILES}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"pilot handoff inputs are missing: {missing}")
    summary = _read_json(sources["sweep_run_summary.json"])
    comparison = _read_json(sources["comparison.json"])
    attestation_hashes = _validate_sources(plan, summary, comparison)
    files = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for name, path in sources.items()
    }
    manifest = {
        "schema_version": 1,
        "claim_scope": "smol_pilot_cross_runtime_handoff",
        "pilot": PILOT,
        "sweep_fingerprint": plan.fingerprint,
        "expected_runs": sorted(_expected_runs(plan)),
        "execution_attestations": attestation_hashes,
        "files": files,
        "quality_claim_authorized": False,
        "promotion_claim_authorized": False,
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    digest = manifest["fingerprint"].split(":", 1)[1]
    destination = Path(output_root).resolve() / PILOT / digest
    manifest_path = destination / "handoff_manifest.json"
    if destination.exists():
        verification = verify_smol_pilot_handoff(destination, plan=plan)
        if verification["manifest"] != manifest:
            raise FileExistsError(
                "content-addressed handoff contains different evidence"
            )
        return {
            **manifest,
            "root": str(destination),
            "reused": True,
        }

    destination.mkdir(parents=True)
    try:
        for name, source in sources.items():
            shutil.copy2(source, destination / name)
        _atomic_write(manifest_path, manifest)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        **manifest,
        "root": str(destination),
        "reused": False,
    }


def verify_smol_pilot_handoff(
    root: str | Path,
    *,
    plan: SweepPlan,
) -> dict[str, Any]:
    """Verify manifest identity, source hashes, and sealed attestation linkage."""

    handoff_root = Path(root).resolve()
    manifest = _read_json(handoff_root / "handoff_manifest.json")
    observed_fingerprint = manifest.get("fingerprint")
    body = dict(manifest)
    body.pop("fingerprint", None)
    if observed_fingerprint != _fingerprint(body):
        raise ValueError("pilot handoff manifest fingerprint is invalid")
    expected_runs = _expected_runs(plan)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("claim_scope")
        != "smol_pilot_cross_runtime_handoff"
        or manifest.get("pilot") != PILOT
        or manifest.get("sweep_fingerprint") != plan.fingerprint
        or set(manifest.get("expected_runs") or []) != expected_runs
        or manifest.get("quality_claim_authorized") is not False
        or manifest.get("promotion_claim_authorized") is not False
    ):
        raise ValueError("pilot handoff manifest identity is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES):
        raise ValueError("pilot handoff manifest file set is invalid")
    for name in FILES:
        path = handoff_root / name
        record = files[name]
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or record.get("bytes") != path.stat().st_size
            or record.get("sha256") != _file_sha256(path)
        ):
            raise ValueError(f"pilot handoff file integrity failed: {name}")
    summary = _read_json(handoff_root / "sweep_run_summary.json")
    comparison = _read_json(handoff_root / "comparison.json")
    attestation_hashes = _validate_sources(plan, summary, comparison)
    if manifest.get("execution_attestations") != attestation_hashes:
        raise ValueError("pilot handoff attestation manifest is invalid")
    return {
        "manifest": manifest,
        "root": str(handoff_root),
        "summary": summary,
        "comparison": comparison,
    }


def restore_smol_pilot_handoff(
    root: str | Path,
    *,
    plan: SweepPlan,
    sweep_root: str | Path,
) -> dict[str, Any]:
    """Restore verified evidence without replacing different local files."""

    verification = verify_smol_pilot_handoff(root, plan=plan)
    destination = Path(sweep_root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    reused: list[str] = []
    for name in FILES:
        source = Path(verification["root"]) / name
        target = destination / name
        if target.exists():
            if _file_sha256(target) != _file_sha256(source):
                raise FileExistsError(
                    f"refusing to replace different local evidence: {target}"
                )
            reused.append(name)
            continue
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination,
            prefix=f".{name}.",
            delete=False,
        ) as handle:
            with source.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, handle)
            temporary = Path(handle.name)
        os.replace(temporary, target)
        restored.append(name)
    return {
        "fingerprint": verification["manifest"]["fingerprint"],
        "restored": restored,
        "reused": reused,
        "sweep_root": str(destination),
    }

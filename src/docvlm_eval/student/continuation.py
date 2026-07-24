"""Fail-closed provenance for failure-driven student curriculum rounds."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import StudentConfig, student_config_fingerprint
from .synthesis_policy import (
    file_fingerprint,
    payload_fingerprint,
    validate_generation_plan,
    validate_generation_plan_source,
)
from .tokenizer import DocumentTokenizer


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "fingerprint": file_fingerprint(path),
    }


def _tree_record(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    records = [_file_record(root / name) for name in names]
    return {
        "path": str(root),
        "files": records,
        "fingerprint": payload_fingerprint(records),
    }


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _final_stage(spec: Mapping[str, Any]) -> str:
    posttraining = spec.get("posttraining")
    if not isinstance(posttraining, Mapping):
        raise ValueError("parent experiment has no posttraining mapping")
    preference = posttraining.get("preference")
    rlvr = posttraining.get("rlvr")
    if isinstance(preference, Mapping) and bool(preference.get("enabled", False)):
        return "preference"
    if isinstance(rlvr, Mapping) and bool(rlvr.get("enabled", True)):
        return "rlvr"
    return "sft"


def _round_index(spec: Mapping[str, Any]) -> int:
    continuation = spec.get("continuation")
    if not isinstance(continuation, Mapping) or not bool(
        continuation.get("enabled", False)
    ):
        return 0
    return int(continuation.get("round_index", 0))


@dataclass(frozen=True)
class ContinuationContract:
    """Exact parent artifacts authorized for one curriculum round."""

    parent_root: str
    parent_experiment_fingerprint: str
    parent_round_index: int
    round_index: int
    optimizer_policy: str
    replay_fraction: float
    replay_seed: int
    final_stage: str
    checkpoint: str
    tokenizer: str
    replay_samples: str
    replay_source_kind: str
    replay_origin_rounds: tuple[int, ...]
    training_policy_plan: str
    student_config_fingerprint: str
    tokenizer_fingerprint: str
    checkpoint_fingerprint: str
    replay_samples_fingerprint: str
    training_policy_fingerprint: str
    training_policy_source_fingerprint: str
    parent_attestation_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["replay_origin_rounds"] = list(self.replay_origin_rounds)
        payload["schema_version"] = 2
        payload["continuation_fingerprint"] = payload_fingerprint(payload)
        return payload


def resolve_continuation_contract(
    spec: Mapping[str, Any],
    *,
    repo_root: str | Path,
    blueprint: Mapping[str, Any],
) -> ContinuationContract:
    """Validate a complete parent run and return its immutable handoff."""

    if not isinstance(spec, Mapping) or not bool(spec.get("enabled", False)):
        raise ValueError("continuation must be an enabled mapping")
    allowed = {
        "enabled",
        "parent_root",
        "round_index",
        "optimizer_policy",
        "replay_fraction",
        "replay_seed",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"continuation has unsupported fields: {sorted(unknown)}")
    root = Path(repo_root).resolve()
    parent_root = _resolve(root, str(spec.get("parent_root") or ""))
    if not parent_root.is_dir():
        raise ValueError(f"continuation parent_root does not exist: {parent_root}")
    optimizer_policy = str(spec.get("optimizer_policy") or "")
    if optimizer_policy != "reset_per_stage":
        raise ValueError(
            "continuation.optimizer_policy must be reset_per_stage"
        )
    replay_fraction = float(spec.get("replay_fraction", -1.0))
    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError(
            "continuation.replay_fraction must be within [0, 1)"
        )
    replay_seed = int(spec.get("replay_seed", -1))
    if replay_seed < 0:
        raise ValueError("continuation.replay_seed must be non-negative")
    round_index = int(spec.get("round_index", 0))
    if round_index <= 0:
        raise ValueError("continuation.round_index must be positive")

    plan_path = parent_root / "experiment_plan.json"
    parent_spec_path = parent_root / "experiment_spec.json"
    summary_path = parent_root / "run_summary.json"
    attestation_path = parent_root / "evidence_attestation.json"
    plan = _load_mapping(plan_path, "parent experiment plan")
    parent_spec = _load_mapping(parent_spec_path, "parent experiment spec")
    summary = _load_mapping(summary_path, "parent run summary")
    attestation = _load_mapping(
        attestation_path,
        "parent experiment attestation",
    )
    parent_fingerprint = str(plan.get("fingerprint") or "")
    if (
        Path(str(plan.get("root") or "")).resolve() != parent_root
        or not parent_fingerprint.startswith("sha256:")
    ):
        raise ValueError("parent experiment plan root or fingerprint is invalid")
    plan_stages = {
        str(row.get("name") or "")
        for row in plan.get("stages", [])
        if isinstance(row, dict)
    }
    summary_stages = {
        str(row.get("stage") or ""): row
        for row in summary.get("stages", [])
        if isinstance(row, dict)
    }
    if (
        int(summary.get("schema_version", 0)) < 2
        or summary.get("pipeline_complete") is not True
        or summary.get("fingerprint") != parent_fingerprint
        or not plan_stages
        or set(summary_stages) != plan_stages
        or any(
            row.get("state_status") != "completed"
            or row.get("signature_matches") is not True
            or row.get("artifacts_valid") is not True
            for row in summary_stages.values()
        )
    ):
        raise ValueError(
            "continuation requires a complete, signature-matched parent run"
        )
    attestation_hash = str(attestation.get("attestation_sha256") or "")
    unsigned_attestation = dict(attestation)
    unsigned_attestation.pop("attestation_sha256", None)
    if (
        int(attestation.get("schema_version", 0)) != 1
        or attestation.get("hash_mode") != "full"
        or attestation.get("contract_status") != "pass"
        or Path(str(attestation.get("experiment_root") or "")).resolve()
        != parent_root
        or attestation.get("experiment_fingerprint") != parent_fingerprint
        or attestation_hash != payload_fingerprint(unsigned_attestation)
    ):
        raise ValueError(
            "continuation requires a valid full-hash parent attestation"
        )
    parent_round_index = _round_index(parent_spec)
    if round_index != parent_round_index + 1:
        raise ValueError(
            "continuation.round_index must increment the parent round by one"
        )

    final_stage = _final_stage(parent_spec)
    required_parent_stages = {
        final_stage,
        "evaluate",
        "plan_next_synthetic_batch",
    }
    if not required_parent_stages.issubset(plan_stages):
        raise ValueError(
            "parent experiment plan lacks final training, evaluation, or "
            "next-batch planning stages"
        )
    pointer = parent_root / "artifacts" / final_stage / "latest_checkpoint.txt"
    if not pointer.is_file():
        raise ValueError(f"parent checkpoint pointer does not exist: {pointer}")
    checkpoint_root = Path(
        pointer.read_text(encoding="utf-8").strip()
    ).resolve()
    expected_stage_root = (parent_root / "artifacts" / final_stage).resolve()
    try:
        checkpoint_root.relative_to(expected_stage_root)
    except ValueError as exc:
        raise ValueError(
            "parent checkpoint pointer escapes its final-stage artifact root"
        ) from exc
    checkpoint = checkpoint_root / "student"
    checkpoint_files = (
        "student_config.json",
        "model.pt",
        "metadata.json",
    )
    if any(not (checkpoint / name).is_file() for name in checkpoint_files):
        raise ValueError("parent student checkpoint is incomplete")

    expected_student = StudentConfig.from_blueprint(dict(blueprint))
    actual_config = json.loads(
        (checkpoint / "student_config.json").read_text(encoding="utf-8")
    )
    if actual_config != expected_student.to_dict():
        raise ValueError(
            "parent student architecture does not match the current blueprint"
        )
    expected_student_fingerprint = student_config_fingerprint(
        expected_student
    )
    metadata = _load_mapping(
        checkpoint / "metadata.json",
        "parent checkpoint metadata",
    )
    expected_stage_prefix = (
        "preference:" if final_stage == "preference" else final_stage
    )
    if not str(metadata.get("run_stage") or "").startswith(
        expected_stage_prefix
    ):
        raise ValueError(
            "parent checkpoint run_stage does not match the parent final stage"
        )

    tokenizer = parent_root / "artifacts" / "tokenizer"
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json")
    if any(not (tokenizer / name).is_file() for name in tokenizer_files):
        raise ValueError("parent tokenizer artifact is incomplete")
    loaded_tokenizer = DocumentTokenizer.from_pretrained(tokenizer)
    tokenizer_fingerprint = loaded_tokenizer.fingerprint
    tokenizer_metadata = _load_mapping(
        tokenizer / "tokenizer_config.json",
        "parent tokenizer metadata",
    )
    if (
        tokenizer_metadata.get("fingerprint") != tokenizer_fingerprint
        or metadata.get("tokenizer_fingerprint") != tokenizer_fingerprint
        or loaded_tokenizer.vocab_size
        != expected_student.language.vocab_size
    ):
        raise ValueError(
            "parent tokenizer, checkpoint metadata, and student vocabulary "
            "do not share one token-ID contract"
        )

    if parent_round_index == 0:
        replay_samples = parent_root / "artifacts" / "samples" / "train.jsonl"
        replay_source_kind = "base_train"
        replay_origin_rounds = (0,)
    else:
        replay_samples = (
            parent_root
            / "artifacts"
            / "continuation"
            / "replay_memory.jsonl"
        )
        replay_source_kind = "cumulative_replay_memory"
    if not replay_samples.is_file():
        raise ValueError(
            f"parent replay samples do not exist: {replay_samples}"
        )
    if parent_round_index > 0:
        replay_rows = _load_jsonl(replay_samples, "parent replay memory")
        replay_origin_rounds = _validate_replay_memory(
            replay_rows,
            parent_round_index=parent_round_index,
        )
    policy_path = (
        parent_root / "artifacts" / "synthetic" / "next_train_plan.json"
    )
    policy = _load_mapping(policy_path, "parent synthesis policy plan")
    validate_generation_plan(policy, require_training_authorized=True)
    source_path = validate_generation_plan_source(policy).resolve()
    expected_source = (
        parent_root
        / "artifacts"
        / "evaluation"
        / "validation"
        / "per_sample.jsonl"
    ).resolve()
    if source_path != expected_source:
        raise ValueError(
            "parent synthesis policy must originate from the parent "
            "validation evaluation"
        )

    checkpoint_record = _tree_record(checkpoint, checkpoint_files)
    attested_files: dict[str, dict[str, Any]] = {}
    for stage in attestation.get("stages", []):
        if not isinstance(stage, dict):
            continue
        for record in stage.get("files", []):
            if isinstance(record, dict) and record.get("path"):
                attested_files[str(record["path"])] = record
    for control in attestation.get("control_files", []):
        if isinstance(control, dict) and control.get("path"):
            attested_files[str(control["path"])] = control

    def require_attested(path: Path) -> None:
        try:
            display = str(path.resolve().relative_to(parent_root))
        except ValueError:
            display = str(path.resolve())
        record = attested_files.get(display)
        if (
            record is None
            or record.get("sha256") != file_fingerprint(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise ValueError(
                f"parent handoff artifact is absent from or differs from "
                f"the full attestation: {display}"
            )

    for path in (
        plan_path,
        parent_spec_path,
        summary_path,
        replay_samples,
        policy_path,
        *(checkpoint / name for name in checkpoint_files),
    ):
        require_attested(path)
    return ContinuationContract(
        parent_root=str(parent_root),
        parent_experiment_fingerprint=parent_fingerprint,
        parent_round_index=parent_round_index,
        round_index=round_index,
        optimizer_policy=optimizer_policy,
        replay_fraction=replay_fraction,
        replay_seed=replay_seed,
        final_stage=final_stage,
        checkpoint=str(checkpoint),
        tokenizer=str(tokenizer),
        replay_samples=str(replay_samples),
        replay_source_kind=replay_source_kind,
        replay_origin_rounds=replay_origin_rounds,
        training_policy_plan=str(policy_path),
        student_config_fingerprint=expected_student_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        checkpoint_fingerprint=str(checkpoint_record["fingerprint"]),
        replay_samples_fingerprint=file_fingerprint(replay_samples),
        training_policy_fingerprint=str(policy["plan_fingerprint"]),
        training_policy_source_fingerprint=str(
            policy["source"]["fingerprint"]
        ),
        parent_attestation_sha256=attestation_hash,
    )


def write_continuation_manifest(
    contract: ContinuationContract,
    output: str | Path,
) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            contract.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def materialize_continuation_tokenizer(
    contract: ContinuationContract,
    output: str | Path,
) -> dict[str, Any]:
    """Copy the verified tokenizer contract into the child experiment root."""

    source = Path(contract.tokenizer)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    names = ("tokenizer.json", "tokenizer_config.json")
    for name in names:
        source_path = source / name
        destination_path = destination / name
        temporary = destination_path.with_name(f".{destination_path.name}.tmp")
        temporary.write_bytes(source_path.read_bytes())
        temporary.replace(destination_path)
    tokenizer = DocumentTokenizer.from_pretrained(destination)
    if tokenizer.fingerprint != contract.tokenizer_fingerprint:
        raise ValueError(
            "materialized tokenizer differs from the continuation contract"
        )
    return _tree_record(destination, names)


def prepare_next_round_spec(
    *,
    parent_root: str | Path,
    output_root: str | Path,
    round_index: int,
    replay_fraction: float,
    replay_seed: int,
) -> dict[str, Any]:
    """Patch a completed parent experiment into its next curriculum round."""

    parent = Path(parent_root).resolve()
    raw = _load_mapping(
        parent / "experiment_spec.json",
        "parent experiment spec",
    )
    spec = copy.deepcopy(raw)
    spec["name"] = f"{raw['name'].split('--round-')[0]}--round-{round_index:03d}"
    spec["output_root"] = str(Path(output_root).resolve())
    spec["continuation"] = {
        "enabled": True,
        "parent_root": str(parent),
        "round_index": int(round_index),
        "optimizer_policy": "reset_per_stage",
        "replay_fraction": float(replay_fraction),
        "replay_seed": int(replay_seed),
    }
    policy_path = (
        parent / "artifacts" / "synthetic" / "next_train_plan.json"
    )
    spec["synthetic"]["training_policy_plan"] = str(policy_path)
    evaluation = spec.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("parent experiment has no evaluation mapping")
    evaluation["baseline_checkpoint_stage"] = "inherited"
    evaluation["baseline_evaluation"] = None
    return spec


def prepare_initial_round_spec(
    *,
    experiment: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Pin a base experiment to curriculum round zero."""

    source = Path(experiment)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("base experiment must be a mapping")
    spec = copy.deepcopy(raw)
    spec["name"] = f"{str(raw['name']).split('--round-')[0]}--round-000"
    spec["output_root"] = str(Path(output_root).resolve())
    spec["continuation"] = {"enabled": False}
    adaptation = (spec.get("synthetic") or {}).get("adaptation_policy")
    if (
        not isinstance(adaptation, dict)
        or not bool(adaptation.get("enabled", False))
        or (spec.get("synthetic") or {}).get("validation_count") is None
    ):
        raise ValueError(
            "curriculum round zero requires validation-backed synthetic "
            "adaptation"
        )
    if (spec.get("synthetic") or {}).get("training_policy_plan"):
        raise ValueError(
            "curriculum round zero cannot start from a prior training policy"
        )
    return spec


def write_round_spec(spec: Mapping[str, Any], output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(spec), sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not str(
                row.get("sample_id") or ""
            ):
                raise ValueError(
                    f"{label} row {line_number} must be a sample object"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} contains no samples")
    return rows


def _validate_unique_sample_ids(
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{label} sample IDs must be unique")


def _validate_replay_memory(
    rows: list[dict[str, Any]],
    *,
    parent_round_index: int,
) -> tuple[int, ...]:
    origins: set[int] = set()
    for row in rows:
        meta = row.get("meta")
        if not isinstance(meta, dict):
            raise ValueError("replay memory sample meta must be a mapping")
        origin = meta.get("curriculum_origin_round_index")
        if not isinstance(origin, int) or not 0 <= origin <= parent_round_index:
            raise ValueError(
                "replay memory origin must be an integer within completed "
                "curriculum rounds"
            )
        source_id = str(meta.get("curriculum_source_sample_id") or "")
        expected_id = f"memory-r{origin:03d}:{source_id}"
        if not source_id or row["sample_id"] != expected_id:
            raise ValueError(
                "replay memory sample ID does not match its origin lineage"
            )
        origins.add(origin)
    expected_origins = set(range(parent_round_index + 1))
    if origins != expected_origins:
        raise ValueError(
            "replay memory must contain every completed curriculum round"
        )
    _validate_unique_sample_ids(rows, "replay memory")
    return tuple(sorted(origins))


def _memory_row(
    row: Mapping[str, Any],
    *,
    origin_round_index: int,
) -> dict[str, Any]:
    item = copy.deepcopy(dict(row))
    meta = item.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("curriculum sample meta must be a mapping")
    source_id = str(item["sample_id"])
    item["sample_id"] = f"memory-r{origin_round_index:03d}:{source_id}"
    meta.update(
        {
            "curriculum_role": "replay_memory",
            "curriculum_origin_round_index": origin_round_index,
            "curriculum_source_sample_id": source_id,
        }
    )
    return item


def _stratified_replay(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    by_origin: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        origin = int(row["meta"]["curriculum_origin_round_index"])
        by_origin.setdefault(origin, []).append(row)
    rng = random.Random(seed)
    origin_order = sorted(by_origin)
    rng.shuffle(origin_order)
    for origin in origin_order:
        rng.shuffle(by_origin[origin])
    selected: list[dict[str, Any]] = []
    offsets = {origin: 0 for origin in origin_order}
    while len(selected) < count:
        progressed = False
        for origin in origin_order:
            offset = offsets[origin]
            group = by_origin[origin]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            offsets[origin] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def build_curriculum_samples(
    *,
    current_samples: str | Path,
    replay_samples: str | Path,
    replay_fraction: float,
    replay_seed: int,
    parent_round_index: int,
    output: str | Path,
    memory_output: str | Path,
    manifest_output: str | Path,
) -> dict[str, Any]:
    """Build bounded active training data and unbounded cumulative replay memory."""

    if not 0.0 <= replay_fraction < 1.0:
        raise ValueError("replay_fraction must be within [0, 1)")
    if replay_seed < 0 or parent_round_index < 0:
        raise ValueError("replay_seed and parent_round_index must be non-negative")
    current_path = Path(current_samples).resolve()
    replay_path = Path(replay_samples).resolve()
    current = _load_jsonl(current_path, "current samples")
    replay = _load_jsonl(replay_path, "replay samples")
    _validate_unique_sample_ids(current, "current samples")
    if parent_round_index == 0:
        replay_memory = [
            _memory_row(row, origin_round_index=0) for row in replay
        ]
        replay_origins = (0,)
    else:
        replay_origins = _validate_replay_memory(
            replay,
            parent_round_index=parent_round_index,
        )
        replay_memory = copy.deepcopy(replay)
    current_round_index = parent_round_index + 1
    current_memory = [
        _memory_row(row, origin_round_index=current_round_index)
        for row in current
    ]
    cumulative_memory = [*replay_memory, *current_memory]
    _validate_unique_sample_ids(cumulative_memory, "cumulative replay memory")
    desired_replay = (
        round(len(current) * replay_fraction / (1.0 - replay_fraction))
        if replay_fraction > 0
        else 0
    )
    replay_count = min(len(replay_memory), desired_replay)
    rng = random.Random(replay_seed)
    selected = _stratified_replay(
        replay_memory,
        count=replay_count,
        seed=replay_seed,
    )
    rows: list[dict[str, Any]] = []
    for row in current:
        item = copy.deepcopy(row)
        meta = item.setdefault("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("current sample meta must be a mapping")
        meta.update(
            {
                "curriculum_role": "new_failure_batch",
                "curriculum_origin_round_index": current_round_index,
                "curriculum_source_sample_id": str(item["sample_id"]),
            }
        )
        rows.append(item)
    for ordinal, row in enumerate(selected):
        item = copy.deepcopy(row)
        memory_id = str(item["sample_id"])
        item["sample_id"] = (
            f"replay-r{parent_round_index:03d}-{ordinal:06d}:{memory_id}"
        )
        meta = item.setdefault("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("replay sample meta must be a mapping")
        meta.update(
            {
                "curriculum_role": "parent_replay",
                "curriculum_replayed_from_round_index": parent_round_index,
                "curriculum_memory_sample_id": memory_id,
            }
        )
        rows.append(item)
    rng.shuffle(rows)
    _validate_unique_sample_ids(rows, "curriculum samples")

    destination = Path(output)
    memory_destination = Path(memory_output)
    _write_jsonl(destination, rows)
    _write_jsonl(memory_destination, cumulative_memory)
    selected_origin_counts: dict[str, int] = {}
    for row in selected:
        origin = str(row["meta"]["curriculum_origin_round_index"])
        selected_origin_counts[origin] = selected_origin_counts.get(origin, 0) + 1
    memory_origin_counts: dict[str, int] = {}
    for row in cumulative_memory:
        origin = str(row["meta"]["curriculum_origin_round_index"])
        memory_origin_counts[origin] = memory_origin_counts.get(origin, 0) + 1
    manifest = {
        "schema_version": 2,
        "policy": (
            "all_new_plus_stratified_cumulative_replay_without_replacement"
        ),
        "current_samples": _file_record(current_path),
        "replay_samples": _file_record(replay_path),
        "replay_origin_rounds": list(replay_origins),
        "requested_replay_fraction": replay_fraction,
        "replay_seed": replay_seed,
        "parent_round_index": parent_round_index,
        "current_round_index": current_round_index,
        "new_sample_count": len(current),
        "available_replay_count": len(replay_memory),
        "desired_replay_count": desired_replay,
        "selected_replay_count": replay_count,
        "selected_replay_origin_counts": selected_origin_counts,
        "output_sample_count": len(rows),
        "realized_replay_fraction": (
            replay_count / len(rows) if rows else 0.0
        ),
        "output": _file_record(destination),
        "memory_sample_count": len(cumulative_memory),
        "memory_origin_counts": memory_origin_counts,
        "memory_output": _file_record(memory_destination),
    }
    manifest["manifest_fingerprint"] = payload_fingerprint(manifest)
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temporary = manifest_path.with_name(
        f".{manifest_path.name}.tmp"
    )
    manifest_temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest_temporary.replace(manifest_path)
    return manifest

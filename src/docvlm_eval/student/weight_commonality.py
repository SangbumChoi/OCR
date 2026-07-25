"""Bounded real-weight sketches for cross-architecture transfer decisions."""

from __future__ import annotations

import hashlib
import json
import math
import time
import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_DTYPE_BYTES = {
    "BOOL": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}
_NUMPY_DTYPES = {
    "BOOL": np.dtype("?"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
}
_UNSTABLE_RATIO = 1e30


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _bounded_retry(call: Callable[[], Any], *, attempts: int = 4) -> Any:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * 2**attempt)
    raise RuntimeError(
        f"remote metadata request failed after bounded retries: {error}"
    )


def semantic_weight_role(
    name: str,
    shape: Sequence[int],
) -> str | None:
    """Map heterogeneous checkpoint names to a conservative semantic role."""

    key = name.lower()
    if not key.endswith((".weight", ".kernel")):
        return None
    connector = any(
        marker in key
        for marker in (
            "connector",
            "projector",
            "multi_modal_project",
            "modality_projection",
            "vision_proj",
        )
    )
    vision = any(
        marker in key
        for marker in (
            "vision",
            "visual",
            "image_encoder",
            "image_tower",
            "davit",
        )
    )
    component = "connector" if connector else "vision" if vision else "language"
    if connector:
        return "connector.projection" if len(shape) >= 2 else None
    if any(marker in key for marker in ("embed_tokens", "token_embedding", "wte")):
        return "language.token_embedding"
    if vision and any(
        marker in key
        for marker in ("patch_embed", "patch_embedding", "patch_projection")
    ):
        return "vision.patch_embedding"
    if len(shape) == 1 and any(
        marker in key
        for marker in ("norm", "layer_normalization", "layernorm")
    ):
        return f"{component}.norm"
    attention_markers = {
        "q_proj": "attention.q",
        "query": "attention.q",
        "k_proj": "attention.k",
        "key": "attention.k",
        "v_proj": "attention.v",
        "value": "attention.v",
        "o_proj": "attention.o",
        "out_proj": "attention.o",
    }
    if any(marker in key for marker in ("attn", "attention")):
        for marker, suffix in attention_markers.items():
            if marker in key:
                return f"{component}.{suffix}"
    if component == "language" and any(
        marker in key
        for marker in ("short_conv", "conv.in_proj", "conv.out_proj")
    ):
        return "language.short_convolution"
    mlp_markers = {
        "gate_proj": "mlp.gate",
        ".w1.": "mlp.gate",
        "up_proj": "mlp.up",
        ".w3.": "mlp.up",
        "down_proj": "mlp.down",
        ".w2.": "mlp.down",
        ".fc1.": "mlp.in",
        ".fc2.": "mlp.out",
    }
    if any(marker in key for marker in ("mlp", "feed_forward", ".ffn.", ".fc")):
        for marker, suffix in mlp_markers.items():
            if marker in key:
                return f"{component}.{suffix}"
    return None


def _evenly_spaced(items: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[len(items) // 2]]
    indices = {
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [items[index] for index in sorted(indices)]


def _decode_values(payload: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        raw = np.frombuffer(payload, dtype="<u2")
        return (raw.astype(np.uint32) << 16).view(np.float32)
    numpy_dtype = _NUMPY_DTYPES.get(dtype)
    if numpy_dtype is None:
        raise ValueError(f"unsupported sketch dtype: {dtype}")
    return np.frombuffer(payload, dtype=numpy_dtype).astype(np.float64)


def _window_plan(
    parameter_count: int,
    *,
    max_values: int,
    windows: int = 3,
) -> list[tuple[int, int]]:
    if parameter_count <= 0 or max_values <= 0:
        return []
    value_count = min(parameter_count, max_values)
    window_count = min(windows, value_count)
    base = value_count // window_count
    remainder = value_count % window_count
    lengths = [
        base + (1 if index < remainder else 0)
        for index in range(window_count)
    ]
    starts = []
    for index, length in enumerate(lengths):
        if window_count == 1:
            start = max(0, (parameter_count - length) // 2)
        else:
            start = round(
                index
                * (parameter_count - length)
                / (window_count - 1)
            )
        starts.append((start, length))
    return starts


def _tensor_statistics(
    values: np.ndarray,
    shape: Sequence[int],
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return {
            "sampled_values": int(values.size),
            "finite_fraction": 0.0,
            "rms": 0.0,
            "fan_in_scaled_rms": 0.0,
            "zero_fraction": 1.0,
            "positive_fraction": 0.0,
            "outlier_ratio": _UNSTABLE_RATIO,
        }
    absolute = np.abs(finite_values)
    rms = float(np.sqrt(np.mean(np.square(finite_values))))
    fan_in = int(math.prod(shape[1:])) if len(shape) >= 2 else 1
    return {
        "sampled_values": int(values.size),
        "finite_fraction": float(np.mean(finite)),
        "rms": rms,
        "fan_in_scaled_rms": rms * math.sqrt(max(1, fan_in)),
        "zero_fraction": float(np.mean(finite_values == 0)),
        "positive_fraction": float(np.mean(finite_values > 0)),
        "outlier_ratio": (
            float(np.quantile(absolute, 0.99) / rms)
            if rms > 0
            else _UNSTABLE_RATIO
        ),
    }


def _aggregate_role(
    tensors: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "tensor_count": len(tensors),
        "sampled_values": sum(
            int(item["sampled_values"]) for item in tensors
        ),
        "finite_fraction": min(
            float(item["finite_fraction"]) for item in tensors
        ),
        "median_rms": median(float(item["rms"]) for item in tensors),
        "min_rms": min(float(item["rms"]) for item in tensors),
        "median_fan_in_scaled_rms": median(
            float(item["fan_in_scaled_rms"]) for item in tensors
        ),
        "median_zero_fraction": median(
            float(item["zero_fraction"]) for item in tensors
        ),
        "max_zero_fraction": max(
            float(item["zero_fraction"]) for item in tensors
        ),
        "median_positive_fraction": median(
            float(item["positive_fraction"]) for item in tensors
        ),
        "max_outlier_ratio": max(
            float(item["outlier_ratio"]) for item in tensors
        ),
    }
    summary["sample_healthy"] = (
        summary["finite_fraction"] == 1.0
        and summary["min_rms"] > 0
        and summary["max_zero_fraction"] < 0.5
        and summary["max_outlier_ratio"] < 50
    )
    return summary


def _summarize_samples(
    records: Sequence[dict[str, Any]],
    *,
    model_id: str,
    revision: str,
    evidence_mode: str,
    bytes_read: int,
    range_requests: int,
    sample_digest: str,
) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selection = []
    for record in records:
        by_role[record["role"]].append(record["statistics"])
        selection.append(
            {
                "file": record["file"],
                "tensor": record["tensor"],
                "dtype": record["dtype"],
                "shape": record["shape"],
            }
        )
    roles = {
        role: _aggregate_role(tensors)
        for role, tensors in sorted(by_role.items())
    }
    summary = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "evidence_mode": evidence_mode,
        "sampled_roles": len(roles),
        "sampled_tensors": len(records),
        "sampled_values": sum(
            item["sampled_values"] for item in roles.values()
        ),
        "bytes_read": int(bytes_read),
        "range_requests": int(range_requests),
        "selection_fingerprint": _fingerprint(selection),
        "sample_digest": sample_digest,
        "roles": roles,
    }
    summary["profile_fingerprint"] = _fingerprint(summary)
    return summary


def sketch_state_dict(
    state: Mapping[str, Any],
    *,
    model_id: str,
    revision: str = "in-memory",
    max_tensors_per_role: int = 3,
    max_values_per_tensor: int = 2048,
) -> dict[str, Any]:
    """Sketch an in-memory state dict without serializing sampled values."""

    import torch

    candidates: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            continue
        role = semantic_weight_role(name, tensor.shape)
        if role is not None:
            candidates[role].append((name, tensor))
    records = []
    digest = hashlib.sha256()
    bytes_read = 0
    for role, items in sorted(candidates.items()):
        for name, tensor in _evenly_spaced(items, max_tensors_per_role):
            flat = tensor.detach().reshape(-1)
            indices = [
                index
                for start, length in _window_plan(
                    flat.numel(),
                    max_values=max_values_per_tensor,
                )
                for index in range(start, start + length)
            ]
            sample = (
                flat[torch.tensor(indices, device=flat.device)]
                .float()
                .cpu()
                .numpy()
            )
            raw = sample.astype("<f4", copy=False).tobytes()
            digest.update(name.encode("utf-8"))
            digest.update(raw)
            bytes_read += len(raw)
            records.append(
                {
                    "file": "in-memory",
                    "tensor": name,
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                    "role": role,
                    "statistics": _tensor_statistics(sample, tensor.shape),
                }
            )
    return _summarize_samples(
        records,
        model_id=model_id,
        revision=revision,
        evidence_mode="bounded_in_memory_sample",
        bytes_read=bytes_read,
        range_requests=0,
        sample_digest=f"sha256:{digest.hexdigest()}",
    )


def sketch_remote_safetensors(
    *,
    model_id: str,
    revision: str,
    max_tensors_per_role: int = 3,
    max_values_per_tensor: int = 2048,
    max_workers: int = 4,
    range_get: Callable[[str, int, int], bytes] | None = None,
) -> dict[str, Any]:
    """Range-read a bounded deterministic sketch from a pinned Hub checkpoint."""

    if not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be within [1, 32]")
    from huggingface_hub import (
        HfApi,
        get_safetensors_metadata,
        hf_hub_url,
    )

    resolved = str(
        _bounded_retry(
            lambda: HfApi().model_info(
                model_id,
                revision=revision,
            )
        ).sha
    )
    if resolved != revision:
        raise ValueError(
            f"resolved revision {resolved} does not match pinned {revision}"
        )
    metadata = _bounded_retry(
        lambda: get_safetensors_metadata(
            model_id,
            revision=revision,
        )
    )
    if range_get is None:
        import requests

        def range_get(url: str, start: int, end: int) -> bytes:
            error: Exception | None = None
            for attempt in range(4):
                try:
                    response = requests.get(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=60,
                    )
                    if response.status_code != 206:
                        raise RuntimeError(
                            "range request returned HTTP "
                            f"{response.status_code}"
                        )
                    expected = end - start + 1
                    if len(response.content) != expected:
                        raise RuntimeError(
                            f"range request returned {len(response.content)} "
                            f"bytes; expected {expected}"
                        )
                    return response.content
                except (requests.RequestException, RuntimeError) as exc:
                    error = exc
                    if attempt < 3:
                        time.sleep(0.25 * 2**attempt)
            raise RuntimeError(
                f"range request failed after bounded retries: {error}"
            )

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    file_urls = {}
    for filename, file_metadata in metadata.files_metadata.items():
        file_urls[filename] = hf_hub_url(
            model_id,
            filename,
            revision=revision,
        )
        for name, info in file_metadata.tensors.items():
            if info.dtype not in _DTYPE_BYTES:
                continue
            role = semantic_weight_role(name, info.shape)
            if role is None:
                continue
            candidates[role].append(
                {
                    "file": filename,
                    "tensor": name,
                    "dtype": info.dtype,
                    "shape": list(info.shape),
                    "data_offsets": tuple(info.data_offsets),
                    "parameter_count": int(info.parameter_count),
                }
            )
    selected = [
        item
        for role in sorted(candidates)
        for item in _evenly_spaced(
            sorted(
                candidates[role],
                key=lambda candidate: (
                    candidate["file"],
                    candidate["tensor"],
                ),
            ),
            max_tensors_per_role,
        )
    ]
    header_lengths = {}
    bytes_read = 0
    requests_made = 0
    for filename in sorted({item["file"] for item in selected}):
        url = file_urls[filename]
        raw_header_length = range_get(url, 0, 7)
        requests_made += 1
        bytes_read += len(raw_header_length)
        header_lengths[filename] = int.from_bytes(
            raw_header_length,
            "little",
        )

    jobs = []
    for item_index, item in enumerate(selected):
        filename = item["file"]
        url = file_urls[filename]
        data_base = 8 + header_lengths[filename]
        element_bytes = _DTYPE_BYTES[item["dtype"]]
        for chunk_index, (value_start, value_count) in enumerate(_window_plan(
            item["parameter_count"],
            max_values=max_values_per_tensor,
        )):
            start = (
                data_base
                + int(item["data_offsets"][0])
                + value_start * element_bytes
            )
            end = start + value_count * element_bytes - 1
            jobs.append(
                (item_index, chunk_index, url, start, end)
            )

    def fetch(job):
        item_index, chunk_index, url, start, end = job
        return (
            item_index,
            chunk_index,
            start,
            range_get(url, start, end),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        fetched = list(executor.map(fetch, jobs))
    chunks_by_tensor: dict[int, list[tuple[int, int, bytes]]] = defaultdict(
        list
    )
    digest = hashlib.sha256()
    for item_index, chunk_index, start, payload in fetched:
        chunks_by_tensor[item_index].append(
            (chunk_index, start, payload)
        )
        item = selected[item_index]
        digest.update(item["file"].encode("utf-8"))
        digest.update(item["tensor"].encode("utf-8"))
        digest.update(start.to_bytes(8, "little"))
        digest.update(payload)
        bytes_read += len(payload)
        requests_made += 1

    records = []
    for item_index, item in enumerate(selected):
        chunks = [
            _decode_values(payload, item["dtype"])
            for _, _, payload in sorted(chunks_by_tensor[item_index])
        ]
        values = (
            np.concatenate(chunks)
            if chunks
            else np.empty(0, dtype=np.float64)
        )
        records.append(
            {
                **item,
                "role": semantic_weight_role(
                    item["tensor"],
                    item["shape"],
                ),
                "statistics": _tensor_statistics(values, item["shape"]),
            }
        )
    return _summarize_samples(
        records,
        model_id=model_id,
        revision=revision,
        evidence_mode="pinned_remote_safetensors_ranges",
        bytes_read=bytes_read,
        range_requests=requests_made,
        sample_digest=f"sha256:{digest.hexdigest()}",
    )


def _transfer_rule(role: str, stable: bool) -> str:
    if not stable:
        return "pairwise_preflight_no_population_prior"
    if role == "language.token_embedding":
        return "identity_mapped_token_rows"
    if ".mlp." in role:
        return "exact_or_joint_structured_channel_selection"
    if role == "connector.projection":
        return "exact_only_with_identical_connector_topology"
    if role == "language.short_convolution":
        return "exact_only_with_identical_operator_contract"
    return "exact_only_with_semantic_and_geometry_match"


def cross_architecture_weight_commonality(
    models: Sequence[Mapping[str, Any]],
    *,
    prevalence_threshold: float = 0.6,
    max_scaled_rms_ratio: float = 4.0,
) -> dict[str, Any]:
    """Find recurrent trained-weight characteristics across model families."""

    if not models:
        raise ValueError("weight commonality requires at least one model")
    if not 0 < prevalence_threshold <= 1:
        raise ValueError("prevalence_threshold must be within (0, 1]")
    if max_scaled_rms_ratio < 1:
        raise ValueError("max_scaled_rms_ratio must be at least 1")
    minimum_count = math.ceil(prevalence_threshold * len(models))
    roles = sorted(
        {
            role
            for model in models
            for role in model.get("roles", {})
        }
    )
    common_roles = []
    for role in roles:
        observations = [
            model["roles"][role]
            for model in models
            if role in model.get("roles", {})
        ]
        if len(observations) < minimum_count:
            continue
        scales = [
            float(item["median_fan_in_scaled_rms"])
            for item in observations
        ]
        positive_scales = [value for value in scales if value > 0]
        scale_ratio = (
            max(positive_scales) / min(positive_scales)
            if len(positive_scales) == len(scales)
            else _UNSTABLE_RATIO
        )
        stable = (
            all(float(item["finite_fraction"]) == 1.0 for item in observations)
            and scale_ratio <= max_scaled_rms_ratio
            and all(
                float(item["median_zero_fraction"]) < 0.5
                for item in observations
            )
            and all(
                float(item["max_outlier_ratio"]) < 50
                for item in observations
            )
        )
        common_roles.append(
            {
                "role": role,
                "model_count": len(observations),
                "prevalence": len(observations) / len(models),
                "median_fan_in_scaled_rms": median(scales),
                "scaled_rms_ratio": scale_ratio,
                "median_zero_fraction": median(
                    float(item["median_zero_fraction"])
                    for item in observations
                ),
                "median_positive_fraction": median(
                    float(item["median_positive_fraction"])
                    for item in observations
                ),
                "stable_across_models": stable,
                "transfer_rule": _transfer_rule(role, stable),
            }
        )
    result = {
        "schema_version": 1,
        "source_count": len(models),
        "prevalence_threshold": prevalence_threshold,
        "max_scaled_rms_ratio": max_scaled_rms_ratio,
        "source_fingerprints": {
            str(model["model_id"]): str(model["profile_fingerprint"])
            for model in models
        },
        "common_roles": common_roles,
        "stable_role_count": sum(
            item["stable_across_models"] for item in common_roles
        ),
        "decision_contract": {
            "raw_basis_alignment": "never_assumed",
            "exact_copy": (
                "requires stable sampled weights plus semantic, shape, "
                "normalization, attention, and position compatibility"
            ),
            "structured_mlp": (
                "requires stable SwiGLU roles and one joint source-channel "
                "selection across gate, up, and down projections"
            ),
            "fallback": "feature_or_relation_distillation",
            "population_instability": (
                "does not veto an otherwise healthy pairwise exact transfer"
            ),
        },
    }
    result["report_fingerprint"] = _fingerprint(result)
    return result


def build_weight_commonality_report(
    profiles: Sequence[Mapping[str, Any]],
    *,
    sketcher: Callable[..., dict[str, Any]] = sketch_remote_safetensors,
    max_tensors_per_role: int = 3,
    max_values_per_tensor: int = 2048,
    max_workers: int = 4,
) -> dict[str, Any]:
    model_sketches = [
        sketcher(
            model_id=str(profile["model_id"]),
            revision=str(profile["revision"]),
            max_tensors_per_role=max_tensors_per_role,
            max_values_per_tensor=max_values_per_tensor,
            max_workers=max_workers,
        )
        for profile in profiles
    ]
    commonality = cross_architecture_weight_commonality(model_sketches)
    report = {
        "schema_version": 1,
        "sampling_contract": {
            "max_tensors_per_role": max_tensors_per_role,
            "max_values_per_tensor": max_values_per_tensor,
            "window_count": 3,
            "max_workers": max_workers,
            "raw_values_persisted": False,
        },
        "models": model_sketches,
        "commonality": commonality,
    }
    report["report_fingerprint"] = _fingerprint(report)
    return report


def refresh_weight_commonality_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute derived health, commonality, and fingerprints from aggregates."""

    refreshed = copy.deepcopy(dict(report))
    models = refreshed.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("weight commonality report has no model aggregates")
    for model in models:
        roles = model.get("roles")
        if not isinstance(roles, dict):
            raise ValueError("weight commonality model roles are invalid")
        for summary in roles.values():
            summary["sample_healthy"] = (
                float(summary["finite_fraction"]) == 1.0
                and float(
                    summary.get("min_rms", summary["median_rms"])
                )
                > 0
                and float(
                    summary.get(
                        "max_zero_fraction",
                        summary["median_zero_fraction"],
                    )
                )
                < 0.5
                and float(summary["max_outlier_ratio"]) < 50
            )
        model.pop("profile_fingerprint", None)
        model["profile_fingerprint"] = _fingerprint(model)
    refreshed["commonality"] = cross_architecture_weight_commonality(models)
    refreshed.pop("report_fingerprint", None)
    refreshed["report_fingerprint"] = _fingerprint(refreshed)
    return refreshed


def validate_weight_commonality_report(
    report: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    *,
    require_remote: bool = True,
) -> dict[str, Any]:
    """Validate source identity, boundedness, and internal report fingerprints."""

    errors = []
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        raise ValueError("weight commonality report must use schema_version 1")
    expected = {
        str(profile["model_id"]): str(profile["revision"])
        for profile in profiles
    }
    models = report.get("models")
    if not isinstance(models, list):
        raise ValueError("weight commonality models must be a list")
    observed = {
        str(model.get("model_id")): str(model.get("revision"))
        for model in models
        if isinstance(model, Mapping)
    }
    if observed != expected:
        errors.append("weight commonality source identities do not match catalog")
    contract = report.get("sampling_contract")
    if not isinstance(contract, Mapping):
        errors.append("weight commonality sampling contract is missing")
        contract = {}
    if contract.get("raw_values_persisted") is not False:
        errors.append("weight commonality report must not persist raw values")
    max_tensors = int(contract.get("max_tensors_per_role", 0))
    max_values = int(contract.get("max_values_per_tensor", 0))
    max_workers = int(contract.get("max_workers", 0))
    if not 0 < max_tensors <= 8 or not 0 < max_values <= 8192:
        errors.append("weight commonality sampling bounds are invalid")
    if not 1 <= max_workers <= 32:
        errors.append("weight commonality worker bound is invalid")
    for model in models:
        if not isinstance(model, Mapping):
            errors.append("weight commonality model record is invalid")
            continue
        fingerprint = model.get("profile_fingerprint")
        body = dict(model)
        body.pop("profile_fingerprint", None)
        if fingerprint != _fingerprint(body):
            errors.append(
                f"{model.get('model_id')}: profile fingerprint mismatch"
            )
        for field in ("selection_fingerprint", "sample_digest"):
            value = model.get(field)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                errors.append(f"{model.get('model_id')}: {field} is invalid")
        if require_remote and model.get("evidence_mode") != (
            "pinned_remote_safetensors_ranges"
        ):
            errors.append(
                f"{model.get('model_id')}: evidence mode is not pinned remote"
            )
        if int(model.get("sampled_tensors", 0)) > (
            int(model.get("sampled_roles", 0)) * max_tensors
        ):
            errors.append(
                f"{model.get('model_id')}: tensor sampling bound exceeded"
            )
        if int(model.get("sampled_values", 0)) > (
            int(model.get("sampled_tensors", 0)) * max_values
        ):
            errors.append(
                f"{model.get('model_id')}: value sampling bound exceeded"
            )
        sampled_values = int(model.get("sampled_values", 0))
        bytes_read = int(model.get("bytes_read", 0))
        sampled_tensors = int(model.get("sampled_tensors", 0))
        if not 0 < bytes_read <= (
            sampled_values * 8 + sampled_tensors * 8
        ):
            errors.append(
                f"{model.get('model_id')}: byte sampling bound exceeded"
            )
        roles = model.get("roles")
        if not isinstance(roles, Mapping) or not roles:
            errors.append(f"{model.get('model_id')}: role evidence is missing")
        elif any(
            not isinstance(summary, Mapping)
            or not isinstance(summary.get("sample_healthy"), bool)
            for summary in roles.values()
        ):
            errors.append(
                f"{model.get('model_id')}: role health evidence is invalid"
            )
    commonality = report.get("commonality")
    if not isinstance(commonality, Mapping):
        errors.append("weight commonality aggregate is missing")
    else:
        common_body = dict(commonality)
        common_fingerprint = common_body.pop("report_fingerprint", None)
        if common_fingerprint != _fingerprint(common_body):
            errors.append("weight commonality aggregate fingerprint mismatch")
        if commonality.get("source_count") != len(expected):
            errors.append("weight commonality source count mismatch")
    report_body = dict(report)
    report_fingerprint = report_body.pop("report_fingerprint", None)
    if report_fingerprint != _fingerprint(report_body):
        errors.append("weight commonality report fingerprint mismatch")
    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "source_count": len(models),
        "stable_role_count": (
            int(commonality.get("stable_role_count", 0))
            if isinstance(commonality, Mapping)
            else 0
        ),
        "report_fingerprint": report_fingerprint,
    }


def load_weight_commonality_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weight commonality report root must be an object")
    return payload

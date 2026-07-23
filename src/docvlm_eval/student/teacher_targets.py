"""Offline sequence distillation for teachers with a different tokenizer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from docvlm_eval.metrics import score_sample


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX = re.compile(r"^\s*(?:final\s+)?answer\s*:\s*", re.IGNORECASE)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(value).encode('utf-8')).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def _valid_answers(raw: Any) -> list[str]:
    return [str(value).strip() for value in (raw or []) if str(value).strip()]


def normalize_teacher_target(raw: str, *, max_length: int = 8192) -> tuple[str, str] | None:
    """Return normalized response and answer used by the quality gate."""

    response = unicodedata.normalize("NFC", str(raw or "")).strip()
    response = _THINK_BLOCK.sub("", response).strip()
    if response.startswith("```") and response.endswith("```"):
        lines = response.splitlines()
        response = "\n".join(lines[1:-1]).strip()
    if not response or len(response) > max_length:
        return None
    lower = response.lower()
    if lower.startswith(("i cannot", "i can't", "i am unable", "unable to", "sorry")):
        return None

    answer = response
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), (str, int, float)):
        answer = str(parsed["answer"]).strip()
        response = _stable_json(parsed)
    elif len(response.splitlines()) == 1:
        answer = _ANSWER_PREFIX.sub("", response).strip()
    if not answer:
        return None
    return response, answer


def _materialize_image(image: Any, output: Path) -> None:
    from PIL import Image

    if isinstance(image, Image.Image):
        image.convert("RGB").save(output, format="PNG")
        return
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            opened.convert("RGB").save(output, format="PNG")
        return
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            import io

            with Image.open(io.BytesIO(image["bytes"])) as opened:
                opened.convert("RGB").save(output, format="PNG")
            return
        if image.get("path"):
            _materialize_image(image["path"], output)
            return
    raise TypeError(f"unsupported teacher-request image payload: {type(image).__name__}")


def export_teacher_requests(
    dataset: Any,
    output_dir: str | Path,
    *,
    max_requests: int | None = None,
    selection_seed: int = 0,
) -> dict[str, Any]:
    """Export immutable image-question requests from an on-disk UDD dataset."""

    if max_requests is not None and max_requests <= 0:
        raise ValueError("teacher max_requests must be positive or null")
    if selection_seed < 0:
        raise ValueError("teacher request selection_seed must be non-negative")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty request output {output_dir}")
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "requests.jsonl"
    from .data import _metadata_view

    metadata = _metadata_view(dataset)
    candidates: list[dict[str, Any]] = []
    metrics: Counter[str] = Counter()
    for row_index in range(len(metadata)):
        row = metadata[row_index]
        instructions = list(row.get("instructions") or [])
        answers = list(row.get("answers") or [])
        if len(instructions) != len(answers):
            raise ValueError(
                f"row {row_index} has {len(instructions)} instructions and "
                f"{len(answers)} answer lists"
            )
        sample_id = str(row.get("sample_id") or f"row-{row_index}")
        metric = str(row.get("metric") or "anls")
        for qa_index, (question, golds) in enumerate(
            zip(instructions, answers)
        ):
            question = str(question).strip()
            valid_answers = _valid_answers(golds)
            if not question or not valid_answers:
                continue
            selection_core = {
                "row_index": row_index,
                "qa_index": qa_index,
                "sample_id": sample_id,
                "question": question,
                "gold_answers": valid_answers,
                "metric": metric,
            }
            candidates.append(
                {
                    **selection_core,
                    "selection_priority": _fingerprint(
                        {
                            "seed": selection_seed,
                            "request": selection_core,
                        }
                    ),
                }
            )
    if not candidates:
        raise ValueError("UDD dataset produced no teacher requests")
    eligible_requests = len(candidates)
    if max_requests is not None:
        candidates = sorted(
            candidates,
            key=lambda item: (
                item["selection_priority"],
                item["row_index"],
                item["qa_index"],
            ),
        )[:max_requests]
    candidates.sort(key=lambda item: (item["row_index"], item["qa_index"]))
    selected_rows = sorted({int(item["row_index"]) for item in candidates})
    image_metadata: dict[int, tuple[Path, str]] = {}
    for row_index in selected_rows:
        row = dataset[row_index]
        image_path = image_dir / f"row-{row_index:08d}.png"
        _materialize_image(row.get("image") or row.get("image_path"), image_path)
        image_metadata[row_index] = (image_path, _file_sha256(image_path))
    with request_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            image_path, image_sha256 = image_metadata[
                int(candidate["row_index"])
            ]
            core = {
                key: value
                for key, value in candidate.items()
                if key != "selection_priority"
            }
            core["image_sha256"] = image_sha256
            request = {
                "schema_version": 1,
                "request_id": _fingerprint(core),
                "image_path": str(image_path.resolve()),
                **core,
            }
            handle.write(_stable_json(request) + "\n")
            metrics[str(candidate["metric"])] += 1
    manifest = {
        "schema_version": 1,
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "eligible_requests": eligible_requests,
        "max_requests": max_requests,
        "selection_seed": selection_seed,
        "rows_with_requests": len(selected_rows),
        "requests": len(candidates),
        "metrics": dict(sorted(metrics.items())),
        "requests_path": str(request_path.resolve()),
        "requests_sha256": _file_sha256(request_path),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def generate_teacher_predictions(
    requests_path: str | Path,
    output_path: str | Path,
    *,
    model_key: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    model_revision: str | None = None,
    temperature: float = 0.0,
    resume: bool = True,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Generate one resumable prediction per request with any registered VLM adapter."""

    from docvlm_eval.models import GenConfig, build_model

    requests_path = Path(requests_path)
    output_path = Path(output_path)
    requests = list(_iter_jsonl(requests_path))
    if adapter is None:
        adapter = build_model(
            model_key,
            revision=model_revision,
            device=device,
            dtype=dtype,
            gen=GenConfig(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
            ),
        )
    teacher_hf_id = str(getattr(adapter, "hf_id", "") or "")
    generation_config = {
        "teacher_model": model_key,
        "teacher_hf_id": teacher_hf_id,
        "teacher_revision": model_revision,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    generation_fingerprint = _fingerprint(generation_config)
    existing: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        if not resume:
            raise FileExistsError(f"prediction output already exists: {output_path}")
        for prediction in _iter_jsonl(output_path):
            request_id = str(prediction.get("request_id") or "")
            if not request_id or request_id in existing:
                raise ValueError("existing predictions contain a missing or duplicate request_id")
            if prediction.get("generation_fingerprint") != generation_fingerprint:
                raise ValueError(
                    "existing predictions use a different teacher or generation configuration"
                )
            existing[request_id] = prediction
    elif output_path.exists():
        raise ValueError(f"prediction output is not a file: {output_path}")

    pending = [
        request for request in requests if str(request["request_id"]) not in existing
    ]
    if pending:
        if not getattr(adapter, "_loaded", False):
            adapter.load()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for request in pending:
            request_id = str(request["request_id"])
            assert adapter is not None
            prompt = (
                f"{request['question']}\n"
                "Answer using only the requested value. Do not explain your answer."
            )
            response, confidence = adapter.generate(str(request["image_path"]), prompt)
            prediction = {
                "schema_version": 1,
                "request_id": request_id,
                "request_sha256": _fingerprint(request),
                "teacher_model": model_key,
                "teacher_hf_id": teacher_hf_id,
                "teacher_revision": model_revision,
                "generation_fingerprint": generation_fingerprint,
                "response": str(response),
                "confidence": (
                    float(confidence) if confidence is not None else None
                ),
            }
            handle.write(_stable_json(prediction) + "\n")
            handle.flush()
            generated += 1
    predictions = list(_iter_jsonl(output_path))
    request_ids = {str(request["request_id"]) for request in requests}
    prediction_ids = {str(prediction.get("request_id") or "") for prediction in predictions}
    missing = request_ids - prediction_ids
    unknown = prediction_ids - request_ids
    if missing or unknown or len(predictions) != len(prediction_ids):
        raise ValueError(
            "teacher prediction coverage mismatch: "
            f"missing={len(missing)}, unknown={len(unknown)}, duplicates="
            f"{len(predictions) - len(prediction_ids)}"
        )
    manifest = {
        "schema_version": 1,
        "teacher_model": model_key,
        "teacher_hf_id": teacher_hf_id,
        "teacher_revision": model_revision,
        "device": device,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "generation_fingerprint": generation_fingerprint,
        "requests": len(requests),
        "generated_now": generated,
        "predictions_sha256": _file_sha256(output_path),
    }
    _atomic_write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def apply_teacher_predictions(
    dataset: Any,
    requests_path: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    min_score: float,
    min_acceptance_rate: float = 0.0,
    target_format: str = "answer",
    accepted_target_count: int | None = None,
    selection_seed: int = 0,
    expected_model: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Quality-gate teacher outputs and add aligned sequence targets without changing gold."""

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("teacher target min_score must be within [0, 1]")
    if not 0.0 <= min_acceptance_rate <= 1.0:
        raise ValueError("teacher target min_acceptance_rate must be within [0, 1]")
    if target_format not in {"answer", "response"}:
        raise ValueError("target_format must be answer or response")
    if accepted_target_count is not None and accepted_target_count <= 0:
        raise ValueError("accepted_target_count must be positive or null")
    if selection_seed < 0:
        raise ValueError("teacher target selection_seed must be non-negative")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty target output {output_dir}")

    requests_path = Path(requests_path)
    predictions_path = Path(predictions_path)
    requests = list(_iter_jsonl(requests_path))
    request_manifest_path = requests_path.with_name("manifest.json")
    if request_manifest_path.is_file():
        request_manifest = json.loads(request_manifest_path.read_text(encoding="utf-8"))
        expected = request_manifest.get("source_fingerprint")
        actual = getattr(dataset, "_fingerprint", None)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError("teacher requests were exported from a different dataset fingerprint")
    request_ids = {str(item.get("request_id") or "") for item in requests}
    if len(request_ids) != len(requests) or "" in request_ids:
        raise ValueError("teacher requests contain missing or duplicate request IDs")
    predictions = {
        str(item.get("request_id") or ""): item
        for item in _iter_jsonl(predictions_path)
    }
    if len(predictions) == 0:
        raise ValueError("teacher predictions are empty")
    if len(predictions) != sum(1 for _ in _iter_jsonl(predictions_path)):
        raise ValueError("teacher predictions contain duplicate request IDs")
    unknown_predictions = set(predictions) - request_ids
    if unknown_predictions:
        raise ValueError(
            f"teacher predictions contain {len(unknown_predictions)} unknown request IDs"
        )
    if expected_model is not None and any(
        prediction.get("teacher_model") != expected_model
        for prediction in predictions.values()
    ):
        raise ValueError("teacher prediction model does not match the experiment")
    if expected_revision is not None and any(
        prediction.get("teacher_revision") != expected_revision
        for prediction in predictions.values()
    ):
        raise ValueError(
            "teacher prediction revision does not match the experiment"
        )

    accepted: dict[tuple[int, int], dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    teachers: Counter[str] = Counter()
    scores: list[float] = []
    for request in requests:
        row_index = int(request["row_index"])
        qa_index = int(request["qa_index"])
        if not 0 <= row_index < len(dataset):
            raise ValueError(f"teacher request row_index is out of range: {row_index}")
        source_row = dataset[row_index]
        instructions = list(source_row.get("instructions") or [])
        answers = list(source_row.get("answers") or [])
        if not 0 <= qa_index < len(instructions):
            raise ValueError(f"teacher request qa_index is out of range: {qa_index}")
        if (
            str(source_row.get("sample_id") or f"row-{row_index}") != request["sample_id"]
            or str(instructions[qa_index]).strip() != request["question"]
            or _valid_answers(answers[qa_index]) != list(request["gold_answers"])
        ):
            raise ValueError("teacher request payload does not match the source UDD row")
        request_id = str(request["request_id"])
        prediction = predictions.get(request_id)
        if prediction is None:
            reasons["missing_prediction"] += 1
            continue
        if prediction.get("request_sha256") != _fingerprint(request):
            reasons["request_fingerprint_mismatch"] += 1
            continue
        normalized = normalize_teacher_target(str(prediction.get("response") or ""))
        if normalized is None:
            reasons["invalid_response"] += 1
            continue
        response, answer = normalized
        metric = str(request.get("metric") or "anls")
        score = float(score_sample(metric, answer, list(request["gold_answers"])))
        scores.append(score)
        if score < min_score:
            reasons["below_score_threshold"] += 1
            continue
        teacher = str(prediction.get("teacher_model") or "unknown")
        teachers[teacher] += 1
        accepted[(row_index, qa_index)] = {
            "target": answer if target_format == "answer" else response,
            "score": score,
            "request_id": request_id,
            "teacher_model": teacher,
            "confidence": prediction.get("confidence"),
            "generation_fingerprint": prediction.get("generation_fingerprint"),
            "metric": metric,
        }

    eligible_targets = len(accepted)
    eligible_acceptance_rate = eligible_targets / max(1, len(requests))
    if eligible_acceptance_rate < min_acceptance_rate:
        raise RuntimeError(
            "teacher target acceptance rate "
            f"{eligible_acceptance_rate:.3f} is below required "
            f"{min_acceptance_rate:.3f}"
        )
    if accepted_target_count is not None:
        if eligible_targets < accepted_target_count:
            raise RuntimeError(
                f"teacher produced {eligible_targets} eligible targets, fewer "
                f"than required fixed dose {accepted_target_count}"
            )
        selected_keys = sorted(
            accepted,
            key=lambda key: _fingerprint(
                {
                    "seed": selection_seed,
                    "request_id": accepted[key]["request_id"],
                }
            ),
        )[:accepted_target_count]
        accepted = {key: accepted[key] for key in selected_keys}
    acceptance_rate = len(accepted) / max(1, len(requests))
    selected_teachers = Counter(
        str(target["teacher_model"]) for target in accepted.values()
    )

    def add_targets(row: dict[str, Any], row_index: int) -> dict[str, Any]:
        instructions = list(row.get("instructions") or [])
        teacher_answers = [""] * len(instructions)
        teacher_scores = [0.0] * len(instructions)
        provenance: dict[str, Any] = {}
        for qa_index in range(len(instructions)):
            target = accepted.get((row_index, qa_index))
            if target is None:
                continue
            teacher_answers[qa_index] = str(target["target"])
            teacher_scores[qa_index] = float(target["score"])
            provenance[str(qa_index)] = {
                key: value
                for key, value in target.items()
                if key != "target"
            }
        return {
            "teacher_answers": teacher_answers,
            "teacher_scores": teacher_scores,
            "teacher_provenance_json": _stable_json(provenance),
        }

    enriched = dataset.map(
        add_targets,
        with_indices=True,
        desc="apply cross-tokenizer teacher targets",
        load_from_cache_file=False,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    enriched.save_to_disk(str(output_dir))
    manifest = {
        "schema_version": 1,
        "source_fingerprint": getattr(dataset, "_fingerprint", None),
        "dataset_fingerprint": getattr(enriched, "_fingerprint", None),
        "requests": len(requests),
        "predictions": len(predictions),
        "eligible": eligible_targets,
        "eligible_acceptance_rate": eligible_acceptance_rate,
        "accepted": len(accepted),
        "acceptance_rate": acceptance_rate,
        "accepted_target_count": accepted_target_count,
        "selection_seed": selection_seed,
        "expected_model": expected_model,
        "expected_revision": expected_revision,
        "min_score": min_score,
        "min_acceptance_rate": min_acceptance_rate,
        "target_format": target_format,
        "mean_candidate_score": sum(scores) / max(1, len(scores)),
        "rejections": dict(sorted(reasons.items())),
        "eligible_teachers": dict(sorted(teachers.items())),
        "teachers": dict(sorted(selected_teachers.items())),
        "requests_sha256": _file_sha256(Path(requests_path)),
        "predictions_sha256": _file_sha256(Path(predictions_path)),
    }
    _atomic_write_json(output_dir / "teacher_target_manifest.json", manifest)
    return manifest

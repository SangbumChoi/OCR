import json
from pathlib import Path

import pytest

from docvlm_eval.student.data import UDDStudentDataset
from docvlm_eval.student.teacher_targets import (
    apply_teacher_predictions,
    export_teacher_requests,
    generate_teacher_predictions,
    normalize_teacher_target,
)
from docvlm_eval.student.tokenizer import iter_udd_text


class _FakeTeacher:
    _loaded = False

    def load(self):
        self._loaded = True

    def generate(self, image_path: str, question: str):
        assert Path(image_path).is_file()
        if "first field" in question:
            return "Answer: alpha", 0.9
        return "wrong", 0.2


class _PerfectTeacher(_FakeTeacher):
    def generate(self, image_path: str, question: str):
        assert Path(image_path).is_file()
        return ("alpha" if "first field" in question else "beta"), 0.9


def _dataset(tmp_path: Path):
    from datasets import Dataset
    from PIL import Image

    image = tmp_path / "page.png"
    Image.new("RGB", (20, 12), "white").save(image)
    return Dataset.from_list(
        [
            {
                "image": str(image),
                "sample_id": "doc-1",
                "source": "synthetic",
                "task": "vqa",
                "instructions": ["What is the first field?", "What is the second field?"],
                "answers": [["alpha"], ["beta"]],
                "metric": "exact",
                "language": "en",
            }
        ]
    )


def test_teacher_target_normalization_handles_json_and_rejections():
    assert normalize_teacher_target('{"answer": "42", "rationale": "sum"}') == (
        '{"answer":"42","rationale":"sum"}',
        "42",
    )
    assert normalize_teacher_target("Answer: value") == ("Answer: value", "value")
    assert normalize_teacher_target("<think>hidden</think>\nvalue") == ("value", "value")
    assert normalize_teacher_target("I cannot read this") is None


def test_cross_tokenizer_teacher_targets_are_quality_gated_and_consumed(tmp_path):
    from datasets import load_from_disk

    source = _dataset(tmp_path)
    requests = tmp_path / "requests"
    export_manifest = export_teacher_requests(source, requests)
    assert export_manifest["requests"] == 2
    assert export_manifest["rows_with_requests"] == 1

    predictions = tmp_path / "predictions.jsonl"
    revision = "a" * 40
    generation = generate_teacher_predictions(
        requests / "requests.jsonl",
        predictions,
        model_key="fake-teacher",
        model_revision=revision,
        device="cpu",
        dtype="float32",
        max_new_tokens=16,
        adapter=_FakeTeacher(),
    )
    assert generation["generated_now"] == 2
    resumed = generate_teacher_predictions(
        requests / "requests.jsonl",
        predictions,
        model_key="fake-teacher",
        model_revision=revision,
        device="cpu",
        dtype="float32",
        max_new_tokens=16,
        adapter=_FakeTeacher(),
    )
    assert resumed["generated_now"] == 0
    assert len(predictions.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError, match="different teacher or generation"):
        generate_teacher_predictions(
            requests / "requests.jsonl",
            predictions,
            model_key="different-teacher",
            model_revision=revision,
            device="cpu",
            dtype="float32",
            max_new_tokens=16,
            adapter=_FakeTeacher(),
        )

    with pytest.raises(RuntimeError, match="acceptance rate"):
        apply_teacher_predictions(
            source,
            requests / "requests.jsonl",
            predictions,
            tmp_path / "rejected-rate",
            min_score=0.8,
            min_acceptance_rate=0.6,
        )

    output = tmp_path / "enriched"
    manifest = apply_teacher_predictions(
        source,
        requests / "requests.jsonl",
        predictions,
        output,
        min_score=0.8,
        accepted_target_count=1,
        expected_model="fake-teacher",
        expected_revision=revision,
    )
    assert manifest["eligible"] == 1
    assert manifest["accepted"] == 1
    assert manifest["rejections"] == {"below_score_threshold": 1}
    assert manifest["teachers"] == {"fake-teacher": 1}

    enriched = load_from_disk(str(output))
    assert enriched[0]["teacher_answers"] == ["alpha", ""]
    assert enriched[0]["teacher_scores"] == pytest.approx([1.0, 0.0])
    provenance = json.loads(enriched[0]["teacher_provenance_json"])
    assert provenance["0"]["teacher_model"] == "fake-teacher"

    distilled = UDDStudentDataset(
        enriched,
        include_grounding=False,
        teacher_target_probability=1.0,
        teacher_min_score=0.8,
    )
    assert distilled[0].answer == "alpha"
    assert distilled[0].target_source == "teacher"
    assert distilled[1].answer == "beta"
    assert distilled[1].target_source == "gold"
    assert distilled.target_sources == ["teacher", "gold"]

    gold_only = UDDStudentDataset(
        enriched,
        include_grounding=False,
        teacher_target_probability=0.0,
    )
    assert gold_only[0].target_source == "gold"
    all_text = list(iter_udd_text(enriched))
    gold_text = list(
        iter_udd_text(enriched, include_teacher_targets=False)
    )
    assert all_text.count("alpha") == gold_text.count("alpha") + 1


def test_teacher_request_and_target_budgets_are_deterministic(tmp_path):
    from datasets import load_from_disk

    source = _dataset(tmp_path)
    first = tmp_path / "requests-first"
    second = tmp_path / "requests-second"
    first_manifest = export_teacher_requests(
        source,
        first,
        max_requests=1,
        selection_seed=19,
    )
    export_teacher_requests(
        source,
        second,
        max_requests=1,
        selection_seed=19,
    )
    assert first_manifest["eligible_requests"] == 2
    assert first_manifest["requests"] == 1
    assert first_manifest["selection_seed"] == 19
    first_request = json.loads(
        (first / "requests.jsonl").read_text(encoding="utf-8")
    )
    second_request = json.loads(
        (second / "requests.jsonl").read_text(encoding="utf-8")
    )
    assert first_request["request_id"] == second_request["request_id"]

    all_requests = tmp_path / "requests-all"
    export_teacher_requests(source, all_requests)
    predictions = tmp_path / "perfect.jsonl"
    revision = "b" * 40
    generate_teacher_predictions(
        all_requests / "requests.jsonl",
        predictions,
        model_key="perfect-teacher",
        model_revision=revision,
        device="cpu",
        dtype="float32",
        max_new_tokens=16,
        adapter=_PerfectTeacher(),
    )
    output = tmp_path / "fixed-dose"
    manifest = apply_teacher_predictions(
        source,
        all_requests / "requests.jsonl",
        predictions,
        output,
        min_score=0.8,
        accepted_target_count=1,
        selection_seed=23,
        expected_model="perfect-teacher",
        expected_revision=revision,
    )
    assert manifest["eligible"] == 2
    assert manifest["accepted"] == 1
    assert manifest["accepted_target_count"] == 1
    enriched = load_from_disk(str(output))
    assert sum(bool(answer) for answer in enriched[0]["teacher_answers"]) == 1

    with pytest.raises(ValueError, match="revision does not match"):
        apply_teacher_predictions(
            source,
            all_requests / "requests.jsonl",
            predictions,
            tmp_path / "wrong-revision",
            min_score=0.8,
            expected_model="perfect-teacher",
            expected_revision="c" * 40,
        )


def test_teacher_apply_rejects_predictions_for_another_request(tmp_path):
    source = _dataset(tmp_path)
    requests = tmp_path / "requests"
    export_teacher_requests(source, requests)
    request = json.loads(
        (requests / "requests.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    prediction = {
        "request_id": "sha256:unknown",
        "request_sha256": "sha256:unknown",
        "teacher_model": "fake",
        "response": "alpha",
    }
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown request IDs"):
        apply_teacher_predictions(
            source,
            requests / "requests.jsonl",
            predictions,
            tmp_path / "out",
            min_score=0.8,
        )
    assert request["request_id"] != prediction["request_id"]

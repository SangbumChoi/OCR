import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _blueprint():
    from docvlm_eval.architecture import load_blueprint

    return load_blueprint(ROOT / "configs" / "sub1b_architecture.yaml")


def _comparison(train_score, heldout_score, languages=None):
    languages = languages or {
        "en": {"score": heldout_score},
        "ko": {"score": heldout_score},
    }
    train = {
        "score": train_score,
        "by_language": languages,
    }
    heldout = {
        "score": heldout_score,
        "by_language": languages,
    }
    return {
        "splits": {"train": train, "heldout": heldout},
        "train_minus_heldout": {
            "headline": {"score": train_score - heldout_score}
        },
    }


def _row(
    sample_id,
    score,
    *,
    answer_type="default",
    confidence=0.9,
    meta=None,
    components=None,
):
    components = components or {}
    return {
        "sample_id": sample_id,
        "score": score,
        "answer": "correct" if score == 1.0 else "",
        "answer_type": answer_type,
        "confidence": confidence,
        "meta": meta or {},
        "reward_components": components,
        "applicable_rewards": list(components),
    }


def _matched_gate_rows():
    current = [
        _row("box", 0.8, components={"box_iou": 0.8}),
        _row(
            "ocr",
            0.8,
            answer_type="ocr-transcription",
            components={"normalized_text_similarity": 0.8},
        ),
        _row(
            "reason-a",
            0.9,
            answer_type="numeric-reasoning",
            meta={"hypothesis": "pair-a", "control": False},
        ),
        _row(
            "reason-a-control",
            0.85,
            answer_type="numeric-reasoning",
            meta={"hypothesis": "pair-a", "control": True},
        ),
        _row(
            "reason-b",
            0.9,
            answer_type="relational-reasoning",
            meta={"hypothesis": "pair-b", "control": False},
        ),
        _row(
            "reason-b-control",
            0.85,
            answer_type="relational-reasoning",
            meta={"hypothesis": "pair-b", "control": True},
        ),
        _row(
            "absence",
            1.0,
            answer_type="context-absence",
            meta={"abstain_expected": True},
        ),
    ]
    baseline = [
        _row("box", 0.7, components={"box_iou": 0.7}),
        _row(
            "ocr",
            0.8,
            answer_type="ocr-transcription",
            components={"normalized_text_similarity": 0.8},
        ),
        _row(
            "reason-a",
            0.7,
            answer_type="numeric-reasoning",
            meta={"hypothesis": "pair-a", "control": False},
        ),
        _row(
            "reason-a-control",
            0.7,
            answer_type="numeric-reasoning",
            meta={"hypothesis": "pair-a", "control": True},
        ),
        _row(
            "reason-b",
            0.7,
            answer_type="relational-reasoning",
            meta={"hypothesis": "pair-b", "control": False},
        ),
        _row(
            "reason-b-control",
            0.7,
            answer_type="relational-reasoning",
            meta={"hypothesis": "pair-b", "control": True},
        ),
        _row(
            "absence",
            1.0,
            answer_type="context-absence",
            meta={"abstain_expected": True},
        ),
    ]
    return current, baseline


def test_gates_do_not_turn_missing_comparisons_into_success():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 900_000_000},
        _comparison(0.8, 0.7),
        {"heldout": [_row("one", 0.7)]},
    )

    statuses = {gate["id"]: gate["status"] for gate in report["gates"]}
    assert statuses["parameter_budget"] == "pass"
    assert set(statuses.values()) == {"pass", "insufficient_evidence"}
    assert report["overall_status"] == "insufficient_evidence"
    assert report["counts"] == {
        "pass": 1,
        "fail": 0,
        "insufficient_evidence": 5,
    }


def test_gates_pass_with_matched_reference_and_monolingual_evidence():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    current_rows, baseline_rows = _matched_gate_rows()
    report = evaluate_deployment_gates(
        _blueprint(),
        {"vision": 100, "language": 200, "total": 300},
        _comparison(0.82, 0.8),
        {"heldout": current_rows},
        baseline_comparison=_comparison(0.74, 0.7),
        baseline_rows={"heldout": baseline_rows},
        monolingual_control_comparison=_comparison(
            0.82,
            0.8,
            {
                "en": {"score": 0.81},
                "ko": {"score": 0.81},
            },
        ),
    )

    assert report["overall_status"] == "pass"
    assert report["counts"] == {
        "pass": 6,
        "fail": 0,
        "insufficient_evidence": 0,
    }
    evidence = {gate["id"]: gate["evidence"] for gate in report["gates"]}
    assert evidence["generalization"]["heldout_score_delta"] == 0.1
    assert evidence["grounding"]["box_iou_delta"] == 0.1
    assert evidence["reasoning"]["complete_counterfactual_groups"] == 2
    assert evidence["multilingual"]["language_drops"] == {
        "en": 0.01,
        "ko": 0.01,
    }


def test_gate_report_fails_real_threshold_violations():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    current_rows, baseline_rows = _matched_gate_rows()
    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 1_000_000_000},
        _comparison(0.9, 0.6),
        {"heldout": current_rows},
        baseline_comparison=_comparison(0.71, 0.7),
        baseline_rows={"heldout": baseline_rows},
        monolingual_control_comparison=_comparison(
            0.9,
            0.9,
            {"en": {"score": 0.9}, "ko": {"score": 0.9}},
        ),
    )

    statuses = {gate["id"]: gate["status"] for gate in report["gates"]}
    assert statuses["parameter_budget"] == "fail"
    assert statuses["generalization"] == "fail"
    assert statuses["multilingual"] == "fail"
    assert report["overall_status"] == "fail"


def test_gate_artifacts_round_trip(tmp_path):
    from docvlm_eval.student.gates import (
        load_evaluation_artifacts,
        write_gate_report,
    )

    evaluation = tmp_path / "evaluation"
    heldout = evaluation / "heldout"
    heldout.mkdir(parents=True)
    comparison = _comparison(0.8, 0.7)
    (evaluation / "comparison.json").write_text(
        json.dumps(comparison),
        encoding="utf-8",
    )
    row = _row("one", 0.7)
    (heldout / "per_sample.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    loaded_comparison, loaded_rows = load_evaluation_artifacts(evaluation)
    assert loaded_comparison == comparison
    assert loaded_rows == {"heldout": [row]}
    path = write_gate_report(
        evaluation / "gates.json",
        {"schema_version": 1, "overall_status": "pass"},
    )
    assert json.loads(path.read_text(encoding="utf-8"))["overall_status"] == "pass"

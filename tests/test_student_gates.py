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


def _visual_report(
    *,
    device_type="cuda",
    resolved_backend="flex",
    speedup=1.2,
    min_speedup=1.1,
    memory_ratio=0.95,
    numerical_delta=0.01,
    dense_speedup=1.15,
    dense_min_speedup=1.05,
    dense_memory_ratio=0.9,
    measured_iterations=10,
    rounds=3,
    schema_version=2,
):
    from docvlm_eval.student.config import (
        StudentConfig,
        student_config_fingerprint,
    )

    student = StudentConfig.from_blueprint(_blueprint())
    return {
        "schema_version": schema_version,
        "scope": "student_vision_tower_and_gated_resampler",
        "student_config_fingerprint": student_config_fingerprint(student),
        "student_config": student.to_dict(),
        "benchmark_config": {
            "mode": "training",
            "warmup_iterations": 3,
            "measured_iterations": measured_iterations,
            "rounds": rounds,
        },
        "resolved_precision": "bfloat16",
        "environment": {
            "device": "cuda:0" if device_type == "cuda" else "cpu",
            "device_type": device_type,
            "device_name": "test-gpu" if device_type == "cuda" else None,
            "torch": "2.9.1",
            "cuda": "12.8" if device_type == "cuda" else None,
        },
        "visual_tokens": 5040,
        "batch_size": 2,
        "results": [
            {
                "status": "ok",
                "requested_backend": "loop",
                "resolved_backend": "loop",
                "median_ms": 120.0,
            },
            {
                "status": "ok",
                "requested_backend": "auto",
                "resolved_backend": resolved_backend,
                "median_ms": 100.0,
                "rounds": rounds,
                "paired_rounds_vs_loop": rounds,
                "paired_rounds_vs_dense_adaptive": rounds,
                "median_speedup_vs_loop": speedup,
                "min_speedup_vs_loop": min_speedup,
                "peak_memory_ratio_vs_loop": memory_ratio,
                "max_abs_delta_vs_loop": numerical_delta,
                "median_speedup_vs_dense_adaptive": dense_speedup,
                "min_speedup_vs_dense_adaptive": dense_min_speedup,
                "peak_memory_ratio_vs_dense_adaptive": dense_memory_ratio,
            },
            {
                "status": "ok",
                "requested_backend": "dense_adaptive",
                "resolved_backend": "dense",
                "median_ms": 115.0,
            },
        ],
    }


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
        "insufficient_evidence": 6,
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
        visual_backend_report=_visual_report(),
    )

    assert report["overall_status"] == "pass"
    assert report["counts"] == {
        "pass": 7,
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
    assert evidence["visual_efficiency"]["median_speedup_vs_loop"] == 1.2
    assert (
        evidence["visual_efficiency"][
            "median_speedup_vs_dense_adaptive"
        ]
        == 1.15
    )


def test_visual_efficiency_gate_rejects_fallback_and_regressions():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    current_rows, baseline_rows = _matched_gate_rows()
    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.82, 0.8),
        {"heldout": current_rows},
        baseline_comparison=_comparison(0.74, 0.7),
        baseline_rows={"heldout": baseline_rows},
        visual_backend_report=_visual_report(
            resolved_backend="loop",
            speedup=0.9,
            memory_ratio=1.1,
            numerical_delta=0.03,
        ),
    )

    gate = next(
        gate for gate in report["gates"] if gate["id"] == "visual_efficiency"
    )
    assert gate["status"] == "fail"
    assert gate["evidence"]["violations"] == [
        "resolved_backend",
        "median_speedup",
        "peak_memory",
        "numerical_parity",
    ]


def test_visual_efficiency_gate_requires_target_gpu_and_measurement_dose():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    cpu = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(device_type="cpu"),
    )
    short = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(measured_iterations=2),
    )
    few_rounds = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(rounds=1),
    )
    legacy = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(schema_version=1),
    )
    malformed = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(schema_version="legacy"),
    )

    cpu_gate = next(
        gate for gate in cpu["gates"] if gate["id"] == "visual_efficiency"
    )
    short_gate = next(
        gate for gate in short["gates"] if gate["id"] == "visual_efficiency"
    )
    few_rounds_gate = next(
        gate
        for gate in few_rounds["gates"]
        if gate["id"] == "visual_efficiency"
    )
    legacy_gate = next(
        gate for gate in legacy["gates"] if gate["id"] == "visual_efficiency"
    )
    malformed_gate = next(
        gate
        for gate in malformed["gates"]
        if gate["id"] == "visual_efficiency"
    )
    assert cpu_gate["status"] == "insufficient_evidence"
    assert short_gate["status"] == "insufficient_evidence"
    assert few_rounds_gate["status"] == "insufficient_evidence"
    assert legacy_gate["status"] == "insufficient_evidence"
    assert malformed_gate["status"] == "insufficient_evidence"


def test_visual_efficiency_gate_rejects_dense_control_regression():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(
            dense_speedup=0.95,
            dense_memory_ratio=1.1,
        ),
    )

    gate = next(
        gate for gate in report["gates"] if gate["id"] == "visual_efficiency"
    )
    assert gate["status"] == "fail"
    assert gate["evidence"]["violations"] == [
        "dense_adaptive_speedup",
        "dense_adaptive_peak_memory",
    ]


def test_visual_efficiency_gate_rejects_any_regressive_round():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(
            min_speedup=0.99,
            dense_min_speedup=0.98,
        ),
    )

    gate = next(
        gate for gate in report["gates"] if gate["id"] == "visual_efficiency"
    )
    assert gate["status"] == "fail"
    assert gate["evidence"]["violations"] == [
        "round_speedup",
        "dense_adaptive_round_speedup",
    ]


def test_visual_efficiency_gate_does_not_approve_dense_execution():
    from copy import deepcopy

    from docvlm_eval.student.gates import evaluate_deployment_gates

    blueprint = deepcopy(_blueprint())
    blueprint["training"]["pretraining"]["input_pipeline"][
        "visual_sequence_mode"
    ] = "dense"
    report = evaluate_deployment_gates(
        blueprint,
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=_visual_report(),
    )

    gate = next(
        gate for gate in report["gates"] if gate["id"] == "visual_efficiency"
    )
    assert gate["status"] == "insufficient_evidence"
    assert gate["evidence"]["visual_sequence_mode"] == "dense"


def test_visual_efficiency_gate_rejects_wrong_student_config():
    from docvlm_eval.student.gates import evaluate_deployment_gates

    visual = _visual_report()
    visual["student_config"]["vision"]["layers"] = 1
    report = evaluate_deployment_gates(
        _blueprint(),
        {"total": 300},
        _comparison(0.8, 0.7),
        {"heldout": []},
        visual_backend_report=visual,
    )

    gate = next(
        gate for gate in report["gates"] if gate["id"] == "visual_efficiency"
    )
    assert gate["status"] == "fail"


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
        load_visual_backend_report,
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
    visual_path = tmp_path / "visual_backend.json"
    visual_path.write_text(
        json.dumps(_visual_report()),
        encoding="utf-8",
    )
    assert load_visual_backend_report(visual_path)["visual_tokens"] == 5040

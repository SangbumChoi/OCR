"""End-to-end pipeline with the dummy model: artefacts, scoring, resume."""

import json
from pathlib import Path

from docvlm_eval.pipeline import run_evaluation


def test_run_writes_artifacts(tiny_benchmark, tmp_path):
    samples, _ = tiny_benchmark
    out = tmp_path / "run"
    summary = run_evaluation("dummy-echo", samples, str(out), device="cpu", benchmark_name="t")
    assert (out / "predictions.jsonl").exists()
    assert (out / "summary.json").exists()
    assert (out / "per_sample.json").exists()
    assert summary["n_samples"] == 3
    assert summary["model"] == "dummy-echo"
    assert summary["benchmark"] == "t"
    assert summary["ece"] is not None  # dummy emits confidences


def test_limit(tiny_benchmark, tmp_path):
    samples, _ = tiny_benchmark
    out = tmp_path / "run"
    summary = run_evaluation("dummy-echo", samples, str(out), device="cpu", limit=2)
    assert summary["n_samples"] == 2


def test_resume_does_not_duplicate(tiny_benchmark, tmp_path):
    samples, _ = tiny_benchmark
    out = tmp_path / "run"
    run_evaluation("dummy-echo", samples, str(out), device="cpu")
    n_first = len((out / "predictions.jsonl").read_text().strip().splitlines())
    # second run should reuse cached predictions, not append duplicates
    run_evaluation("dummy-echo", samples, str(out), device="cpu", resume=True)
    n_second = len((out / "predictions.jsonl").read_text().strip().splitlines())
    assert n_first == n_second == 3


def test_no_resume_truncates_not_appends(tiny_benchmark, tmp_path):
    samples, _ = tiny_benchmark
    out = tmp_path / "run"
    run_evaluation("dummy-echo", samples, str(out), device="cpu")
    # a second run WITHOUT resume must rewrite, not append duplicate lines
    run_evaluation("dummy-echo", samples, str(out), device="cpu", resume=False)
    n = len((out / "predictions.jsonl").read_text().strip().splitlines())
    assert n == 3  # not 6


def test_predictions_jsonl_schema(tiny_benchmark, tmp_path):
    samples, _ = tiny_benchmark
    out = tmp_path / "run"
    run_evaluation("dummy-echo", samples, str(out), device="cpu")
    line = (out / "predictions.jsonl").read_text().splitlines()[0]
    rec = json.loads(line)
    assert {"sample_id", "prediction", "confidence"}.issubset(rec)

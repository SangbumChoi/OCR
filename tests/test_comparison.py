"""Comparison-table builder from synthetic run summaries."""

import json
from pathlib import Path

from docvlm_eval.comparison import build_tables, robustness_retention


def _write_run(root: Path, model: str, bench: str, score: float, ece=None, per_sample=None):
    d = root / model / bench
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "model": model, "benchmark": bench, "param_count_m": 900,
        "primary_metric": "anls", "score": score, "accuracy": score,
        "ece": ece, "answer_rate": 1.0,
    }))
    if per_sample is not None:
        (d / "per_sample.json").write_text(json.dumps(per_sample))


def test_build_tables_outputs(tmp_path):
    res = tmp_path / "results"
    _write_run(res, "m1", "docvqa", 0.8, ece=0.1)
    _write_run(res, "m2", "docvqa", 0.6, ece=0.2)
    md = build_tables(res, res)
    assert "m1" in md and "m2" in md and "docvqa" in md
    assert (res / "comparison_table.md").exists()
    assert (res / "comparison_table.csv").exists()
    assert (res / "comparison_table.json").exists()


def test_robustness_retention(tmp_path):
    res = tmp_path / "results"
    per = [
        {"perturbation": "clean", "score": 1.0},
        {"perturbation": "clean", "score": 1.0},
        {"perturbation": "blur", "score": 0.5},
        {"perturbation": "jpeg", "score": 0.8},
    ]
    _write_run(res, "m1", "robustness", 0.7, per_sample=per)
    ret = robustness_retention(res)
    assert ret["m1"]["blur"] == 0.5
    assert ret["m1"]["jpeg"] == 0.8


def test_no_summaries_raises(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        build_tables(tmp_path, tmp_path)

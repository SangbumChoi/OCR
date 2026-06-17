"""CLI entry points: argument handling + in-process invocation."""

import json

import pytest

from docvlm_eval import cli


def test_list_models(capsys):
    cli.evaluate(["--list-models"])
    out = capsys.readouterr().out
    assert "dummy-echo" in out
    assert "internvl3-1b" in out


def test_evaluate_requires_args():
    with pytest.raises(SystemExit):
        cli.evaluate([])  # no --model/--benchmark/--out


def test_evaluate_runs(tiny_benchmark, tmp_path, capsys):
    _, path = tiny_benchmark
    out = tmp_path / "o"
    cli.evaluate(["--model", "dummy-echo", "--benchmark", path,
                  "--benchmark-name", "t", "--out", str(out), "--device", "cpu"])
    printed = capsys.readouterr().out
    summary = json.loads(printed[printed.index("{"):])
    assert summary["n_samples"] == 3
    assert (out / "summary.json").exists()


def test_robustness_cli(tiny_benchmark, tmp_path):
    _, path = tiny_benchmark
    out = tmp_path / "rob"
    cli.build_robustness(["--base", path, "--out-dir", str(out), "--limit", "1",
                          "--perturbations", "blur", "term_paraphrase"])
    assert (out / "robustness.jsonl").exists()

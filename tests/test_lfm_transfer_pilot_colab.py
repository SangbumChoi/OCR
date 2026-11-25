from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_colab_lfm_pilot_launcher_dry_run_is_compact(tmp_path):
    log = tmp_path / "pilot.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_lfm_transfer_pilot_colab.py"),
            "--dry-run",
            "--poll-seconds",
            "0.05",
            "--heartbeat-seconds",
            "60",
            "--log",
            str(log),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    assert lines[0]["readiness"] == "pass"
    assert lines[0]["checks"] == {"pass": 15, "fail": 0}
    assert lines[1]["public_dataset_readiness"]["status"] == "pass"
    assert lines[1]["public_dataset_readiness"][
        "pilot_selection_feasible"
    ] is True
    assert lines[-1]["status"] == "completed"
    assert lines[-1]["dry_run"] is True
    assert log.is_file()
    assert log.stat().st_size < 100_000


def test_colab_smol_pilot_launcher_dry_run_is_compact(tmp_path):
    log = tmp_path / "smol-pilot.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_transfer_pilot_colab.py"),
            "--pilot",
            "smol-vision",
            "--dry-run",
            "--poll-seconds",
            "0.05",
            "--heartbeat-seconds",
            "60",
            "--log",
            str(log),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    assert lines[0]["readiness"] == "pass"
    assert lines[0]["checks"] == {"pass": 14, "fail": 0}
    assert lines[1]["public_dataset_readiness"]["status"] == "pass"
    assert lines[1]["public_dataset_readiness"][
        "pilot_selection_feasible"
    ] is True
    assert lines[-1]["status"] == "completed"
    assert lines[-1]["pilot"] == "smol-vision"
    assert lines[-1]["dry_run"] is True
    assert log.is_file()
    assert log.stat().st_size < 100_000


def test_colab_smol_confirmatory_dry_run_reports_pending_gate(tmp_path):
    log = tmp_path / "smol-confirmatory.log"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_smol_confirmatory_colab.py"),
            "--dry-run",
            "--poll-seconds",
            "0.05",
            "--heartbeat-seconds",
            "60",
            "--log",
            str(log),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{") and line.endswith("}")
    ]
    assert lines[0]["readiness"] == "pending"
    assert lines[0]["checks"] == {"pass": 2, "pending": 5, "fail": 0}
    assert lines[1]["public_dataset_readiness"]["status"] == "pass"
    assert lines[-1]["status"] == "completed"
    assert lines[-1]["pilot"] == "smol-confirmatory"
    assert lines[-1]["dry_run"] is True
    assert log.is_file()
    assert log.stat().st_size < 100_000


def test_smol_confirmatory_live_launch_fails_closed_while_pending():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_smol_confirmatory_colab.py"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "submission readiness audit did not pass" in completed.stderr


def test_colab_summary_does_not_repeat_full_promotion_evidence():
    compact = runpy.run_path(
        str(ROOT / "scripts" / "run_lfm_transfer_pilot_colab.py")
    )["_compact_summary"]
    summary = compact(
        {
            "status": "completed",
            "variants": [{"run": "candidate--seed_0", "status": "completed"}],
            "promotion": {
                "status": "promote",
                "selected_variants": ["candidate"],
                "multiple_comparisons": {
                    "method": "bonferroni",
                    "comparison_count": 8,
                    "familywise_alpha": 0.05,
                    "resamples": 10_000,
                },
                "candidates": {
                    "candidate": {
                        "large_metric_table": list(range(10_000)),
                    }
                },
            },
        }
    )

    assert summary["promotion"] == {
        "status": "promote",
        "selected_variants": ["candidate"],
        "multiple_comparisons": {
            "method": "bonferroni",
            "comparison_count": 8,
            "familywise_alpha": 0.05,
        },
    }
    assert len(json.dumps(summary)) < 1_000

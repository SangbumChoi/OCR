from __future__ import annotations

import json
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
    assert lines[-1]["status"] == "completed"
    assert lines[-1]["dry_run"] is True
    assert log.is_file()
    assert log.stat().st_size < 100_000

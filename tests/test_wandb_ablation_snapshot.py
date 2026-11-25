import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_wandb_ablation_snapshot.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_wandb_ablation_snapshot",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
analyze_snapshot = MODULE.analyze_snapshot


def _snapshot():
    return json.loads(
        (ROOT / "docs" / "results" / "lfm_ablation_wandb_snapshot.json").read_text(encoding="utf-8")
    )


def test_live_snapshot_excludes_incomplete_runs_and_limits_claim():
    result = analyze_snapshot(_snapshot())

    assert result["evidence_status"] == "preliminary_direction_only"
    assert result["promotion_eligible"] is False
    assert result["source_snapshot_sha256"].startswith("sha256:")
    assert len(result["source_snapshot_sha256"]) == 71
    assert result["run_quality"]["observed_runs"] == 10
    assert result["run_quality"]["evaluated_runs"] == 5
    assert len(result["run_quality"]["finished_without_evaluation"]) == 4
    assert result["run_quality"]["crashed_runs"] == ["r0t65g1h"]
    assert result["run_quality"]["crash_diagnostics"]["r0t65g1h"]["planned_micro_steps"] == 7728
    pair = result["comparable_preliminary_pair"]
    assert pair["heldout_vision_minus_connector"]["score"] == pytest.approx(0.0486)
    assert pair["heldout_vision_minus_connector"]["grounding"] == pytest.approx(0.0607)
    assert pair["heldout_vision_minus_connector"]["L1-locate"] == pytest.approx(0.0234)
    assert pair["heldout_vision_minus_connector"]["kie"] == pytest.approx(0)


def test_snapshot_rejects_a_confounded_preliminary_pair():
    snapshot = deepcopy(_snapshot())
    connector = next(run for run in snapshot["runs"] if run["id"] == "zivt0ner")
    connector["learning_rate"] = 0.0002

    with pytest.raises(ValueError, match="controls differ"):
        analyze_snapshot(snapshot)


def test_snapshot_rejects_out_of_range_metrics():
    snapshot = deepcopy(_snapshot())
    vision = next(run for run in snapshot["runs"] if run["id"] == "prh5gy29")
    vision["heldout"]["grounding"] = 1.01

    with pytest.raises(ValueError, match="out-of-range"):
        analyze_snapshot(snapshot)

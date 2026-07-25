from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "smol_vision_transfer_confirmatory.ipynb"


def test_smol_confirmatory_notebook_is_gated_and_compact():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source") or [])
        for cell in notebook.get("cells") or []
    )

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert "snapshot_wandb_run_inventory.py" in source
    assert "audit_smol_vision_transfer_pilot_execution.py" in source
    assert "audit_smol_confirmatory_submission.py" in source
    assert "confirmatory_submission_authorized" in source
    assert "run_smol_confirmatory_colab.py" in source
    assert "build_smol_confirmatory_evidence.py" in source
    assert "audit_end_to_end_goal_readiness.py" in source
    assert "per-sample outputs or long tables" in source
    assert all(not cell.get("outputs") for cell in notebook["cells"])

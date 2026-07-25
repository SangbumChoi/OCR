from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "smol_vision_transfer_pilot.ipynb"


def test_smol_vision_transfer_notebook_is_compact_and_pinned():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    source = "\n".join(
        "".join(cell.get("source") or []) for cell in cells
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["colab"]["gpuType"] == "L4"
    assert len(cells) == 6
    assert all(not cell.get("outputs") for cell in cells)
    assert "claude/new-session-w79q0i" in source
    assert "run_transfer_pilot_colab.py" in source
    assert "'smol-vision'" in source
    assert "docvlm-smol-vision-transfer-pilot" in source
    assert "WANDB_API_KEY" in source
    assert "lfm_language_only" not in source
    assert "lfm_smol_dual" not in source
    assert NOTEBOOK.stat().st_size < 10_000

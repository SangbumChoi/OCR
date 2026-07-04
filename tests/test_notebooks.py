"""E2E tests for every notebook — each at the depth that is appropriate for it.

Two tiers:

* **structural (always on)** — every notebook must be valid nbformat, every code cell must
  compile (after IPython magic/shell transformation), and no cell may carry a stored error
  output. This catches syntax rot and committed failing runs in the Colab/GPU notebooks
  (full comparison, fine-grained comparison, fine-tune ablations, flash-attention, latency
  profile) that cannot train on a CPU CI box.
* **execution (opt-in: ``RUN_NB_E2E=1``)** — the CPU-runnable notebooks are executed end to
  end with a real kernel: ``udd_ablation.ipynb`` (needs the merged corpus at
  ``data/udd/hf/_all``) and ``synthetic_data_design.ipynb``. Slow (minutes each), hence
  env-gated; run locally or in a scheduled job, not on every push.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
# executable on CPU end-to-end; the rest are Colab/GPU-targeted and tested structurally
CPU_E2E = ("synthetic_data_design.ipynb", "udd_ablation.ipynb")


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_compile(nb):
    from IPython.core.inputtransformer2 import TransformerManager
    tm = TransformerManager()
    for i, c in enumerate(_cells(nb)):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            ast.parse(tm.transform_cell(src))
        except SyntaxError as e:  # pragma: no cover - failure path
            pytest.fail(f"{nb.name} cell {i} does not compile: {e}")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_no_stored_error_outputs(nb):
    errs = [(i, o.get("ename")) for i, c in enumerate(_cells(nb))
            for o in (c.get("outputs") or []) if o.get("output_type") == "error"]
    assert not errs, f"{nb.name} carries stored error outputs from a failed run: {errs}"


@pytest.mark.parametrize("name", CPU_E2E)
def test_executes_end_to_end(name):
    if not os.environ.get("RUN_NB_E2E"):
        pytest.skip("set RUN_NB_E2E=1 to execute the CPU-runnable notebooks (slow)")
    if name == "udd_ablation.ipynb" and not (ROOT / "data/udd/hf/_all").exists():
        pytest.skip("merged UDD corpus not built (scripts/build_udd.py)")
    import nbformat
    from nbclient import NotebookClient
    doc = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    NotebookClient(doc, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()

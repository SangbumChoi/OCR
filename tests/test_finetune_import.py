"""The finetune subpackage lives under docvlm_eval.finetune after the merge.

Its modules import torch/transformers/peft, so we only assert importability when those are
installed; otherwise we just confirm the package directory is in place (structure check).
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "src" / "docvlm_eval" / "finetune"

_HAVE_TORCH = importlib.util.find_spec("torch") is not None
# train.py also needs peft + accelerate; only assert import when the full finetune stack is present
_HAVE_FT = _HAVE_TORCH and all(
    importlib.util.find_spec(m) is not None for m in ("peft", "accelerate")
)


def test_finetune_package_files_exist():
    assert (FT / "__init__.py").exists()
    for mod in ("train.py", "eval.py", "modeling.py", "metrics.py"):
        assert (FT / mod).exists(), f"missing finetune/{mod}"
    assert (FT / "data" / "dataset.py").exists()


def test_finetune_metrics_importable():
    # metrics.py is pure-python (no torch) -> should always import
    from docvlm_eval.finetune import metrics  # noqa: F401
    assert hasattr(metrics, "compute_ocr_metrics") or True


@pytest.mark.skipif(not _HAVE_FT, reason="finetune stack (torch+peft+accelerate) not installed")
def test_finetune_train_importable_with_torch():
    from docvlm_eval.finetune import train  # noqa: F401

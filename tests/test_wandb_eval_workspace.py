import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "configure_wandb_eval_workspace.py"
_SPEC = importlib.util.spec_from_file_location("configure_wandb_eval_workspace", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
collect_eval_metric_pairs = _MODULE.collect_eval_metric_pairs


def test_collect_eval_metric_pairs_matches_common_axes() -> None:
    pairs = collect_eval_metric_pairs(
        [
            "eval/train_score",
            "eval/heldout_score",
            "eval/train_grounding",
            "eval/heldout_grounding",
            "eval/train_only",
            "eval/heldout_other",
            "train/loss",
        ]
    )

    assert pairs == {
        "grounding": ("eval/train_grounding", "eval/heldout_grounding"),
        "score": ("eval/train_score", "eval/heldout_score"),
    }


def test_collect_eval_metric_pairs_accepts_held_alias_and_prefers_heldout() -> None:
    pairs = collect_eval_metric_pairs(
        [
            "eval/train_kie",
            "eval/held_kie",
            "eval/heldout_kie",
        ]
    )

    assert pairs == {"kie": ("eval/train_kie", "eval/heldout_kie")}

"""Robustness probe: paraphrase rewriting + paired clean/perturbed generation."""

from docvlm_eval.benchmarks.robustness import VISUAL, _paraphrase, build_robustness_set
from docvlm_eval.schema import Sample


def test_paraphrase_rewrites_terms():
    assert _paraphrase("How much is the total?") != "How much is the total?"
    assert "aggregate" in _paraphrase("What is the total?").lower()


def test_paraphrase_idempotent_on_neutral_text():
    q = "Identify the colour shown"
    assert _paraphrase(q) == q  # no trigger terms -> unchanged


def test_build_robustness_set_pairs(tmp_path, tiny_image):
    base = [Sample("s0", tiny_image, "What is the total?", ["100"], "form", "anls")]
    perts = VISUAL + ["term_paraphrase"]
    out = build_robustness_set(base, tmp_path, perturbations=perts)
    # 1 clean + len(perts) variants
    assert len(out) == 1 + len(perts)
    kinds = {s.meta["perturbation"] for s in out}
    assert "clean" in kinds
    assert set(perts).issubset(kinds)
    # every variant carries base_id + answers + valid image path
    for s in out:
        assert s.meta["base_id"] == "s0"
        assert s.answers == ["100"]
        from pathlib import Path
        assert Path(s.image_path).exists()


def test_term_paraphrase_keeps_image_changes_question(tmp_path, tiny_image):
    base = [Sample("s0", tiny_image, "What is the total?", ["100"], "form", "anls")]
    out = build_robustness_set(base, tmp_path, perturbations=["term_paraphrase"])
    para = [s for s in out if s.meta["perturbation"] == "term_paraphrase"][0]
    assert para.question != "What is the total?"

"""The **layer-value ablation**: which annotation layer actually helps the END task?

The end task of document understanding is getting the right *value* out (KIE + QA), not OCR or
layout for their own sake. Every arm below trains on the SAME annotated images and always includes
the end-task targets; arms differ only in which *auxiliary* layers are also emitted as targets
(and in the QA target style). So a Δ on the held-out end task is attributable to that layer.

Two families, both needed:
  * **additive**  — ``target`` + ONE layer family: "does this layer help on its own?"
  * **leave-one-out** — ``all`` − ONE family: "is this layer still needed once the rest is there?"
A layer that wins additively but not leave-one-out is *redundant* with another (e.g. OCR vs
line grounding); one that wins in both is load-bearing.

Hold the **training steps** fixed across arms (``run_ablation.py --steps``): more layers mean more
samples per image, so a fixed step budget makes the auxiliary layers *compete* with the target for
updates instead of adding free compute. Combine Δ with the per-layer annotation time recorded in
provenance (:func:`annotation_cost`) to get **Δ per annotation-hour** — the number an individual
annotator needs to decide what to label next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..schema import Sample
from .export import export_samples
from .geometry import materialize_rotations
from .schema import LAYERS, DocAnnotation

# the end task: always present in every arm
TARGET_LAYERS = ("kie", "qa")

# layer FAMILY -> export layers it switches on (+ the QA style it implies, if any)
FAMILIES: dict[str, dict] = {
    "orientation": {"layers": ("orientation", "doctype"), "rotate": True},
    "layout": {"layers": ("layout",)},
    "ocr": {"layers": ("ocr",)},
    "line_grounding": {"layers": ("line_grounding",)},
    "table": {"layers": ("table",)},
    "spotting": {"layers": ("kie_spotting",)},
    "reasoning": {"reasoning": "chain"},
    # caption + reading-guide targets; when "reasoning" is also on, chains open with the guide
    # notes in scope (guided_chain) — so each family stays ONE factor in additive and LOO arms
    "knowledge": {"layers": ("caption", "guide")},
}

# family -> provenance layer whose human seconds it costs
FAMILY_COST_LAYER = {"orientation": "page", "layout": "layout", "ocr": "ocr",
                     "line_grounding": "ocr", "table": "table", "spotting": "kie",
                     "reasoning": "understanding", "knowledge": "understanding"}


@dataclass(frozen=True)
class LayerArm:
    name: str
    families: tuple[str, ...]
    description: str = ""
    layers: tuple[str, ...] = field(default=())
    reasoning: str = "answer"
    rotate: bool = False

    @classmethod
    def of(cls, name: str, families: tuple[str, ...], description: str = "") -> LayerArm:
        layers = list(TARGET_LAYERS)
        for fam in families:
            layers += [x for x in FAMILIES[fam].get("layers", ()) if x not in layers]
        rotate = any(FAMILIES[f].get("rotate", False) for f in families)
        reasoning = "answer"
        if "reasoning" in families:
            reasoning = "guided_chain" if "knowledge" in families else "chain"
        return cls(name, tuple(families), description, tuple(layers), reasoning, rotate)


def _arms() -> dict[str, LayerArm]:
    arms = {"L0_target": LayerArm.of("L0_target", (), "end task only: KIE + answer-only QA")}
    for fam in FAMILIES:
        n = f"L1_+{fam}"
        arms[n] = LayerArm.of(n, (fam,), f"end task + {fam}")
    # the interaction the knowledge layer exists for: chains that open with the reading guide,
    # read against L1_+reasoning (same chains, no guide) -> "does interpretation knowledge help?"
    arms["L1_+reasoning+knowledge"] = LayerArm.of(
        "L1_+reasoning+knowledge", ("reasoning", "knowledge"), "end task + guided reasoning chains")
    every = tuple(FAMILIES)
    arms["L2_all"] = LayerArm.of("L2_all", every, "every layer")
    for fam in FAMILIES:
        n = f"L2_all-{fam}"
        arms[n] = LayerArm.of(n, tuple(f for f in every if f != fam), f"every layer except {fam}")
    return arms


ARMS: dict[str, LayerArm] = _arms()


def build_arm(anns: list[DocAnnotation], arm: LayerArm | str, *, base_dir: str | Path | None = None,
              rot_dir: str | Path | None = None, split: str = "train",
              include_pseudo: bool = True) -> list[Sample]:
    """Training samples for one arm over the ``split`` records.

    With the orientation family on, rotated copies of each image are written to ``rot_dir`` and
    contribute ONLY the orientation question (their other labels would duplicate the upright ones)."""
    arm = ARMS[arm] if isinstance(arm, str) else arm
    out: list[Sample] = []
    for ann in anns:
        if ann.split != split:
            continue
        out += export_samples(ann, arm.layers, reasoning=arm.reasoning, base_dir=base_dir,
                              include_pseudo=include_pseudo)
        if arm.rotate and rot_dir is not None:
            img = export_samples(ann, ("orientation",), base_dir=base_dir)
            if not img:
                continue
            for rann, _ in materialize_rotations(ann, img[0].image_path, str(rot_dir)):
                out += export_samples(rann, ("orientation",))
    return out


def eval_samples(anns: list[DocAnnotation], arm: LayerArm | str = "L0_target", *,
                 base_dir: str | Path | None = None) -> list[Sample]:
    """The held-out END-TASK set, prompted in the arm's own QA style (so a chain-trained arm is
    asked for a chain). Scored by ``kie_f1`` / ``anls`` / ``final_answer``, which read the final
    answer either way — the comparison across arms stays on the same gold values."""
    arm = ARMS[arm] if isinstance(arm, str) else arm
    out = []
    for ann in anns:
        if ann.split == "heldout":
            out += export_samples(ann, TARGET_LAYERS, reasoning=arm.reasoning, base_dir=base_dir)
    return out


def annotation_cost(anns: list[DocAnnotation]) -> dict[str, float]:
    """Total human HOURS per provenance layer across the corpus."""
    tot = {name: 0.0 for name in LAYERS}
    for a in anns:
        for name, lp in a.provenance.layers.items():
            tot[name] = tot.get(name, 0.0) + (lp.seconds or 0.0) / 3600.0
    return tot


def value_per_hour(scores: dict[str, float], anns: list[DocAnnotation]) -> dict[str, dict]:
    """For each additive arm ``L1_+<family>``: Δ vs ``L0_target`` and Δ per annotation hour.

    ``scores`` maps arm name -> held-out end-task score (e.g. mean of kie_f1 and final_answer)."""
    cost = annotation_cost(anns)
    base = scores.get("L0_target")
    out = {}
    for fam in FAMILIES:
        name = f"L1_+{fam}"
        if base is None or name not in scores:
            continue
        hours = cost.get(FAMILY_COST_LAYER[fam], 0.0)
        delta = scores[name] - base
        out[fam] = {"delta": delta, "hours": hours,
                    "delta_per_hour": (delta / hours) if hours > 0 else None}
    return out

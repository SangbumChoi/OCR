"""Custom robustness probe.

The task asks us to go *beyond generic VQA accuracy* and probe properties that matter for
real document understanding. Headline ANLS is measured on clean, high-resolution scans -
but production documents are phone photos, faxes, and re-compressed PDFs, and users ask
questions with domain jargon. This module builds a **paired** probe so we can attribute a
model's failures to *capture quality* and *terminology*, not just "it got it wrong".

For each base sample we emit:
  * a ``clean`` copy (baseline), and
  * one copy per perturbation.

Robustness is then reported as **retention** = score(perturbed) / score(clean) per
perturbation family, so a model that is accurate *and* stable scores high, while a model
that is accurate only on pristine inputs is exposed.

Perturbation families (each justified by a real document-capture failure mode):
  * ``downscale``  - low-DPI scan / small phone photo (small-text legibility)
  * ``jpeg``       - heavy re-compression artifacts (fax / messaging apps)
  * ``blur``       - out-of-focus capture
  * ``rotate``     - skewed page on a flatbed / handheld tilt
  * ``noise``      - sensor / photocopier speckle
  * ``term_paraphrase`` - question rewritten with domain-style phrasing (terminology
                          robustness; image untouched)
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import Sample

# Lightweight, rule-based question rephrasings that swap everyday phrasing for the kind of
# terse domain wording seen in finance/forms. No model needed -> deterministic & reproducible.
_TERM_SUBS = [
    (r"\bhow much\b", "what is the amount"),
    (r"\btotal\b", "aggregate"),
    (r"\bdate\b", "date of issue"),
    (r"\bcompany\b", "entity"),
    (r"\bphone number\b", "contact no."),
    (r"\bwhat is the\b", "state the"),
    (r"\bwho\b", "which party"),
    (r"\bcost\b", "amount payable"),
]


def _paraphrase(question: str) -> str:
    q = question
    for pat, rep in _TERM_SUBS:
        q = re.sub(pat, rep, q, flags=re.IGNORECASE)
    return q


def _perturb_image(img, kind: str):
    from PIL import Image, ImageFilter

    if kind == "downscale":
        w, h = img.size
        small = img.resize((max(1, w // 3), max(1, h // 3)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2.0))
    if kind == "rotate":
        return img.rotate(5, expand=False, fillcolor=(255, 255, 255))
    if kind == "noise":
        import numpy as np

        arr = np.asarray(img.convert("RGB")).astype("int16")
        noise = np.random.default_rng(0).normal(0, 18, arr.shape)
        arr = (arr + noise).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
    raise ValueError(f"unknown perturbation {kind}")


VISUAL = ["downscale", "jpeg", "blur", "rotate", "noise"]


def build_robustness_set(
    base_samples: list[Sample],
    out_dir: str,
    perturbations: list[str] | None = None,
    jpeg_quality: int = 18,
    seed: int = 0,
) -> list[Sample]:
    """Materialise a paired clean/perturbed probe from ``base_samples``.

    Images are written under ``<out_dir>/images``. Returns the expanded sample list
    (clean + one per perturbation), each tagged with ``meta['perturbation']`` and
    ``meta['base_id']`` so the comparison tooling can compute retention.
    """
    from PIL import Image

    perturbations = perturbations or (VISUAL + ["term_paraphrase"])
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    expanded: list[Sample] = []
    for s in base_samples:
        base_img = Image.open(s.image_path).convert("RGB")

        # clean baseline
        clean_path = img_dir / f"{s.sample_id}__clean.png"
        base_img.save(clean_path)
        expanded.append(
            _variant(s, str(clean_path), s.question, "clean")
        )

        for kind in perturbations:
            if kind == "term_paraphrase":
                expanded.append(
                    _variant(s, str(clean_path), _paraphrase(s.question), kind)
                )
                continue
            p = img_dir / f"{s.sample_id}__{kind}.{'jpg' if kind == 'jpeg' else 'png'}"
            if kind == "jpeg":
                base_img.save(p, format="JPEG", quality=jpeg_quality)
            else:
                _perturb_image(base_img, kind).save(p)
            expanded.append(_variant(s, str(p), s.question, kind))
    return expanded


def _variant(s: Sample, image_path: str, question: str, pert: str) -> Sample:
    return Sample(
        sample_id=f"{s.sample_id}__{pert}",
        image_path=image_path,
        question=question,
        answers=s.answers,
        answer_type=pert,  # slice analysis is by perturbation here
        metric=s.metric,
        meta={**s.meta, "perturbation": pert, "base_id": s.sample_id},
    )

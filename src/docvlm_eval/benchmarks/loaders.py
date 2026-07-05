"""Read/write the normalised benchmark JSONL.

JSONL schema (one object per line) mirrors :class:`~docvlm_eval.schema.Sample`::

    {"sample_id": "docvqa_val_0", "image_path": "data/docvqa/img/0.png",
     "question": "What is the total?", "answers": ["$1,200"],
     "answer_type": "form/total", "metric": "anls", "meta": {}}
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Sample

ROOT = Path(__file__).resolve().parents[3]     # src/docvlm_eval/benchmarks/ -> repo root


def _portable_path(p: str) -> str:
    """Re-anchor a fixture image path at THIS repo root when it does not exist here.

    Checked-in jsonls (probe suites) carry absolute paths from the machine that generated them
    (e.g. ``/home/user/OCR/data/...``); on a clone rooted elsewhere (Colab: ``/content/OCR``)
    those break. The ``data/...`` tail is machine-independent, so resolve it against this repo."""
    if not p or Path(p).exists():
        return p
    q = Path(p)
    if not q.is_absolute():
        cand = ROOT / q
        return str(cand) if cand.exists() else p
    parts = q.parts
    if "data" in parts:
        cand = ROOT.joinpath(*parts[parts.index("data"):])
        if cand.exists():
            return str(cand)
    return p


def load_jsonl(path: str | Path) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(
                Sample(
                    sample_id=str(d["sample_id"]),
                    image_path=_portable_path(d["image_path"]),
                    question=d["question"],
                    answers=[str(a) for a in d["answers"]],
                    answer_type=d.get("answer_type", "default"),
                    metric=d.get("metric", "anls"),
                    meta=d.get("meta", {}),
                )
            )
    return samples


def save_jsonl(samples: list[Sample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "sample_id": s.sample_id,
                        "image_path": s.image_path,
                        "question": s.question,
                        "answers": s.answers,
                        "answer_type": s.answer_type,
                        "metric": s.metric,
                        "meta": s.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

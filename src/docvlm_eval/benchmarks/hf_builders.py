"""Build normalised benchmark JSONL from public HuggingFace datasets.

Each builder downloads a dataset split, writes images to ``<out_dir>/images`` and emits a
JSONL of :class:`~docvlm_eval.schema.Sample`. We keep these as small, explicit functions
(rather than one mega-loader) so design decisions - which split, which answer field, how we
tag ``answer_type`` for slice analysis - are auditable.

These run wherever ``datasets`` + network are available (e.g. Colab). All accept a
``limit`` so you can build a fast smoke subset.

Dataset choices & rationale (see the report for the full argument):
* DocVQA  - lmms-lab/DocVQA, config "DocVQA", split "validation". Official ANLS, has
            gold answers on val (test is eval-server only).
* InfoVQA - lmms-lab/DocVQA, config "InfographicVQA", split "validation". Layout-heavy
            infographics; complements DocVQA's scanned business docs.
* ChartQA - lmms-lab/ChartQA, split "test". Carries a human/augmented split we keep as
            ``answer_type`` (human questions are the harder, reasoning-heavy slice).
* OCRBench- echo840/OCRBench. Carries a fine-grained category we keep as ``answer_type``
            (regular text, handwriting, artistic, KIE, handwritten math, ...).
"""

from __future__ import annotations

from pathlib import Path

from ..schema import Sample
from .loaders import save_jsonl


def _dump_image(image, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.png"
    if not p.exists():
        image.convert("RGB").save(p)
    return str(p)


def build_docvqa(out_dir: str, split: str = "validation", limit: int | None = None) -> str:
    from datasets import load_dataset

    out = Path(out_dir)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for i, ex in enumerate(ds):
        sid = f"docvqa_{split}_{ex.get('questionId', i)}"
        img_path = _dump_image(ex["image"], out / "images", sid)
        samples.append(
            Sample(
                sample_id=sid,
                image_path=img_path,
                question=ex["question"],
                answers=list(ex["answers"]),
                answer_type=ex.get("question_types", ["default"])[0]
                if ex.get("question_types")
                else "default",
                metric="anls",
                meta={"benchmark": "docvqa"},
            )
        )
    path = out / "docvqa.jsonl"
    save_jsonl(samples, path)
    return str(path)


def build_infovqa(out_dir: str, split: str = "validation", limit: int | None = None) -> str:
    from datasets import load_dataset

    out = Path(out_dir)
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for i, ex in enumerate(ds):
        sid = f"infovqa_{split}_{ex.get('questionId', i)}"
        img_path = _dump_image(ex["image"], out / "images", sid)
        samples.append(
            Sample(
                sample_id=sid,
                image_path=img_path,
                question=ex["question"],
                answers=list(ex["answers"]),
                answer_type="infographic",
                metric="anls",
                meta={"benchmark": "infovqa"},
            )
        )
    path = out / "infovqa.jsonl"
    save_jsonl(samples, path)
    return str(path)


def build_chartqa(out_dir: str, split: str = "test", limit: int | None = None) -> str:
    from datasets import load_dataset

    out = Path(out_dir)
    ds = load_dataset("lmms-lab/ChartQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for i, ex in enumerate(ds):
        sid = f"chartqa_{split}_{i}"
        img_path = _dump_image(ex["image"], out / "images", sid)
        ans = ex["answer"]
        answers = ans if isinstance(ans, list) else [ans]
        samples.append(
            Sample(
                sample_id=sid,
                image_path=img_path,
                question=ex["question"],
                answers=[str(a) for a in answers],
                answer_type=ex.get("type", "augmented"),  # human vs augmented slice
                metric="relaxed_acc",
                meta={"benchmark": "chartqa"},
            )
        )
    path = out / "chartqa.jsonl"
    save_jsonl(samples, path)
    return str(path)


def build_ocrbench(out_dir: str, limit: int | None = None) -> str:
    from datasets import load_dataset

    out = Path(out_dir)
    ds = load_dataset("echo840/OCRBench", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    samples = []
    for i, ex in enumerate(ds):
        sid = f"ocrbench_{i}"
        img_path = _dump_image(ex["image"], out / "images", sid)
        ans = ex.get("answer")
        answers = ans if isinstance(ans, list) else [ans]
        samples.append(
            Sample(
                sample_id=sid,
                image_path=img_path,
                question=ex.get("question", "What is written in the image?"),
                answers=[str(a) for a in answers],
                answer_type=ex.get("question_type", ex.get("dataset", "ocr")),
                metric="ocrbench",
                meta={"benchmark": "ocrbench"},
            )
        )
    path = out / "ocrbench.jsonl"
    save_jsonl(samples, path)
    return str(path)


BUILDERS = {
    "docvqa": build_docvqa,
    "infovqa": build_infovqa,
    "chartqa": build_chartqa,
    "ocrbench": build_ocrbench,
}

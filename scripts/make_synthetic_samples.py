#!/usr/bin/env python3
"""Generate illustrative samples for taxonomy categories that are not cleanly fetchable from
HuggingFace (full-page text recognition, scene text) plus a reliability/robustness example.

These are *synthetic / derived* and clearly labelled as such in each ``sample.json`` (they
render the TASK so the suite is complete and inspectable; they are NOT the official benchmark
data). Deterministic (fixed seed) -> reproducible.

    python scripts/make_synthetic_samples.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


def _save(key: str, img: Image.Image, label: dict) -> None:
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(folder / "sample.png")
    (folder / "sample.json").write_text(
        json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok] {key} -> {folder}")


def make_fullpage_recognition() -> None:
    """Category 1: full-page printed-text recognition. GT = the exact transcript."""
    lines = [
        "QUARTERLY FINANCIAL SUMMARY",
        "",
        "Revenue for Q3 2025 reached $4,820,000, a 12.4% increase",
        "year over year. Operating expenses totalled $3,110,500,",
        "leaving an operating margin of 35.5%. The board approved",
        "a dividend of $0.28 per share, payable on 15 December 2025.",
        "",
        "Contact: investor.relations@example.com   Ref: FIN-2025-Q3-007",
    ]
    img = Image.new("RGB", (1180, 520), "white")
    d = ImageDraw.Draw(img)
    y = 40
    for i, ln in enumerate(lines):
        f = _font(30, bold=(i == 0))
        d.text((50, y), ln, fill="black", font=f)
        y += 56 if ln else 28
    _save(
        "recognition_fullpage",
        img,
        {
            "benchmark": "recognition_fullpage",
            "category": "1. Full-page / printed text recognition",
            "metric": "CER / WER / NED",
            "metric_note": "Character/word error rate or normalized edit distance vs transcript.",
            "source": "SYNTHETIC illustrative sample (rendered locally; not an official benchmark)",
            "ground_truth": {"text": "\n".join([l for l in lines if l])},
        },
    )


def make_scenetext() -> None:
    """Category 2: scene-text detection & recognition. GT = word list (+ rough boxes)."""
    img = Image.new("RGB", (800, 450), (38, 70, 110))
    d = ImageDraw.Draw(img)
    # a faux storefront sign with a couple of words at angles
    d.rectangle([60, 60, 740, 200], fill=(245, 222, 120))
    d.text((90, 95), "CAFÉ AURORA", fill=(40, 40, 40), font=_font(70, bold=True))
    d.rectangle([120, 250, 520, 340], fill=(200, 60, 60))
    d.text((140, 262), "OPEN 24/7", fill="white", font=_font(58, bold=True))
    d.text((560, 270), "No. 42", fill=(230, 230, 230), font=_font(44, bold=True))
    words = ["CAFÉ", "AURORA", "OPEN", "24/7", "No.", "42"]
    _save(
        "scenetext",
        img,
        {
            "benchmark": "scenetext",
            "category": "2. Scene-text detection & recognition",
            "metric": "detection H-mean / word accuracy / 1-NED",
            "metric_note": "F-measure on detected boxes; case-insensitive word accuracy for recognition.",
            "source": "SYNTHETIC illustrative sample (rendered locally; not ICDAR/Total-Text)",
            "ground_truth": {"words": words},
        },
    )


def make_robustness() -> None:
    """Category 10: reliability/robustness. Derive a degraded copy of the DocVQA sample."""
    base_path = OUT / "docvqa" / "sample.png"
    base_label = OUT / "docvqa" / "sample.json"
    if not base_path.exists():
        print("[skip] robustness: run fetch_benchmark_samples.py for docvqa first")
        return
    img = Image.open(base_path).convert("RGB")
    w, h = img.size
    # simulate a poor phone photo: downscale -> blur -> heavy JPEG
    deg = img.resize((w // 3, h // 3)).resize((w, h)).filter(ImageFilter.GaussianBlur(1.6))
    folder = OUT / "robustness"
    folder.mkdir(parents=True, exist_ok=True)
    deg.save(folder / "sample.png", format="PNG")
    deg.save(folder / "sample_degraded.jpg", format="JPEG", quality=18)
    gt = json.loads(base_label.read_text())["ground_truth"] if base_label.exists() else {}
    _save(
        "robustness",
        deg,
        {
            "benchmark": "robustness",
            "category": "10. Reliability / robustness / calibration",
            "metric": "ANLS retention + ECE",
            "metric_note": "score(degraded)/score(clean) per perturbation; ECE for confidence calibration.",
            "source": "DERIVED from data/benchmarks/docvqa (downscale+blur+JPEG q18); see scripts/build_robustness_set.py",
            "ground_truth": {
                "question": gt.get("question"),
                "answers": gt.get("answers"),
                "perturbations": ["downscale", "blur", "jpeg"],
            },
        },
    )


def main() -> None:
    make_fullpage_recognition()
    make_scenetext()
    make_robustness()
    print("\n[done] synthetic/derived samples written under data/benchmarks/")


if __name__ == "__main__":
    main()

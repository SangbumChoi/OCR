#!/usr/bin/env python3
"""Download ONE representative sample (image + ground-truth label + metric note) for each
document-understanding benchmark in the taxonomy, so the suite is inspectable at a glance.

For every benchmark it writes:
    data/benchmarks/<key>/sample.png     # the image
    data/benchmarks/<key>/sample.json    # GT label + metric + source ("what & how it's scored")

Uses HF `datasets` streaming, so it pulls a single example without downloading the full set.
Network-dependent datasets that fail (gated/moved) are skipped with a warning; re-run later.

    python scripts/fetch_benchmark_samples.py
    python scripts/fetch_benchmark_samples.py --only docvqa chartqa
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks"

# (key, hf_repo, config, split, category, metric, metric_note, source_url)
REGISTRY = [
    ("docvqa", "lmms-lab/DocVQA", "DocVQA", "validation",
     "3. Document VQA", "ANLS",
     "Average Normalized Levenshtein Similarity vs best gold; 0 below 0.5 similarity.",
     "https://arxiv.org/abs/2007.00398"),
    ("infovqa", "lmms-lab/DocVQA", "InfographicVQA", "validation",
     "3. Document VQA (infographics)", "ANLS",
     "ANLS; infographics need layout + numeric reasoning.",
     "https://arxiv.org/abs/2104.12756"),
    ("textvqa", "lmms-lab/textvqa", None, "validation",
     "3. Scene-text VQA", "VQA accuracy",
     "VQA acc = min(#humans_agree/3, 1) over 10 answers.",
     "https://arxiv.org/abs/1904.08920"),
    ("chartqa", "lmms-lab/ChartQA", None, "test",
     "6. Chart understanding", "relaxed_acc",
     "Relaxed accuracy: numeric within 5% rel. error, else exact match.",
     "https://arxiv.org/abs/2203.10244"),
    ("ocrbench", "echo840/OCRBench", None, "test",
     "8. LMM OCR capability suite", "OCRBench (/1000)",
     "1 pt/item if gold substring of prediction; 5 sub-skills, score out of 1000.",
     "https://arxiv.org/abs/2305.07895"),
    ("funsd", "nielsr/funsd-layoutlmv3", None, "test",
     "4. Key Information Extraction (forms)", "entity-level F1",
     "Entity F1 (type+value exact match) + relation/linking F1 for key->value.",
     "https://arxiv.org/abs/1905.13538"),
    ("cord", "naver-clova-ix/cord-v2", None, "test",
     "4. Key Information Extraction (receipts)", "entity-level F1",
     "Field/entity F1 over ~30 fine-grained receipt fields.",
     "https://github.com/clovaai/cord"),
    ("sroie", "priyank-m/SROIE_2019_text_recognition", None, "train",
     "4. Key Information Extraction (receipts)", "field-level F1",
     "Exact match per field (company/date/address/total), micro-F1.",
     "https://rrc.cvc.uab.es/?ch=13"),
    ("pubtabnet", "apoidea/pubtabnet-html", None, "validation",
     "5. Table recognition", "TEDS / GriTS",
     "Tree-Edit-Distance Similarity on HTML table tree (structure + cells).",
     "https://github.com/ibm-aur-nlp/PubTabNet"),
    ("im2latex", "OleehyO/latex-formulas", "cleaned_formulas", "train",
     "7. Formula recognition", "edit distance / BLEU / exact",
     "Image->LaTeX; token accuracy, BLEU, normalized edit distance.",
     "https://github.com/OleehyO/TexTeller"),
    ("omnidocbench", "opendatalab/OmniDocBench", None, "train",
     "9. End-to-end page parsing", "edit distance / TEDS / CDM",
     "Per-element NED (text), TEDS (tables), CDM (formulas), reading-order edit dist.",
     "https://arxiv.org/abs/2412.07626"),
]


def _find_image(ex: dict):
    """Return the first PIL image found in a record (under 'image' or any image-like value)."""
    from PIL import Image

    if "image" in ex and isinstance(ex["image"], Image.Image):
        return ex["image"]
    for v in ex.values():
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, list) and v and isinstance(v[0], Image.Image):
            return v[0]
    return None


def _json_safe(ex: dict) -> dict:
    """Drop the image, keep the rest as a JSON-serialisable GT label (truncated)."""
    out = {}
    for k, v in ex.items():
        try:
            json.dumps(v)
            out[k] = v if not isinstance(v, str) else v[:2000]
        except (TypeError, ValueError):
            out[k] = f"<{type(v).__name__}>"
    return out


def fetch_one(entry, force: bool = False) -> bool:
    from datasets import load_dataset

    key, repo, cfg, split, cat, metric, note, src = entry
    folder = OUT / key
    img_path = folder / "sample.png"
    if img_path.exists() and not force:
        print(f"[skip] {key}: already present")
        return True
    try:
        ds = load_dataset(repo, cfg, split=split, streaming=True)
        ex = dict(next(iter(ds)))
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:140]}")
        return False

    img = _find_image(ex)
    folder.mkdir(parents=True, exist_ok=True)
    if img is not None:
        img.convert("RGB").save(img_path)
    label = {
        "benchmark": key,
        "category": cat,
        "metric": metric,
        "metric_note": note,
        "hf_id": repo,
        "config": cfg,
        "split": split,
        "source": src,
        "ground_truth": _json_safe(ex),
    }
    (folder / "sample.json").write_text(
        json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok]   {key}: image={'yes' if img is not None else 'NONE'} -> {folder}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", help="subset of benchmark keys")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()

    entries = [e for e in REGISTRY if not args.only or e[0] in args.only]
    ok = sum(fetch_one(e, force=args.force) for e in entries)
    print(f"\n[done] {ok}/{len(entries)} benchmarks fetched -> {OUT}")
    # avoid a known pyarrow/threading teardown abort by exiting hard after flush
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

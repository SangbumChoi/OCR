#!/usr/bin/env python3
"""Measure the *richness* of a generated synthetic corpus (PRD: prd_synthetic_diversity.md).

Computes, over data/probes/realistic_cases (or --root): doc-type coverage, visual spread
(brightness/colour/aspect CoV), layout spread (#fields, table rows, page sizes), task-type
distribution, language distribution, near-duplicate rate (perceptual hash), and unique-content rate.
Then checks the v1 "rich enough" acceptance criteria. Writes docs/results/synthetic_diversity_report.md.

    python scripts/make_realistic_cases.py --count 30 --out /tmp/rich   # generate at scale
    python scripts/measure_diversity.py --root /tmp/rich
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
UNDERSTANDING = {"L1-locate", "L1-region", "H-count", "H1-aggregate", "H-accounting",
                 "H-extract-strict", "H-action", "H-comprehension"}


_HASH_SIDE = 16          # 16x16 average hash (256 bits) — 8x8 is too coarse for text-heavy pages
_EXACT_BITS = 4          # near-IDENTICAL (memorization risk: the same render seen twice) if <=4/256
_TMPL_BITS = 16          # "same template family" similarity descriptor (<=16/256) — not pass/fail


def _phash(img: Image.Image) -> int:
    """16x16 average-hash perceptual fingerprint (cheap near-duplicate detector)."""
    g = np.asarray(img.convert("L").resize((_HASH_SIDE, _HASH_SIDE), Image.LANCZOS), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _cov(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = st.mean(xs)
    return round(st.pstdev(xs) / m, 3) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "probes" / "realistic_cases"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "synthetic_diversity_report.md"))
    ap.add_argument("--label", default=None,
                    help="human label for the corpus in the report header (defaults to --root); "
                         "use this so a scale snapshot doesn't bake an ephemeral /tmp path into the doc")
    a = ap.parse_args()
    root = Path(a.root)
    gts = sorted(root.rglob("gt.json"))
    if not gts:
        sys.exit(f"no gt.json under {root} — generate first")

    doctypes, langs, atypes = Counter(), Counter(), Counter()
    bright, redm, aspect, nfields, rows, sizes, hashes, sigs = [], [], [], [], [], [], [], []
    values: set = set()
    n = 0
    for gp in gts:
        d = json.loads(gp.read_text())
        img_p = gp.parent / "clean.png"
        if not img_p.exists():
            continue
        n += 1
        doctypes[d.get("type", "?")] += 1
        for lg in d.get("languages", ["en"]):
            langs[lg] += 1
        for q in d.get("qa_detailed", []):
            atypes[q["answer_type"]] += 1
        for f in d.get("fields_detailed", []):
            if f.get("value"):
                values.add(f["value"])
        nfields.append(len([f for f in d.get("fields_detailed", []) if f.get("value")]))
        rk = [v for k, v in d.get("fields", {}).items() if k.endswith("_rows")]
        if rk:
            rows.append(rk[0])
        sz = d.get("render", {}).get("size_px") or [0, 0]
        sizes.append(tuple(sz))
        if sz[1]:
            aspect.append(round(sz[0] / sz[1], 3))
        arr = np.asarray(Image.open(img_p).convert("RGB").resize((64, 64)))
        bright.append(float(arr.mean()))
        redm.append(float(arr[:, :, 0].mean()))
        hashes.append(_phash(Image.open(img_p)))
        # content signature = the doc's gold answers (a true duplicate = same image AND same labels)
        sigs.append(tuple(sorted(tuple(q["answers"]) for q in d.get("qa_detailed", []))))

    # signals: (a) TRUE duplicate = near-identical image AND identical gold labels (memorization
    # risk); (b) same-template visual similarity (descriptor); (c) mean pairwise hamming.
    true_dup = tmpl_sim = 0
    dists = []
    for i in range(len(hashes)):
        near_img = near_tmpl = same_content = False
        for j in range(len(hashes)):
            if i == j:
                continue
            hd = bin(hashes[i] ^ hashes[j]).count("1")
            if j > i:
                dists.append(hd)
            if hd <= _TMPL_BITS:
                near_tmpl = True
            if hd <= _EXACT_BITS:
                near_img = True
                if sigs[i] and sigs[i] == sigs[j]:
                    same_content = True
        true_dup += int(near_img and same_content)
        tmpl_sim += int(near_tmpl)
    dup_rate = round(true_dup / n, 3) if n else 0.0          # acceptance: same image AND same labels
    tmpl_rate = round(tmpl_sim / n, 3) if n else 0.0          # descriptor (layout variety -> PRD v2)
    mean_ham = round(st.mean(dists), 1) if dists else 0.0
    n_qa = sum(atypes.values())
    reason_share = round(sum(atypes[a] for a in atypes if a in UNDERSTANDING) / n_qa, 3) if n_qa else 0.0
    uniq_rate = round(len(values) / max(1, sum(nfields)), 3)

    checks = {
        "doc_types >= 14": len(doctypes) >= 14,
        "true_dup_rate (same image+labels) < 0.05": dup_rate < 0.05,
        "brightness_CoV > 0.1": _cov(bright) > 0.1,
        "color_CoV > 0.1": _cov(redm) > 0.1,
        "reasoning_share >= 0.40": reason_share >= 0.40,
        "answer_type_families >= 6": len(atypes) >= 6,
    }
    L = ["# Synthetic diversity / richness report\n",
         "_Generated by `scripts/measure_diversity.py` (PRD: `docs/report/prd_synthetic_diversity.md`). "
         "Reproduce a scale snapshot with `python scripts/make_realistic_cases.py --count 20 --out <dir> "
         "&& python scripts/measure_diversity.py --root <dir> --label 'count=20 scale snapshot'`._\n",
         f"Corpus: **{a.label or root}** — **{n} documents**, {len(set(sizes))} distinct page sizes.\n",
         "## Coverage", f"- doc types: **{len(doctypes)}** {dict(doctypes)}",
         f"- languages: {dict(langs)}", f"- answer-type families: **{len(atypes)}**",
         "\n## Visual spread (coefficient of variation)",
         f"- brightness CoV **{_cov(bright)}** · red-channel CoV **{_cov(redm)}** · aspect CoV **{_cov(aspect)}**",
         f"- **true-duplicate rate** (near-identical image AND identical gold labels): **{dup_rate}** "
         f"(the memorization signal — want ~0)",
         f"- same-template visual similarity (≤{_TMPL_BITS}/256): **{tmpl_rate}** · mean pairwise "
         f"hamming **{mean_ham}**/256  (layout-variety descriptor -> PRD v2 lever)",
         "\n## Layout spread",
         f"- #fields/doc: min {min(nfields)} / max {max(nfields)} / mean {round(st.mean(nfields),1)}",
         f"- table rows: {sorted(set(rows)) if rows else 'n/a'}",
         "\n## Task difficulty mix",
         f"- reasoning/understanding share of QAs: **{reason_share}** ({n_qa} QAs)",
         f"- per answer_type: {dict(atypes)}",
         f"- unique field-value rate: **{uniq_rate}**",
         "\n## Acceptance (v1 'rich enough')"]
    for k, v in checks.items():
        L.append(f"- [{'x' if v else ' '}] {k}")
    verdict = "RICH ENOUGH (v1)" if all(checks.values()) else "NOT yet (see unchecked above)"
    L.append(f"\n**Verdict: {verdict}**")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] -> {a.out}")


if __name__ == "__main__":
    main()

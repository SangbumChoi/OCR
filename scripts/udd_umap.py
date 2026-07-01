#!/usr/bin/env python3
"""Feature UMAP of the UDD (Universal Document Dataset).

Embeds each record's *content* (task + instruction + answers + a snippet of full_text/fields) with
TF-IDF and projects to 2D with UMAP, then plots the standardized space two ways: coloured by **task**
and by **source dataset**. This visualises how the scattered public benchmarks land in one common
representation once unified. Writes docs/report/figures/udd_umap.png.

    python scripts/udd_umap.py --src data/udd/hf/_all
    python scripts/udd_umap.py --repo danelcsb/UDD          # pull from the Hub instead
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(r) -> str:
    fields = json.loads(r.get("fields_json") or "[]")
    fkeys = " ".join(f.get("key", "") for f in fields[:20])
    return " ".join(str(x) for x in [
        r.get("task", ""), r.get("instruction", ""),
        " ".join(r.get("answers") or []), (r.get("full_text") or "")[:200], fkeys])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"),
                   help="local load_from_disk path")
    p.add_argument("--repo", default=None, help="HF repo to pull instead of --src")
    p.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" / "udd_umap.png"))
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import umap
    from sklearn.feature_extraction.text import TfidfVectorizer

    if args.repo:
        from datasets import load_dataset
        ds = load_dataset(args.repo, split="train")
    else:
        from datasets import load_from_disk
        ds = load_from_disk(args.src)
    rows = [{k: ds[k][i] for k in ("task", "source", "instruction", "answers", "full_text",
                                   "fields_json")} for i in range(len(ds))]
    texts = [_text(r) for r in rows]
    tasks = [r["task"] for r in rows]
    sources = [r["source"] for r in rows]

    X = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(texts)
    n = X.shape[0]
    emb = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, metric="cosine",
                    random_state=42).fit_transform(X.toarray())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, labels, title in ((axes[0], tasks, "by task"), (axes[1], sources, "by source dataset")):
        cats = sorted(set(labels))
        cmap = plt.get_cmap("tab20" if len(cats) > 10 else "tab10")
        for i, c in enumerate(cats):
            m = np.array([lb == c for lb in labels])
            ax.scatter(emb[m, 0], emb[m, 1], s=22, alpha=0.8, color=cmap(i % 20), label=c)
        ax.set_title(f"UDD feature UMAP — {title}", fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best", ncol=1 if len(cats) <= 8 else 2, framealpha=0.9)
    fig.suptitle(f"UDD — {n} records, TF-IDF(content) → UMAP", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out}  ({n} points, {len(set(tasks))} tasks, {len(set(sources))} sources)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

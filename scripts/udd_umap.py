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
    els = json.loads(r.get("elements_json") or r.get("fields_json") or "[]")
    fkeys = " ".join(e.get("key", "") for e in els[:20])
    flat_answers = " ".join(a for inner in (r.get("answers") or []) for a in inner)
    return " ".join(str(x) for x in [
        r.get("task", ""), " ".join(r.get("instructions") or []),
        flat_answers, (r.get("full_text") or "")[:200], fkeys])


def _image_features(ds, model_id: str):
    """CLIP image embeddings (L2-normalized) for every row — the visual feature space of the docs."""
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(model_id)
    feats = []
    imgs = [ds[i]["image"].convert("RGB") for i in range(len(ds))]
    with torch.no_grad():
        for i in range(0, len(imgs), 16):
            inp = proc(images=imgs[i:i + 16], return_tensors="pt").to(dev)
            f = model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"),
                   help="local load_from_disk path")
    p.add_argument("--repo", default=None, help="HF repo to pull instead of --src")
    p.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" / "udd_umap.png"))
    p.add_argument("--features", choices=["image", "text"], default="image",
                   help="'image' = CLIP image embeddings (default); 'text' = TF-IDF of the content")
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--sample", type=int, default=3000,
                   help="max points, stratified per source (CPU CLIP over a huge corpus does not "
                        "scale; also dedups QA fan-out rows sharing one image). 0 = every row.")
    p.add_argument("--seed", type=int, default=42)
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
    # one point per distinct IMAGE (QA fan-out repeats the image), then stratified per-source cap
    all_sources = ds["source"]
    all_sids = ds["sample_id"]
    by_src: dict[str, list[int]] = {}
    seen_img: set[str] = set()
    for i in range(len(ds)):
        img_key = f"{all_sources[i]}:{all_sids[i].rsplit('_', 1)[0]}"
        if img_key in seen_img:
            continue
        seen_img.add(img_key)
        by_src.setdefault(all_sources[i], []).append(i)
    if args.sample:
        import random as _random
        rng = _random.Random(args.seed)
        per_src = max(1, args.sample // max(1, len(by_src)))
        idx = [i for lst in by_src.values() for i in (rng.sample(lst, per_src)
                                                      if len(lst) > per_src else lst)]
    else:
        idx = [i for lst in by_src.values() for i in lst]
    idx.sort()
    ds = ds.select(idx)
    tasks = list(ds["task"])
    sources = list(ds["source"])

    if args.features == "image":
        X = _image_features(ds, args.clip)          # CLIP image embeddings (dense, L2-normalized)
        feat_desc = f"CLIP({args.clip.split('/')[-1]}) image"
    else:
        rows = [{k: ds[k][i] for k in ("task", "source", "instructions", "answers", "full_text",
                                       "elements_json")} for i in range(len(ds))]
        X = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(
            [_text(r) for r in rows]).toarray()
        feat_desc = "TF-IDF(content)"
    n = X.shape[0]
    emb = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.1, metric="cosine",
                    random_state=42).fit_transform(X)

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
    fig.suptitle(f"UDD — {n} records, {feat_desc} → UMAP", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[ok] wrote {out}  ({n} points, {len(set(tasks))} tasks, {len(set(sources))} sources)")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

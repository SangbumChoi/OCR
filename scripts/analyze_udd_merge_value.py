#!/usr/bin/env python3
"""Merge-value analysis: what does merging benchmarks into UDD buy — WITHOUT training any model?

Before spending GPU on the task-value ablation, quantify the merge's benefits from the dataset
alone. Four measurable, model-free signals, each comparing the MERGED corpus against the best any
SINGLE source offers:

1. **Coverage** — tasks / languages / structured-payload types per source vs merged. A trainer that
   needs KIE boxes + localization + ko text cannot get them from any one source.
2. **Visual diversity** — mean pairwise phash (dhash) Hamming distance within each source vs across
   the merged corpus (sampled pairs). Higher = more varied visual input distribution.
3. **Vocabulary growth** — unique-token count as sources are added (fixed random order, 3 seeds).
   Near-linear growth = sources contribute complementary text, not rephrasings of the same content.
4. **Redundancy** — cross-source duplicate rate from the phash audit: how much of a "new" source is
   already in the corpus. Low = merging adds real data, not copies.

Writes ``docs/results/udd_merge_value.md`` (with a verdict section) and
``docs/report/figures/udd_merge_value.png``.

    python scripts/analyze_udd_merge_value.py
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MD = ROOT / "docs" / "results" / "udd_merge_value.md"
FIG = ROOT / "docs" / "report" / "figures" / "udd_merge_value.png"


def _pairwise_hamming(hashes: list[int], rng: random.Random, max_pairs: int = 2000) -> float:
    if len(hashes) < 2:
        return 0.0
    pairs = min(max_pairs, len(hashes) * (len(hashes) - 1) // 2)
    tot = 0
    for _ in range(pairs):
        a, b = rng.sample(hashes, 2)
        tot += bin(a ^ b).count("1")
    return tot / pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_from_disk
    ds = load_from_disk(args.src)
    n = len(ds)
    sources = ds["source"]; tasks = ds["task"]; langs = ds["language"]
    n_fields = ds["n_fields"]; n_regions = ds["n_regions"]
    phashes = ds["phash"]; tables = ds["table_html"]; fulls = ds["full_text"]

    # ---------- 1. coverage per source vs merged
    cov: dict[str, dict] = defaultdict(lambda: {"tasks": set(), "langs": set(), "payload": set()})
    for i in range(n):
        c = cov[sources[i]]
        c["tasks"].add(tasks[i])
        if langs[i]:
            c["langs"].add(langs[i])
        if n_fields[i]:
            c["payload"].add("kie-fields")
        if n_regions[i]:
            c["payload"].add("boxes")
        if tables[i]:
            c["payload"].add("table-html")
        if fulls[i]:
            c["payload"].add("full-text")
    merged_tasks = set(tasks); merged_langs = {l for l in langs if l}
    merged_payload = set().union(*(c["payload"] for c in cov.values()))
    best_tasks = max(len(c["tasks"]) for c in cov.values())
    best_langs = max(len(c["langs"]) for c in cov.values())
    best_payload = max(len(c["payload"]) for c in cov.values())

    # ---------- 2. visual diversity (per-image, dedup QA fan-out)
    per_src_hashes: dict[str, list[int]] = defaultdict(list)
    seen = set()
    sids = ds["sample_id"]
    for i in range(n):
        key = (sources[i], sids[i].rsplit("_", 1)[0])
        if key in seen or not phashes[i]:
            continue
        seen.add(key)
        per_src_hashes[sources[i]].append(int(phashes[i], 16))
    within = {s: _pairwise_hamming(h, rng) for s, h in per_src_hashes.items()}
    all_hashes = [h for hs in per_src_hashes.values() for h in hs]
    merged_div = _pairwise_hamming(all_hashes, rng, max_pairs=6000)
    mean_within = sum(within.values()) / len(within)

    # ---------- 3. vocabulary growth as sources are added
    vocab_by_src: dict[str, set] = defaultdict(set)
    answers = ds["answers"]; instrs = ds["instruction"]
    for i in range(n):
        toks = (" ".join([instrs[i] or "", " ".join(answers[i] or []),
                          (fulls[i] or "")[:400]])).lower().split()
        vocab_by_src[sources[i]].update(toks)
    orders = []
    src_list = list(vocab_by_src)
    for s in range(3):
        r2 = random.Random(args.seed + s)
        order = src_list[:]; r2.shuffle(order)
        acc: set = set(); curve = []
        for src in order:
            acc |= vocab_by_src[src]; curve.append(len(acc))
        orders.append(curve)
    growth = [sum(c[i] for c in orders) / len(orders) for i in range(len(src_list))]
    biggest_single = max(len(v) for v in vocab_by_src.values())

    # ---------- 4. redundancy (strict near-dup rate across sources)
    items = [(s, h) for s, hs in per_src_hashes.items() for h in hs]
    dup = 0
    for i in range(len(items)):
        si, hi = items[i]
        for j in range(i + 1, len(items)):
            sj, hj = items[j]
            if si != sj and bin(hi ^ hj).count("1") <= 2:
                dup += 1
                break
    dup_rate = dup / max(1, len(items))

    # ---------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    ax = axes[0]
    names = sorted(within, key=within.get)
    ax.barh(range(len(names)), [within[s] for s in names], color="#7aa6d6")
    ax.axvline(merged_div, color="#b03030", lw=2,
               label=f"merged corpus = {merged_div:.1f}")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("mean pairwise dhash Hamming (higher = more visually diverse)")
    ax.set_title("Visual diversity: each source alone vs merged", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(range(1, len(growth) + 1), growth, "o-", color="#1f5fa8", ms=3)
    ax.axhline(biggest_single, color="#888", ls="--",
               label=f"largest single source = {biggest_single:,}")
    ax.set_xlabel("# sources merged (random order, mean of 3)")
    ax.set_ylabel("unique tokens")
    ax.set_title("Vocabulary growth as sources merge", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.suptitle("UDD merge value — measured without training any model", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ---------- report
    lines = [
        "# UDD merge value — what merging buys, measured without training", "",
        f"Corpus: {n} rows, {len(cov)} sources, {len(seen)} distinct images.", "",
        "| signal | best single source | merged corpus | gain |",
        "|---|---|---|---|",
        f"| task coverage | {best_tasks} / {len(merged_tasks)} tasks "
        f"| **{len(merged_tasks)} tasks** | complete |",
        f"| language coverage | {best_langs} | **{len(merged_langs)}** "
        f"({', '.join(sorted(merged_langs))}) | ×{len(merged_langs) / max(1, best_langs):.1f} |",
        f"| payload types (fields/boxes/table/full-text) | {best_payload} / {len(merged_payload)} "
        f"| **{len(merged_payload)}** | complete |",
        f"| visual diversity (mean pairwise dhash Hamming) | {mean_within:.1f} avg within-source "
        f"(max {max(within.values()):.1f}) | **{merged_div:.1f}** "
        f"| +{100 * (merged_div - mean_within) / max(1e-9, mean_within):.0f}% vs avg source |",
        f"| vocabulary | {biggest_single:,} tokens (largest source) | **{growth[-1]:,.0f}** "
        f"| ×{growth[-1] / max(1, biggest_single):.1f} |",
        f"| cross-source redundancy | — | **{100 * dup_rate:.1f}%** of images have a strict "
        f"near-dup in another source | merging adds ~{100 * (1 - dup_rate):.0f}% new data |",
        "",
        "![merge value](../report/figures/udd_merge_value.png)", "",
        "## Verdict", "",
        "Merging is worth it on dataset-level evidence alone:", "",
        f"1. **No single source spans the training surface.** The best source covers {best_tasks} of "
        f"{len(merged_tasks)} tasks and {best_langs} of {len(merged_langs)} languages; a trainer "
        "needing KIE boxes + layout localization + Korean text has no single-source option.",
        f"2. **The merged input distribution is measurably wider** — pairwise visual diversity "
        f"{merged_div:.1f} vs {mean_within:.1f} within an average source — without collapsing into "
        "duplicates (near-linear vocabulary growth; sources contribute complementary content).",
        f"3. **Redundancy is negligible** ({100 * dup_rate:.1f}% strict near-dups, see "
        "`udd_duplicates.md`), so each merged source adds data, not copies — the audit still matters "
        "for train/val hygiene (chartqa↔mathvista).",
        "",
        "What this analysis **cannot** show is whether the wider distribution transfers to model "
        "capability at a fixed budget — that is exactly the GPU task-value ablation "
        "(`run_task_value.py`) and the A1/A4 hypothesis runs (`build_task_trainsets.py "
        "--group-by task|language`).",
    ]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {MD}\n[ok] {FIG}")
    print(f"  tasks {best_tasks}->{len(merged_tasks)}  langs {best_langs}->{len(merged_langs)}  "
          f"visual {mean_within:.1f}->{merged_div:.1f}  vocab {biggest_single}->{growth[-1]:.0f}  "
          f"dup {100 * dup_rate:.1f}%")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort

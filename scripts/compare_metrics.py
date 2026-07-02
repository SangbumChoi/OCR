#!/usr/bin/env python3
"""Compare the metric bank's tendencies: what does each evaluation metric forgive or punish?

Two modes, both scoring the SAME (prediction, golds) pairs under EVERY bank metric
(``docvlm_eval.metrics.bank``: exact, anls, ned, relaxed_acc, ocrbench, token_f1, drop_em,
cer_sim, semantic_match):

1. **Characterization (default, no model needed)** — take real answers from UDD, apply CONTROLLED
   perturbations that each represent one equivalence class (case change, punctuation, number->word,
   thousand separators, one-char typo, sentence wrapping, truncation, word shuffle, wrong answer),
   and score every metric on (perturbed, [original]). The perturbation × metric mean-score matrix
   IS each metric's tolerance profile — which surface changes it treats as "still correct". Sanity
   anchors: identity ≈ 1.0 everywhere, wrong-answer ≈ 0.0 everywhere.

2. **Predictions mode (--preds)** — score a real predictions jsonl
   (rows: {"prediction": ..., "answers": [...]} or {"pred": ..., "golds": [...]}) under all metrics:
   per-metric means, the pairwise Pearson correlation matrix (which metrics rank models the same),
   and the top disagreement examples (max spread across metrics) for eyeballing.

Writes docs/results/metric_tendency.md + docs/report/figures/metric_tendency.png.

    python scripts/compare_metrics.py                       # characterize on UDD answers
    python scripts/compare_metrics.py --preds outputs/preds.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.metrics import METRIC_BANK, score_all  # noqa: E402

MD = ROOT / "docs" / "results" / "metric_tendency.md"
FIG = ROOT / "docs" / "report" / "figures" / "metric_tendency.png"

_NUM2WORD = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
             "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def perturbations(rng: random.Random) -> dict:
    """Each perturbation = one equivalence class of surface change (or a control)."""
    def sep(a):        # 1000 -> 1,000
        return re.sub(r"\b(\d)(\d{3})\b", r"\1,\2", a)

    def num2word(a):
        return re.sub(r"\b(10|[1-9])\b", lambda m: _NUM2WORD[m.group(1)], a)

    def typo(a):
        if len(a) < 4:
            return a + "x"
        i = rng.randrange(1, len(a) - 2)
        return a[:i] + a[i + 1] + a[i] + a[i + 2:]

    def shuffle(a):
        w = a.split()
        if len(w) < 3:
            return a
        rng.shuffle(w)
        return " ".join(w)

    return {
        "identity (control≈1)": lambda a: a,
        "case change": lambda a: a.upper() if a != a.upper() else a.lower(),
        "punctuation": lambda a: f'"{a}".',
        "whitespace": lambda a: "  " + a.replace(" ", "  ") + " ",
        "thousands separator": sep,
        "number -> word": num2word,
        "1-char typo": typo,
        "sentence wrapping": lambda a: f"The answer is {a}.",
        "truncated (half)": lambda a: a[: max(1, len(a) // 2)],
        "word shuffle": shuffle,
        "wrong answer (control≈0)": None,   # replaced by another row's answer
    }


def characterize(args) -> tuple[list[str], list[str], list[list[float]], str]:
    from datasets import load_from_disk
    ds = load_from_disk(args.src)
    rng = random.Random(args.seed)
    # short factual answers only (recognition/table golds are pages/HTML — CER territory, and
    # perturbing a 2k-char transcript by one char tells us nothing)
    pool = []
    for i in range(len(ds)):
        ans = ds["answers"][i]
        if ans and 1 <= len(ans[0]) <= 60 and ds["task"][i] in ("vqa", "reasoning", "kie",
                                                                "classification"):
            pool.append(ans[0])
    rng.shuffle(pool)
    pool = pool[: args.n]
    perts = perturbations(rng)
    metrics = list(METRIC_BANK)
    M: list[list[float]] = []
    for pname, fn in perts.items():
        rowscores = {m: 0.0 for m in metrics}
        for k, a in enumerate(pool):
            pred = pool[(k + len(pool) // 2) % len(pool)] if fn is None else fn(a)
            s = score_all(pred, [a])
            for m in metrics:
                rowscores[m] += s[m]
        M.append([rowscores[m] / len(pool) for m in metrics])
    src_desc = f"{len(pool)} real UDD answers (vqa/reasoning/kie/classification), seed={args.seed}"
    return list(perts), metrics, M, src_desc


def preds_mode(args) -> None:
    rows = [json.loads(ln) for ln in Path(args.preds).read_text().splitlines() if ln.strip()]
    metrics = list(METRIC_BANK)
    per = []
    for r in rows:
        pred = r.get("prediction", r.get("pred", ""))
        golds = r.get("answers", r.get("golds", []))
        per.append((r.get("sample_id", ""), pred, golds, score_all(pred, golds)))
    n = len(per)
    means = {m: sum(p[3][m] for p in per) / n for m in metrics}
    # Pearson correlation between metric score vectors
    import math
    def corr(a, b):
        ma, mb = sum(a) / n, sum(b) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        va = math.sqrt(sum((x - ma) ** 2 for x in a)) * math.sqrt(sum((y - mb) ** 2 for y in b))
        return cov / va if va else 1.0
    vecs = {m: [p[3][m] for p in per] for m in metrics}
    lines = [f"# Metric comparison on {args.preds} ({n} predictions)", "",
             "| metric | mean |", "|---|---|"]
    lines += [f"| {m} | {means[m]:.3f} |" for m in sorted(means, key=means.get, reverse=True)]
    lines += ["", "## Pairwise Pearson correlation", "",
              "| | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for a in metrics:
        lines.append(f"| **{a}** | " + " | ".join(f"{corr(vecs[a], vecs[b]):.2f}"
                                                  for b in metrics) + " |")
    per.sort(key=lambda p: max(p[3].values()) - min(p[3].values()), reverse=True)
    lines += ["", "## Top disagreements (max metric spread)", ""]
    for sid, pred, golds, s in per[:10]:
        hi = max(s, key=s.get); lo = min(s, key=s.get)
        lines.append(f"- `{sid}` pred={pred[:60]!r} gold={golds[:2]!r} — "
                     f"{hi}={s[hi]:.2f} vs {lo}={s[lo]:.2f}")
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {MD}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(ROOT / "data" / "udd" / "hf" / "_all"))
    p.add_argument("--preds", default=None, help="predictions jsonl -> correlation/disagreement mode")
    p.add_argument("--n", type=int, default=300, help="answers sampled for characterization")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if args.preds:
        preds_mode(args)
        return

    perts, metrics, M, src_desc = characterize(args)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.15 * len(metrics) + 3, 0.55 * len(perts) + 2.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(perts))); ax.set_yticklabels(perts, fontsize=9)
    for i in range(len(perts)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Metric tolerance profile — mean score of each metric on each perturbation\n"
                 f"(prediction = perturbed gold; {src_desc})", fontsize=11, fontweight="bold")
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=130, bbox_inches="tight"); plt.close(fig)

    lines = ["# Metric tendency — what each evaluation metric forgives", "",
             f"Prediction = a controlled perturbation of the gold; score = mean over {src_desc}. "
             "A cell near 1.0 means the metric treats that change as still-correct; near 0.0 means "
             "it punishes it. `identity` and `wrong answer` are sanity controls.", "",
             "| perturbation | " + " | ".join(metrics) + " |",
             "|---|" + "---|" * len(metrics)]
    for pname, row in zip(perts, M):
        lines.append(f"| {pname} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    lines += ["", "![metric tendency](../report/figures/metric_tendency.png)", "",
              "Reading guide: `exact` collapses on every change except pure surface noise it "
              "normalizes away; `anls`/`ned` forgive small edits (typos) but not semantic "
              "rewrites (number->word); `drop_em` is the only EM that survives number->word and "
              "separators; `token_f1` uniquely gives credit through sentence wrapping and word "
              "shuffle (order-free); `cer_sim` degrades smoothly with edit distance; "
              "`semantic_match` = the layered union (surface OR canonical OR token overlap).", "",
              "Re-run on real model outputs: `python scripts/compare_metrics.py --preds "
              "<predictions.jsonl>` -> per-metric means + Pearson correlations + top disagreements."]
    MD.parent.mkdir(parents=True, exist_ok=True)
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {MD}\n[ok] {FIG}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)   # avoid pyarrow/threading teardown abort

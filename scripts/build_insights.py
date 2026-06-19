#!/usr/bin/env python3
"""Synthesize cross-model insights from all stored results into docs/report/insights.md.

Reads whatever exists under docs/results/ (matrices, probe signals, custom-eval breakdown, per-model
summaries for efficiency, and raw predictions for the OOV fallback study) and emits automated
findings: capability leaders, reasoning emergence vs size, grounding gap, efficiency frontier,
per-language / per-class leaders, PaddleOCR version progression, and OOV fallback patterns.

Degrades gracefully when a section has no data (e.g. before the GPU run).
    python scripts/build_insights.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import load_jsonl  # noqa: E402

R = ROOT / "docs" / "results"


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _summaries():
    out = {}
    for f in R.glob("*/*/summary.json"):
        s = _load(f)
        if s:
            out[(f.parent.parent.name, f.parent.name)] = s
    return out


def classify_oov(pred: str) -> str:
    p = (pred or "").strip().lower()
    if not p:
        return "empty"
    if re.search(r"unread|cannot|can't|unable|not able|no (text|legible)|unknown|don't|do not", p):
        return "abstain"
    if re.search(r"[a-z]", p) and not re.search(r"[^\x00-\x7f]", p):
        return "latin/guess"
    if re.search(r"[^\x00-\x7f]", p):
        return "nonlatin-copy"
    if re.search(r"\d", p):
        return "numeric"
    return "other"


def main():
    L = ["# Cross-model insights (auto-generated)\n",
         "Synthesized by `scripts/build_insights.py` from `docs/results/`. Run it after a full sweep "
         "(GPU) to populate every section; partial data degrades gracefully.\n"]
    summaries = _summaries()
    models = sorted({m for (m, _b) in summaries})
    L.append(f"**Models with results:** {', '.join(models) or '(none yet)'}\n")

    # ---- 1. capability leaders (capability probe) ----
    capj = _load(R / "matrix_capability.json")
    if capj and capj.get("scores"):
        L.append("## 1. Capability leaders (capability probe)\n")
        axes = capj["benchmarks"]
        scores = capj["scores"]
        L.append("| axis | best model | score |")
        L.append("|---|---|---|")
        for ax in axes:
            ranked = sorted(((m, s.get(ax, 0.0)) for m, s in scores.items() if m != "dummy-echo"),
                            key=lambda x: x[1], reverse=True)
            if ranked:
                L.append(f"| {ax} | {ranked[0][0]} | {ranked[0][1]:.2f} |")
        # reasoning emergence
        L.append("\n**Reasoning emergence vs size:** integrative axes by model "
                 "(params from summaries):")
        for m in models:
            params = next((s.get("param_count_m") for (mm, _), s in summaries.items() if mm == m), None)
            sc = scores.get(m, {})
            if "cap_integ_sum" in sc:
                psize = f"~{params:.0f}M" if params else "size n/a"
                L.append(f"- {m} ({psize}): sum={sc.get('cap_integ_sum')}, "
                         f"rel={sc.get('cap_integ_rel', 'n/a')}")

    # ---- 2. grounding gap ----
    gnd = []
    for (m, b), s in summaries.items():
        if b == "capability":
            ps = _load(R / m / b / "per_sample.json") or []
            g = next((r["score"] for r in ps if r["sample_id"] == "cap_ground"), None)
            if g is not None:
                gnd.append((m, g))
    if gnd:
        best = max(gnd, key=lambda x: x[1])
        L.append("\n## 2. Grounding (spatial localisation)\n")
        L.append(f"Best grounding score: **{best[0]} = {best[1]:.2f}**. "
                 + ("No model produces usable boxes (general VLMs lack a spotting head)."
                    if best[1] < 0.3 else "Spotting-capable models lead here.") + "\n")

    # ---- 3. efficiency frontier ----
    eff = defaultdict(lambda: {"acc": [], "lat": None, "cpu": None, "gpu": None, "params": None})
    for (m, b), s in summaries.items():
        if s.get("score") is not None:
            eff[m]["acc"].append(s["score"])
        for k_src, k_dst in (("avg_latency_s", "lat"), ("peak_cpu_mb", "cpu"),
                             ("peak_gpu_mb", "gpu"), ("param_count_m", "params")):
            if s.get(k_src) is not None:
                eff[m][k_dst] = s[k_src]
    if eff:
        L.append("\n## 3. Efficiency vs quality\n")
        L.append("| model | params(M) | mean score | avg lat(s) | peak CPU(MB) | peak GPU(MB) |")
        L.append("|---|---|---|---|---|---|")
        for m in sorted(eff):
            e = eff[m]
            mean = sum(e["acc"]) / len(e["acc"]) if e["acc"] else None
            L.append(f"| {m} | {e['params'] or '–'} | "
                     f"{mean:.3f} | {e['lat'] or '–'} | {e['cpu'] or '–'} | {e['gpu'] or '–'} |"
                     if mean is not None else f"| {m} | – | – | – | – | – |")

    # ---- 4. per-language / per-class (custom_eval) ----
    ce = _load(R / "custom_eval_breakdown.json")
    if ce:
        L.append("\n## 4. Custom-eval leaders (class / language)\n")
        # best per class
        classes = sorted({c for r in ce.values() for c in r.get("by_content_class", {})})
        for axis_name, key in (("content class", "by_content_class"), ("language", "by_language")):
            L.append(f"\n**By {axis_name}:**")
            keys = sorted({c for r in ce.values() for c in r.get(key, {})})
            for c in keys:
                ranked = sorted(((m, (r.get(key, {}).get(c) or 0)) for m, r in ce.items()),
                                key=lambda x: x[1], reverse=True)
                if ranked and ranked[0][1] > 0:
                    L.append(f"- {c}: {ranked[0][0]} ({ranked[0][1]})")

    # ---- 5. PaddleOCR-VL version progression ----
    paddles = [m for m in models if m.startswith("paddleocr-vl")]
    if paddles:
        L.append("\n## 5. PaddleOCR-VL version progression\n")
        for m in sorted(paddles):
            s = next((v for (mm, b), v in summaries.items() if mm == m and b == "capability"), None)
            if s:
                L.append(f"- {m}: capability score {s.get('score')}, "
                         f"avg lat {s.get('avg_latency_s')}s, peak GPU {s.get('peak_gpu_mb')}MB")

    # ---- 6. OOV fallback patterns ----
    oov_jsonl = ROOT / "data" / "benchmarks" / "oov_probe" / "oov.jsonl"
    if oov_jsonl.exists():
        meta = {s.sample_id: s for s in load_jsonl(oov_jsonl)}
        fb_rows = []
        for (m, b), s in summaries.items():
            if b != "oov":
                continue
            ps = _load(R / m / b / "per_sample.json") or []
            dist = defaultdict(int)
            legend_ok = None
            for r in ps:
                sid = r["sample_id"]
                if meta.get(sid) and meta[sid].meta.get("fallback_probe"):
                    dist[classify_oov(r.get("prediction", ""))] += 1
                if sid == "oov_legend":
                    legend_ok = r["score"]
            if dist or legend_ok is not None:
                fb_rows.append((m, dict(dist), legend_ok))
        if fb_rows:
            L.append("\n## 6. OOV fallback behaviour (un-tokenizable glyphs)\n")
            L.append("How models respond to glyphs absent from their tokenizer (fallback), and "
                     "whether an in-image legend lets them decode (reasoning).\n")
            L.append("| model | fallback distribution | legend-decode |")
            L.append("|---|---|---|")
            for m, dist, leg in fb_rows:
                L.append(f"| {m} | {dist} | {leg if leg is not None else '–'} |")

    # ---- top findings (heuristic synthesis) ----
    L.insert(2, _top_findings(capj, gnd, ce) + "\n")

    from docvlm_eval.report_md import prettify_tables
    (ROOT / "docs" / "report" / "insights.md").write_text(prettify_tables("\n".join(L)) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[done] -> docs/report/insights.md")


def _top_findings(capj, gnd, ce) -> str:
    bullets = ["## Top findings\n"]
    if capj and capj.get("scores"):
        sc = capj["scores"]
        reasoners = [m for m, s in sc.items() if s.get("cap_integ_rel", 0) >= 0.5]
        if reasoners:
            bullets.append(f"- **Relational reasoning** is cleared by: {', '.join(reasoners)} "
                           "(emerges around ~1B; smaller models fail).")
    if gnd and max(g for _, g in gnd) < 0.3:
        bullets.append("- **No model grounds**: spatial localisation (bbox) is ~0 for the tested "
                       "general VLMs — a systemic gap, motivating spotting-capable models.")
    if len(bullets) == 1:
        bullets.append("- (Run the full GPU sweep to populate findings.)")
    return "\n".join(bullets)


if __name__ == "__main__":
    main()

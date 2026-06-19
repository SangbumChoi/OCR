#!/usr/bin/env python3
"""Slice custom_eval results along the proposed axes: content-class, language, rotation
(retention vs 0deg), reading-direction, and spotting. Joins each model's per_sample.json with
the benchmark metadata (data/probes/custom_eval/custom_eval.jsonl) and writes
docs/results/custom_eval_breakdown.md + .json.

    python scripts/analyze_custom_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import load_jsonl  # noqa: E402

JSONL = ROOT / "data" / "benchmarks" / "custom_eval" / "custom_eval.jsonl"
RESULTS = ROOT / "docs" / "results"


def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None


def analyze(meta_by_id, rows):
    score = {r["sample_id"]: r["score"] for r in rows}
    by_class, by_lang = defaultdict(list), defaultdict(list)
    by_rot, direction, spotting = defaultdict(list), [], []
    for sid, m in meta_by_id.items():
        if sid not in score:
            continue
        s = score[sid]
        cls = m.answer_type
        by_class[cls].append(s)
        if cls == "text" and m.meta.get("rotation_deg", 0) == 0:
            by_lang[m.meta.get("language", "en")].append(s)
        if sid.startswith("ce_rot") and sid.endswith("_read"):
            by_rot[m.meta["rotation_deg"]].append(s)
        if cls == "direction":
            direction.append(s)
        if m.meta.get("spotting"):
            spotting.append(s)
    # rotation retention vs 0deg
    base = _mean(by_rot.get(0, [])) or 0.0
    retention = {f"{a}deg": (round((_mean(v) or 0) / base, 3) if base else None)
                 for a, v in sorted(by_rot.items())}
    return {
        "by_content_class": {k: _mean(v) for k, v in sorted(by_class.items())},
        "by_language": {k: _mean(v) for k, v in sorted(by_lang.items())},
        "rotation_read_score": {f"{a}deg": _mean(v) for a, v in sorted(by_rot.items())},
        "rotation_retention": retention,
        "reading_direction_acc": _mean(direction),
        "spotting_mean_iou": _mean(spotting),
    }


def main():
    meta_by_id = {s.sample_id: s for s in load_jsonl(JSONL)}
    out = {}
    for ps in sorted(RESULTS.glob("*/custom_eval/per_sample.json")):
        model = ps.parent.parent.name
        out[model] = analyze(meta_by_id, json.loads(ps.read_text()))
    if not out:
        raise SystemExit("No custom_eval results yet. Run run_matrix on custom_eval.jsonl first.")

    lines = ["# Custom-eval breakdown (proposed format)\n",
             "Per-model scores sliced by the axes the format is built around.\n"]
    # content class table
    classes = sorted({c for r in out.values() for c in r["by_content_class"]})
    lines.append("## By content class\n")
    lines.append("| model | " + " | ".join(classes) + " |")
    lines.append("|" + "|".join(["---"] * (len(classes) + 1)) + "|")
    for m, r in out.items():
        lines.append("| " + m + " | " + " | ".join(
            str(r["by_content_class"].get(c, "–")) for c in classes) + " |")
    # language
    langs = sorted({l for r in out.values() for l in r["by_language"]})
    lines.append("\n## By language (text)\n")
    lines.append("| model | " + " | ".join(langs) + " |")
    lines.append("|" + "|".join(["---"] * (len(langs) + 1)) + "|")
    for m, r in out.items():
        lines.append("| " + m + " | " + " | ".join(str(r["by_language"].get(l, "–")) for l in langs) + " |")
    # rotation retention + direction + spotting
    lines.append("\n## Rotation retention (read score / 0deg), reading-direction acc, spotting IoU\n")
    angs = sorted({a for r in out.values() for a in r["rotation_retention"]})
    lines.append("| model | " + " | ".join(angs) + " | dir-acc | spot-IoU |")
    lines.append("|" + "|".join(["---"] * (len(angs) + 3)) + "|")
    for m, r in out.items():
        lines.append("| " + m + " | " + " | ".join(str(r["rotation_retention"].get(a, "–")) for a in angs)
                     + f" | {r['reading_direction_acc']} | {r['spotting_mean_iou']} |")

    from docvlm_eval.report_md import prettify_tables
    (RESULTS / "custom_eval_breakdown.md").write_text(prettify_tables("\n".join(lines)) + "\n", encoding="utf-8")
    (RESULTS / "custom_eval_breakdown.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[done] -> {RESULTS/'custom_eval_breakdown.md'}")


if __name__ == "__main__":
    main()

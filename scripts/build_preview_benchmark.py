#!/usr/bin/env python3
"""Turn the first answerable row from each benchmark annotation file into ONE
normalised, answerable benchmark JSONL so every model can be run across every benchmark that
has a (question, answer) — the cross-benchmark "preview matrix" set.

Benchmarks whose ground truth is a *structure* (tables/KIE/parsing) rather than a Q/A pair
are skipped here (they need task-specific scoring, not the generic VQA loop) and reported.

    python scripts/build_preview_benchmark.py            # -> examples/benchmarks/all_preview.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

BENCH = Path("examples/benchmarks")
TRANSCRIBE_Q = "Read and transcribe all the text in the image."
# Small VLMs are verbose; short-answer benchmarks are run (e.g. in VLMEvalKit) with an explicit
# brevity instruction so the answer can be matched. Transcription tasks get no such suffix.
CONCISE = " Answer concisely with only the value, no explanation."


def _extract(key: str, gt: dict, metric: str):
    """Return (question, answers, metric) or None if not answerable by the generic loop."""
    # transcription tasks: gt has 'text'/'latex_formula'
    if "text" in gt and isinstance(gt["text"], str):
        return TRANSCRIBE_Q, [gt["text"]], "anls"
    if "latex_formula" in gt and isinstance(gt["latex_formula"], str):
        return "Write the LaTeX for the formula in the image.", [gt["latex_formula"]], "anls"
    # Q/A tasks
    q = gt.get("question") or gt.get("query")
    if isinstance(gt.get("questions"), list) and gt["questions"]:
        q = gt["questions"][0]
    if not q:
        return None
    # answers may be a list, single string, or under 'answer'/'label'
    ans = gt.get("answers")
    if ans is None:
        ans = gt.get("answer")
    if ans is None:
        ans = gt.get("label")
    if isinstance(gt.get("answers"), list) and gt["answers"] and isinstance(gt["answers"][0], list):
        ans = gt["answers"][0]  # ocrvqa: list-of-lists
    if ans is None:
        return None
    answers = ans if isinstance(ans, list) else [ans]
    answers = [str(a) for a in answers if a is not None and str(a) != ""]
    if not answers:
        return None
    # MCQ: AI2D's 'answer' is an INDEX into 'options' (not the text) -> map it to the option.
    opts = gt.get("options") or gt.get("choices")
    if isinstance(opts, list) and opts:
        mapped = []
        for a in answers:
            if a.isdigit() and int(a) < len(opts):
                mapped.append(str(opts[int(a)]))
            else:
                mapped.append(a)
        answers = mapped
        q = q + "  Options: " + " | ".join(str(o) for o in opts)
    return q + CONCISE, answers, metric


def main() -> None:
    samples: list[Sample] = []
    skipped: list[str] = []
    from docvlm_eval.benchmarks.catalog import load_catalog

    catalog = {entry["key"]: entry for entry in load_catalog()}
    for annotation_file in sorted(BENCH.glob("*/annotations.jsonl")):
        key = annotation_file.parent.name
        rows = annotation_file.read_text(encoding="utf-8").splitlines()
        if not rows:
            skipped.append(f"{key} (no annotations)")
            continue
        row = json.loads(rows[0])
        img = annotation_file.parent / row["image"]
        if not img.exists():
            skipped.append(f"{key} (no downloaded image)")
            continue
        gt = row.get("ground_truth", {})
        entry = catalog.get(key, {})
        metric = str(entry.get("metric") or "anls").lower()
        metric = {"relaxed_acc": "relaxed_acc"}.get(metric, "anls" if "anls" in metric else
                  "relaxed_acc" if "relaxed" in metric else
                  "ocrbench" if "ocrbench" in metric else
                  "exact" if "exact" in metric or "acc" in metric else "anls")
        got = _extract(key, gt, metric)
        if got is None:
            skipped.append(f"{key} (structure GT - not generic-answerable)")
            continue
        q, answers, m = got
        samples.append(Sample(
            sample_id=key, image_path=str(img), question=q, answers=answers,
            answer_type=key, metric=m,
            meta={"benchmark": key, "category": entry.get("category")},
        ))
    out = BENCH / "all_preview.jsonl"
    save_jsonl(samples, out)
    print(f"[done] {len(samples)} answerable benchmarks -> {out}")
    if skipped:
        print("[skip] not generic-answerable (need task-specific scoring):")
        for s in skipped:
            print("   -", s)


if __name__ == "__main__":
    main()

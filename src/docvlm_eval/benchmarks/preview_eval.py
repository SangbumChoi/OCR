"""Turn the committed 10-sample previews (``data/benchmarks/<key>/samples.jsonl``, raw HF GT) into
eval-pipeline :class:`Sample` records — **offline, reproducible, no network**.

``fetch_many`` saved the raw HF record per image; here we map each benchmark's fields to a
(question, answers, metric) using a small per-benchmark spec, so models can be evaluated on a
10-sample slice of every benchmark with the normal harness (``run_matrix`` / ``docvlm-eval``).

Benchmarks whose fetched split has no usable gold (e.g. ST-VQA test) or no flat GT (omnidocbench,
funsd tokens, cord/charxiv nested structures) are skipped — they are listed by ``build_preview_eval``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Sample

# Per-benchmark extraction. `q`=question field; `fixed`=fixed instruction when there is no NL
# question (transcription/structure tasks); `a`=ordered answer fields (first non-empty wins);
# flags: mc=multiple-choice (answer indexes `options`), yesno=map 1/0 to yes/no, multi=parallel
# questions[]/answers[] -> several samples.
SPEC: dict[str, dict] = {
    "docvqa":      {"q": "question", "a": ["answers"], "metric": "anls"},
    "infovqa":     {"q": "question", "a": ["answers"], "metric": "anls"},
    "textvqa":     {"q": "question", "a": ["answers"], "metric": "anls"},
    "ocrvqa":      {"q": "questions", "a": ["answers"], "metric": "anls", "multi": True},
    "ai2d":        {"q": "question", "a": ["answer"], "metric": "exact", "mc": "options"},
    "chartqa":     {"q": "question", "a": ["answer"], "metric": "relaxed_acc"},
    "mathvista":   {"q": "question", "a": ["answer"], "metric": "exact"},
    "ocrbench":    {"q": "question", "a": ["answer"], "metric": "ocrbench", "at": "question_type"},
    "ocrbench_v2": {"q": "question", "a": ["answers"], "metric": "ocrbench", "at": "type"},
    "pope":        {"q": "question", "a": ["answer"], "metric": "exact", "yesno": True},
    "hallusionbench": {"q": "question", "a": ["gt_answer"], "metric": "exact", "yesno": True},
    "iam":         {"fixed": "Transcribe the handwritten text.", "a": ["text"], "metric": "ned"},
    "sroie":       {"fixed": "Transcribe all the text in the receipt.", "a": ["text"], "metric": "ned"},
    "im2latex":    {"fixed": "Write the formula in the image as LaTeX.", "a": ["latex_formula"], "metric": "ned"},
    "latexocr":    {"fixed": "Write the formula in the image as LaTeX.", "a": ["text"], "metric": "ned"},
    "pubtabnet":   {"fixed": "Convert the table to HTML (a <table> with <tr>/<td>).",
                    "a": ["html_table"], "metric": "teds"},
}
CONCISE = " Answer concisely, no explanation."


def _as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None and str(x) != ""]
    return [str(v)] if str(v) != "" else []


def _yesno(golds: list[str]) -> list[str]:
    out = []
    for g in golds:
        gl = g.strip().lower()
        out.append("yes" if gl in ("1", "yes", "true") else "no" if gl in ("0", "no", "false") else gl)
    return out


def _first_answer(gt: dict, fields: list[str]) -> list[str]:
    for f in fields:
        v = _as_list(gt.get(f))
        if v:
            return v
    return []


def case_samples(key: str, gt: dict, image_path: str, spec: dict, idx: int) -> list[Sample]:
    metric = spec["metric"]
    at = gt.get(spec["at"]) if spec.get("at") else key
    concise = "" if metric in ("ned", "teds", "grounding") else CONCISE

    if spec.get("multi"):  # parallel questions[]/answers[]
        qs, ans = _as_list(gt.get(spec["q"])), _as_list(gt.get(spec["a"][0]))
        out = []
        for j, (q, a) in enumerate(zip(qs, ans)):
            if q and a:
                out.append(Sample(f"{key}_{idx}_{j}", image_path, q + concise, [a], key, metric,
                                  {"benchmark": key}))
            if j >= 2:  # cap per image to keep the slice small
                break
        return out

    golds = _first_answer(gt, spec["a"])
    if spec.get("yesno"):
        golds = _yesno(golds)
    if spec.get("mc"):  # answer indexes options
        opts = _as_list(gt.get(spec["mc"]))
        try:
            golds = [opts[int(golds[0])]]
        except (ValueError, IndexError, TypeError):
            pass
    if not golds:
        return []

    if spec.get("fixed"):
        question = spec["fixed"]
    else:
        q = gt.get(spec["q"])
        if not q:
            return []
        question = str(q)
        if spec.get("mc"):
            question += " Options: " + ", ".join(_as_list(gt.get(spec["mc"]))) + "."
        question += concise
    return [Sample(f"{key}_{idx}", image_path, question, golds, str(at or key), metric,
                   {"benchmark": key})]


def build_preview_eval(bench_root: str | Path) -> tuple[list[Sample], dict]:
    """Convert every ``<key>/samples.jsonl`` with a known spec into Samples. Returns (samples, stats)."""
    root = Path(bench_root)
    samples: list[Sample] = []
    stats = {"benchmarks": 0, "skipped": []}
    for jsonl in sorted(root.glob("*/samples.jsonl")):
        key = jsonl.parent.name
        spec = SPEC.get(key)
        if not spec:
            stats["skipped"].append(key)
            continue
        n_before = len(samples)
        for i, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            img = jsonl.parent / row["image"]  # "samples/NN.jpg"
            samples.extend(case_samples(key, row.get("ground_truth", {}), str(img), spec, i))
        if len(samples) > n_before:
            stats["benchmarks"] += 1
        else:
            stats["skipped"].append(key)
    return samples, stats

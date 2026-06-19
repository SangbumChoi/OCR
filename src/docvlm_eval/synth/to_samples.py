"""Turn the synth ground truth (``gt.json``) into eval-pipeline :class:`Sample` records.

The synth GT is uniform (see ``patterns.DocBuilder``), so one converter maps every case into
answerable Samples for the existing pipeline:

  * ``qa``         -> a Sample per (question, answers, metric, answer_type)
  * ``spotting``   -> a grounding Sample per box (text-prompted "return [x1,y1,x2,y2]")
  * ``table_html`` -> a TEDS Sample
  * ``probes``     -> a Sample per control question (abstain / consistency / direction / order),
                      with answer lists chosen so the existing string metrics score them loosely

This is what lets ``run_matrix`` evaluate the realistic cases alongside the other benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Sample

# Acceptable surface forms for an abstention (anti-hallucination target).
ABSTAIN_OK = ["[redacted]", "redacted", "not present", "not shown", "not legible", "n/a", "na",
              "none", "unknown", "cannot determine", "can't tell", "no", "absent"]


def _probe_answers(probe: dict) -> tuple[list[str], str]:
    """Map a probe's free-text `expected` to an answer list + metric the string metrics can score."""
    kind = probe.get("kind", "")
    exp = (probe.get("expected") or "").lower()
    if kind == "abstain":
        return ABSTAIN_OK, "anls"
    if kind == "direction":
        if "right" in exp or "rtl" in exp:
            return ["right-to-left", "rtl"], "anls"
        if "vert" in exp:
            return ["vertical"], "anls"
        return ["left-to-right", "ltr", "horizontal"], "anls"
    if kind == "consistency":
        return (["yes", "they agree", "agree"], "anls") if exp.startswith("yes") \
            else (["no", "they disagree", "disagree"], "anls")
    # order or anything else: keep the descriptive expectation, score with anls
    return [probe.get("expected", "")], "anls"


def case_to_samples(gt: dict, image_path: str, prefix: str, *,
                    include_probes: bool = True) -> list[Sample]:
    """Convert one case's GT dict + image into a list of Samples (ids prefixed by `prefix`)."""
    out: list[Sample] = []
    base_meta = {"case": prefix, "doc_type": gt.get("type"), "stressors": gt.get("stressors")}

    for i, qa in enumerate(gt.get("qa", [])):
        out.append(Sample(
            f"{prefix}:qa{i}", image_path, qa["question"], list(qa["answers"]),
            qa.get("answer_type", "kie"), qa.get("metric", "anls"),
            {**base_meta, "qa_key": qa.get("key")},
        ))

    size = gt.get("render", {}).get("size_px") or [None, None]
    w, h = size[0], size[1]
    for key, box in gt.get("spotting", {}).items():
        q = (f"Return the bounding box of the {key.replace('_', ' ')} as [x1, y1, x2, y2] in "
             f"pixel coordinates. The image is {w}x{h} pixels.")
        ans = f"{box[0]},{box[1]},{box[2]},{box[3]};{w},{h}"
        out.append(Sample(f"{prefix}:spot:{key}", image_path, q, [ans], "grounding", "grounding",
                          {**base_meta, "box": box, "size": [w, h]}))

    if gt.get("table_html"):
        out.append(Sample(
            f"{prefix}:table", image_path,
            "Convert the table to HTML (a <table> with <tr>/<td>).",
            [gt["table_html"]], "table", "teds", dict(base_meta)))

    if include_probes:
        for i, p in enumerate(gt.get("probes", [])):
            answers, metric = _probe_answers(p)
            out.append(Sample(
                f"{prefix}:probe:{p.get('kind','x')}{i}", image_path, p["question"], answers,
                f"probe:{p.get('kind','x')}", metric, {**base_meta, "probe": p}))
    return out


def load_case_dir(case_dir: str | Path, *, variant: str = "clean",
                  include_probes: bool = True) -> list[Sample]:
    """Load one ``realistic_cases/<key>/`` directory (uses ``<variant>.png``: clean|degraded)."""
    case_dir = Path(case_dir)
    gt = json.loads((case_dir / "gt.json").read_text(encoding="utf-8"))
    img = case_dir / f"{variant}.png"
    if not img.exists():
        img = case_dir / "clean.png"
    return case_to_samples(gt, str(img), f"{case_dir.name}", include_probes=include_probes)


def load_realistic_samples(root: str | Path, *, variant: str = "clean",
                           include_probes: bool = True) -> list[Sample]:
    """Load every case under ``data/probes/realistic_cases/`` into one Sample list.

    Handles both layouts: a case dir with ``gt.json`` directly (``--count 1``) and per-variant
    subdirs (``<key>/0000/gt.json`` from ``--count N``). The sample-id prefix is the path
    relative to ``root`` (``invoice`` or ``invoice_0003``) so ids stay unique across variants.
    """
    root = Path(root)
    samples: list[Sample] = []
    for gt_path in sorted(root.rglob("gt.json")):
        case_dir = gt_path.parent
        prefix = str(case_dir.relative_to(root)).replace("/", "_")
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        img = case_dir / f"{variant}.png"
        if not img.exists():
            img = case_dir / "clean.png"
        samples.extend(case_to_samples(gt, str(img), prefix, include_probes=include_probes))
    return samples

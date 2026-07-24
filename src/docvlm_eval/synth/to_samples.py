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
        expected = str(probe.get("expected") or "").strip()
        answers = [*ABSTAIN_OK, *([expected] if expected else [])]
        return list(dict.fromkeys(answers)), "anls"
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


def _degradation_label(gt: dict, render_variant: str | None) -> str:
    if render_variant == "clean":
        return "clean"
    if render_variant == "degraded":
        degradation = gt.get("degradation") or {}
        return str(
            degradation.get("preset")
            or gt.get("degraded_preset")
            or "degraded"
        )
    return "unknown"


def _box_page_index(box: object, render: dict) -> int | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        center_x = (float(box[0]) + float(box[2])) / 2
        center_y = (float(box[1]) + float(box[3])) / 2
    except (TypeError, ValueError):
        return None
    origins = render.get("page_origins_px") or []
    sizes = render.get("page_sizes_px") or []
    for index, (origin, size) in enumerate(zip(origins, sizes)):
        if (
            len(origin) >= 2
            and len(size) >= 2
            and float(origin[0]) <= center_x <= float(origin[0]) + float(size[0])
            and float(origin[1]) <= center_y <= float(origin[1]) + float(size[1])
        ):
            return index
    return 0 if int(render.get("rendered_page_count") or 1) == 1 else None


def case_to_samples(
    gt: dict,
    image_path: str,
    prefix: str,
    *,
    include_probes: bool = True,
    render_variant: str | None = None,
) -> list[Sample]:
    """Convert one case's GT dict + image into a list of Samples (ids prefixed by `prefix`)."""
    out: list[Sample] = []
    render = gt.get("render") or {}
    size = render.get("size_px") or [None, None]
    document_family = (
        (gt.get("semantic_graph") or {}).get("template_family")
        or gt.get("doc_type")
        or gt.get("type")
        or "unknown"
    )
    overlays = render.get("overlays") or []
    base_meta = {
        "case": prefix,
        "doc_type": gt.get("type"),
        "document_family": str(document_family),
        "stressors": gt.get("stressors"),
        "size": size,
        "language": (gt.get("languages") or ["und"])[0],
        "degradation": _degradation_label(gt, render_variant),
        "render_variant": render_variant or "unknown",
        "split": gt.get("split", "synthetic"),
        "difficulty": gt.get("difficulty"),
        "template_family": (gt.get("semantic_graph") or {}).get("template_family"),
        "counterfactual": gt.get("counterfactual"),
        "overlay_types": [
            str(mark.get("kind"))
            for mark in overlays
            if isinstance(mark, dict) and mark.get("kind")
        ],
        "overlay_count": len(overlays),
        "overlay_fingerprint": (gt.get("render") or {}).get(
            "overlay_fingerprint"
        ),
        "page_count": int(render.get("rendered_page_count") or 1),
        "page_mode": str(render.get("page_mode") or "first"),
        "page_origins_px": render.get("page_origins_px") or [[0, 0]],
        "page_sizes_px": render.get("page_sizes_px") or [size],
    }

    for i, qa in enumerate(gt.get("qa", [])):
        boxes = qa.get("evidence_bboxes") or []
        evidence_count = len(boxes or qa.get("evidence_keys") or [])
        if qa.get("box") and not evidence_count:
            evidence_count = 1
        meta = {
            **base_meta,
            "qa_key": qa.get("key"),
            "language": (qa.get("languages") or [base_meta["language"]])[0],
            "evidence_count": evidence_count,
        }
        graph_query_id = qa.get("graph_query_id")
        if graph_query_id:
            meta["graph_query_id"] = graph_query_id
        counterfactual = gt.get("counterfactual")
        if (
            isinstance(counterfactual, dict)
            and graph_query_id
            and str(qa.get("answer_type", "")).startswith("H-")
        ):
            pair = f"{counterfactual['pair_id']}:{graph_query_id}"
            meta.update(
                {
                    "counterfactual_pair": pair,
                    "counterfactual_group": pair,
                    "counterfactual_role": counterfactual["role"],
                    "control": counterfactual["role"] == "edited",
                }
            )
        if qa.get("rationale"):
            meta["rationale"] = qa["rationale"]
        if qa.get("box"):
            meta["box"] = qa["box"]
        if boxes:
            meta["boxes"] = boxes
        spatial_boxes = list(boxes)
        if qa.get("box"):
            spatial_boxes.append(qa["box"])
        evidence_pages = sorted(
            {
                page
                for box in spatial_boxes
                if (page := _box_page_index(box, render)) is not None
            }
        )
        if evidence_pages:
            meta["evidence_pages"] = evidence_pages
            meta["cross_page_evidence"] = len(evidence_pages) > 1
        out.append(Sample(
            f"{prefix}:qa{i}", image_path, qa["question"], list(qa["answers"]),
            qa.get("answer_type", "kie"), qa.get("metric", "anls"),
            meta,
        ))

    w, h = size[0], size[1]
    for key, box in gt.get("spotting", {}).items():
        q = (f"Return the bounding box of the {key.replace('_', ' ')} as [x1, y1, x2, y2] in "
             f"pixel coordinates. The image is {w}x{h} pixels.")
        ans = f"{box[0]},{box[1]},{box[2]},{box[3]};{w},{h}"
        out.append(Sample(f"{prefix}:spot:{key}", image_path, q, [ans], "grounding", "grounding",
                          {
                              **base_meta,
                              "box": box,
                              "size": [w, h],
                              "evidence_count": 1,
                              "evidence_pages": [
                                  page
                                  for page in [_box_page_index(box, render)]
                                  if page is not None
                              ],
                          }))

    if gt.get("table_html"):
        out.append(Sample(
            f"{prefix}:table", image_path,
            "Convert the table to HTML (a <table> with <tr>/<td>).",
            [gt["table_html"]], "table", "teds",
            {**base_meta, "evidence_count": 0}))

    if include_probes:
        for i, p in enumerate(gt.get("probes", [])):
            answers, metric = _probe_answers(p)
            kind = str(p.get("kind", "x"))
            out.append(Sample(
                f"{prefix}:probe:{kind}{i}", image_path, p["question"], answers,
                f"probe:{kind}", metric,
                {
                    **base_meta,
                    "evidence_count": 0,
                    "probe": p,
                    "abstain_expected": kind == "abstain",
                }))
    return out


def _validate_counterfactual_pairs(samples: list[Sample]) -> None:
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        group = sample.meta.get("counterfactual_group")
        if group:
            groups.setdefault(str(group), []).append(sample)
    for grouped in groups.values():
        roles = {
            str(sample.meta.get("counterfactual_role"))
            for sample in grouped
        }
        answers = {
            tuple(str(answer).strip() for answer in sample.answers)
            for sample in grouped
        }
        eligible = (
            len(grouped) == 2
            and roles == {"factual", "edited"}
            and len(answers) == 2
        )
        if eligible:
            for sample in grouped:
                sample.meta["counterfactual_eligible"] = True
            continue
        for sample in grouped:
            sample.meta["counterfactual_eligible"] = False
            sample.meta.pop("counterfactual_group", None)
            sample.meta.pop("control", None)


def load_case_dir(case_dir: str | Path, *, variant: str = "clean",
                  include_probes: bool = True) -> list[Sample]:
    """Load one ``realistic_cases/<key>/`` directory (uses ``<variant>.png``: clean|degraded)."""
    case_dir = Path(case_dir)
    gt = json.loads((case_dir / "gt.json").read_text(encoding="utf-8"))
    img = case_dir / f"{variant}.png"
    actual_variant = variant
    if not img.exists():
        img = case_dir / "clean.png"
        actual_variant = "clean"
    return case_to_samples(
        gt,
        str(img),
        f"{case_dir.name}",
        include_probes=include_probes,
        render_variant=actual_variant,
    )


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
        actual_variant = variant
        if not img.exists():
            img = case_dir / "clean.png"
            actual_variant = "clean"
        samples.extend(
            case_to_samples(
                gt,
                str(img),
                prefix,
                include_probes=include_probes,
                render_variant=actual_variant,
            )
        )
    _validate_counterfactual_pairs(samples)
    return samples

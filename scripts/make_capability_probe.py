#!/usr/bin/env python3
"""Build the CUSTOM capability probe — a small, fully-controlled benchmark that isolates the
capability axes we care about for document VLMs (see report/capability_axes.md):

  1. text-recognition   : read an exact printed string
  2. kie-localized      : extract one field's value from a single region (clear KIE answer)
  3. integrative-sum    : compute a value by combining several regions (sum of line items)
  4. integrative-rel    : reason over relationships between regions (which item is largest)
  5. chart-dependent    : read a value off a chart (can't be answered from text alone)
  6. location-grounding : return the bounding box of a named element (spatial understanding)

Because the images are rendered here, the ground truth — including exact pixel boxes for the
grounding task — is known precisely. Output:
    data/benchmarks/capability_probe/images/*.png
    data/benchmarks/capability_probe/capability.jsonl   (normalised Sample records)
    data/benchmarks/capability_probe/sample.{png,json}  (catalog/index preview)

    python scripts/make_capability_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

OUT = Path("data/benchmarks/capability_probe")
IMG = OUT / "images"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


def render_invoice():
    """Return (image, boxes) for an invoice with known fields, line items, and a total box."""
    W, H = 820, 600
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    boxes = {}

    d.text((40, 30), "INVOICE  #INV-2025-0042", fill="black", font=font(30, True))
    d.text((40, 80), "Vendor: Acme Corporation", fill="black", font=font(22))
    d.text((40, 110), "Date: 2025-06-14", fill="black", font=font(22))

    # line items
    items = [("Widget A", 45.00), ("Gadget B", 80.00), ("Cable C", 20.50)]
    y = 180
    d.text((40, y - 35), "Item", fill="black", font=font(20, True))
    d.text((520, y - 35), "Price", fill="black", font=font(20, True))
    for name, price in items:
        d.text((40, y), name, fill="black", font=font(22))
        d.text((520, y), f"${price:.2f}", fill="black", font=font(22))
        y += 40

    total = sum(p for _, p in items)  # 145.50
    ty = y + 30
    total_str = f"TOTAL: ${total:.2f}"
    d.text((40, ty), total_str, fill=(150, 0, 0), font=font(26, True))
    bbox = d.textbbox((40, ty), total_str, font=font(26, True))  # exact pixel box
    boxes["total"] = [bbox[0], bbox[1], bbox[2], bbox[3]]
    return im, (W, H), boxes, items, total


def render_chart():
    W, H = 640, 460
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((220, 20), "Monthly Sales", fill="black", font=font(24, True))
    bars = [("A", 30), ("B", 70), ("C", 45)]
    base_y, x = 400, 120
    for name, val in bars:
        h = val * 4
        d.rectangle([x, base_y - h, x + 90, base_y], fill=(70, 120, 200))
        d.text((x + 30, base_y + 8), name, fill="black", font=font(22))
        d.text((x + 20, base_y - h - 28), str(val), fill="black", font=font(20))
        x += 160
    d.line([100, base_y, W - 30, base_y], fill="black", width=2)
    return im, (W, H), bars


CONCISE = " Answer concisely with only the value, no explanation."


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    inv, inv_sz, boxes, items, total = render_invoice()
    chart, ch_sz, bars = render_chart()
    inv.save(IMG / "invoice.png")
    chart.save(IMG / "chart.png")
    inv_p, ch_p = str(IMG / "invoice.png"), str(IMG / "chart.png")
    biggest = max(items, key=lambda x: x[1])[0]

    samples = [
        Sample("cap_text", inv_p, "Read the invoice number." + CONCISE, ["INV-2025-0042"],
               "text-recognition", "anls", {"axis": "text", "prompt_strategy": "direct read"}),
        Sample("cap_kie", inv_p, "What is the vendor name?" + CONCISE, ["Acme Corporation"],
               "kie-localized", "anls", {"axis": "text", "prompt_strategy": "single-field KIE"}),
        Sample("cap_integ_sum", inv_p,
               "Add up the prices of all line items and give the total." + CONCISE, [f"{total:.2f}"],
               "integrative-sum", "relaxed_acc",
               {"axis": "text+reasoning", "prompt_strategy": "multi-region aggregation"}),
        Sample("cap_integ_rel", inv_p,
               "Which line item has the highest price?" + CONCISE, [biggest],
               "integrative-rel", "exact",
               {"axis": "text+reasoning", "prompt_strategy": "cross-region comparison"}),
        Sample("cap_chart", ch_p, "What is the value of bar B?" + CONCISE, ["70"],
               "chart-dependent", "relaxed_acc",
               {"axis": "chart", "prompt_strategy": "chart value read"}),
        Sample("cap_ground", inv_p,
               "Return the bounding box of the TOTAL field as [x1, y1, x2, y2] in pixel "
               f"coordinates. The image is {inv_sz[0]}x{inv_sz[1]} pixels.",
               [f"{boxes['total'][0]},{boxes['total'][1]},{boxes['total'][2]},{boxes['total'][3]};{inv_sz[0]},{inv_sz[1]}"],
               "location-grounding", "grounding",
               {"axis": "location", "prompt_strategy": "text-prompted box (fair-comparison normalisation)"}),
    ]
    save_jsonl(samples, OUT / "capability.jsonl")

    # catalog/index preview (one representative)
    inv.save(OUT / "sample.png")
    (OUT / "sample.json").write_text(json.dumps({
        "benchmark": "capability_probe",
        "name": "Custom capability probe",
        "category": "11. Custom capability axes",
        "metric": "anls / relaxed_acc / exact / grounding",
        "purpose": "Isolate document-VLM capability axes: text recognition, localized KIE, "
                   "integrative reasoning (sum/relations), chart reading, and spatial grounding.",
        "source": "SYNTHETIC (scripts/make_capability_probe.py); GT incl. exact pixel boxes",
        "ground_truth": {"n_samples": len(samples),
                         "axes": sorted({s.answer_type for s in samples})},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(samples)} capability samples -> {OUT/'capability.jsonl'}")
    for s in samples:
        print(f"   {s.answer_type:<20} metric={s.metric:<12} gold={s.answers[0][:40]}")


if __name__ == "__main__":
    main()

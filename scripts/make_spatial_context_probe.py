#!/usr/bin/env python3
"""Hypothesis-driven probes for SPATIAL and CONTEXT understanding (see
docs/report/spatial_context_probes.md). The design goal is *falsifiability*: each hypothesis comes
with a CONTROL that rules out a shortcut (language prior, guessing, hallucination, position
bias). Images are rendered so ground truth (incl. boxes) is exact.

Output: data/probes/spatial_context_probe/{images/, probe.jsonl, sample.png, sample.json}
    python scripts/make_spatial_context_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.fonts import load_font  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

OUT = Path("data/probes/spatial_context_probe")
IMG = OUT / "images"
CONCISE = " Answer concisely, no explanation."


def fnt(s, b=False):
    return load_font(s, b)


def blank(w=800, h=600):
    im = Image.new("RGB", (w, h), "white")
    return im, ImageDraw.Draw(im)


def save(name, im):
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f"{name}.png"
    im.save(p)
    return str(p)


def S(id, img, q, ans, axis, metric, hyp, control=False, **meta):
    return Sample(id, img, q + (CONCISE if metric != "grounding" else ""), ans, axis, metric,
                  {"hypothesis": hyp, "control": control, **meta})


def build():
    samples = []

    # ---------------- SPATIAL ----------------
    # H-quadrant: marker in each quadrant -> accuracy vs 25% chance, position-bias-controlled
    W, H = 800, 600
    quads = {"top-left": (120, 120), "top-right": (560, 120),
             "bottom-left": (120, 460), "bottom-right": (560, 460)}
    for q, (x, y) in quads.items():
        im, d = blank(W, H)
        for fx, fy, t in [(120, 120, "alpha"), (560, 120, "beta"),
                          (120, 460, "gamma"), (560, 460, "delta")]:
            d.text((fx, fy), t, fill=(180, 180, 180), font=fnt(28))
        d.text((x, y), "ZEBRA", fill="black", font=fnt(34, True))  # overwrites filler at q
        p = save(f"quad_{q}", im)
        samples.append(S(f"sp_quad_{q}", p,
                         "Which quadrant contains the word ZEBRA? Answer one of: "
                         "top-left, top-right, bottom-left, bottom-right.",
                         [q], "spatial-quadrant", "exact", "H1: absolute region id"))

    # H-relpos with CONTROL: TOTAL at bottom (prior-consistent) vs at top (counterfactual)
    for variant, total_top in [("normal", False), ("counterfactual", True)]:
        im, d = blank(800, 520)
        items_y = 90 if not total_top else 230
        total_y = 430 if not total_top else 70
        d.text((40, total_y), "TOTAL: $145.50", fill=(150, 0, 0), font=fnt(28, True))
        for i, it in enumerate(["Widget A   $45.00", "Gadget B   $80.00", "Cable C   $20.50"]):
            d.text((40, items_y + i * 40), it, fill="black", font=fnt(24))
        gold = "above" if total_top else "below"
        p = save(f"relpos_{variant}", im)
        samples.append(S(f"sp_relpos_{variant}", p,
                         "Is the TOTAL line ABOVE or BELOW the list of items? Answer 'above' or 'below'.",
                         [gold], "spatial-relative", "exact",
                         "H2: relative position from perception (control falsifies prior)",
                         control=(variant == "counterfactual")))

    # H-box-tracking: same word ANCHOR at 3 vertical positions -> does the box follow?
    for tag, y in [("top", 60), ("mid", 280), ("bot", 500)]:
        im, d = blank(800, 600)
        txt = "ANCHOR"
        d.text((300, y), txt, fill="black", font=fnt(34, True))
        bbox = d.textbbox((300, y), txt, font=fnt(34, True))
        p = save(f"anchor_{tag}", im)
        samples.append(S(f"sp_box_{tag}", p,
                         "Return the bounding box of the word ANCHOR as [x1, y1, x2, y2] in "
                         "pixel coordinates. The image is 800x600 pixels.",
                         [f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]};800,600"],
                         "spatial-box-tracking", "grounding",
                         "H3: box tracks true position (constant-box prior fails this)",
                         true_cy=(bbox[1] + bbox[3]) // 2))

    # ---------------- CONTEXT ----------------
    def invoice(total_str, with_discount=False, subtotal=False):
        im, d = blank(800, 560)
        d.text((40, 30), "INVOICE  #INV-77", fill="black", font=fnt(26, True))
        for i, it in enumerate(["Widget A   $45.00", "Gadget B   $80.00", "Cable C   $20.50"]):
            d.text((40, 110 + i * 40), it, fill="black", font=fnt(24))
        y = 250
        if subtotal:
            d.text((40, y), "Subtotal: $145.50", fill="black", font=fnt(24)); y += 40
        if with_discount:
            d.text((40, y), "Discount: $10.00", fill="black", font=fnt(24)); y += 40
        d.text((40, y + 20), f"TOTAL: {total_str}", fill=(150, 0, 0), font=fnt(26, True))
        return im

    # C1 consistency-check (control pair): correct vs wrong total
    for variant, total, gold in [("consistent", "$145.50", "yes"), ("inconsistent", "$200.00", "no")]:
        p = save(f"consistency_{variant}", invoice(total))
        samples.append(S(f"ctx_consistency_{variant}", p,
                         "Do the line items add up to the printed TOTAL? Answer yes or no.",
                         [gold], "context-consistency", "exact",
                         "C1: cross-region numeric consistency (inconsistent variant resists rubber-stamping)",
                         control=(variant == "inconsistent")))

    # C2 absence / anti-hallucination: ask for a field that does NOT exist
    p = save("absence", invoice("$145.50", with_discount=False))
    samples.append(S("ctx_absence", p,
                     "What is the discount amount on this invoice? If there is no discount, answer 'none'.",
                     ["none"], "context-absence", "exact",
                     "C2: absent-field -> should abstain, not hallucinate a number", control=True))

    # C3 distractor: Subtotal present; ask for Total (must not return subtotal)
    p = save("distractor", invoice("$135.50", subtotal=True))
    samples.append(S("ctx_distractor", p,
                     "What is the TOTAL amount (not the subtotal)?",
                     ["135.50"], "context-distractor", "relaxed_acc",
                     "C3: pick the right field among look-alikes"))

    # C4 cross-reference with counterfactual sensitivity: name in header -> amount in table
    table = [("Bob", "$45.00"), ("Alice", "$80.00")]
    for name, amt in table:
        im, d = blank(800, 420)
        d.text((40, 30), f"Bill to: {name}", fill="black", font=fnt(26, True))
        d.text((40, 110), "Name        Amount due", fill="black", font=fnt(22, True))
        for i, (n, a) in enumerate(table):
            d.text((40, 150 + i * 40), f"{n}        {a}", fill="black", font=fnt(24))
        p = save(f"xref_{name.lower()}", im)
        samples.append(S(f"ctx_xref_{name.lower()}", p,
                         f"How much does the person on the 'Bill to' line owe? (Use the table.)",
                         [amt.replace("$", "")], "context-xref", "relaxed_acc",
                         "C4: link header->table; counterfactual name flips the answer",
                         control=(name == "Alice")))

    save_jsonl(samples, OUT / "probe.jsonl")
    # catalog preview
    rep = invoice("$145.50")
    rep.save(OUT / "sample.png")
    (OUT / "sample.json").write_text(json.dumps({
        "benchmark": "spatial_context_probe", "name": "Spatial & context understanding probe",
        "category": "F1. Custom capability axes", "metric": "exact / relaxed_acc / grounding",
        "purpose": "Falsifiable hypothesis tests for spatial (quadrant, relative position, box "
                   "tracking) and context (consistency, absence/anti-hallucination, distractor, "
                   "cross-reference) understanding, each paired with a control that rules out a shortcut.",
        "source": "SYNTHETIC (scripts/make_spatial_context_probe.py)",
        "ground_truth": {"n_samples": len(samples),
                         "axes": sorted({s.answer_type for s in samples})},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(samples)} samples -> {OUT/'probe.jsonl'}")
    for s in samples:
        print(f"   {s.sample_id:<26} {s.answer_type:<22} gold={s.answers[0][:28]} ctrl={s.meta['control']}")


if __name__ == "__main__":
    build()

#!/usr/bin/env python3
"""Build the proposed CUSTOM evaluation benchmark (data/probes/custom_eval/).

This is *our* evaluation format: every sample carries rich metadata so results can be sliced by
the axes that matter for real-world document AI — content class, language, rotation, reading
direction — and each item declares the metric appropriate to its class (see the folder README
for the rationale). Rendered with exact ground truth incl. spotting boxes.

Axes covered:
  * content_class : text / table / formula / chart / qr / barcode / stamp / logo / figure
  * language      : en / ko / ja / zh   (+ vertical CJK for reading-direction)
  * rotation_deg  : 0 / 15 / 90 / 180 / 270   (robustness; retention vs 0deg)
  * reading_direction : ltr / vertical
  * spotting      : gold bounding box for localisable items (basis-of-extraction)

    python scripts/make_custom_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.fonts import have_cjk, load_cjk_font, load_font  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

OUT = ROOT / "data" / "probes" / "custom_eval"
IMG = OUT / "images"
CONCISE = " Answer concisely, no explanation."
samples: list[Sample] = []


def save_img(name: str, im: Image.Image) -> str:
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f"{name}.png"
    im.convert("RGB").save(p)
    return str(p)


def add(sid, img, q, ans, cls, metric, *, language="en", rotation=0,
        reading="ltr", spotting=None, needs_reasoning=False, add_concise=True):
    samples.append(Sample(
        sid, img, q + (CONCISE if add_concise and metric not in ("grounding", "teds") else ""),
        ans if isinstance(ans, list) else [ans], cls, metric,
        {"content_class": cls, "language": language, "rotation_deg": rotation,
         "reading_direction": reading, "spotting": spotting, "needs_reasoning": needs_reasoning},
    ))


# ---------------------------------------------------------------- per-class items
def build_text_and_spotting():
    im = Image.new("RGB", (760, 200), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 40), "Invoice No: INV-2025-0042", fill="black", font=load_font(30, True))
    box = d.textbbox((40, 90), "Total: $145.50", font=load_font(28, True))
    d.text((40, 90), "Total: $145.50", fill=(150, 0, 0), font=load_font(28, True))
    p = save_img("text_en", im)
    add("ce_text_read", p, "Read the invoice number.", "INV-2025-0042", "text", "ned")
    add("ce_text_value", p, "What is the total amount?", "145.50", "text", "anls", needs_reasoning=True)
    add("ce_spot_total", p,
        "Return the bounding box of the Total line as [x1, y1, x2, y2] in pixels. The image is 760x200.",
        f"{box[0]},{box[1]},{box[2]},{box[3]};760,200", "text", "grounding",
        spotting=f"{box[0]},{box[1]},{box[2]},{box[3]}")


def build_table():
    im = Image.new("RGB", (520, 220), "white")
    d = ImageDraw.Draw(im)
    rows = [("Item", "Qty", "Price"), ("Widget", "2", "$45"), ("Cable", "5", "$20")]
    for i, (a, b, c) in enumerate(rows):
        y = 30 + i * 50
        d.text((30, y), a, fill="black", font=load_font(24, i == 0))
        d.text((240, y), b, fill="black", font=load_font(24, i == 0))
        d.text((360, y), c, fill="black", font=load_font(24, i == 0))
        d.line([20, y + 40, 500, y + 40], fill="#bbb")
    p = save_img("table_en", im)
    gold_html = ("<table><tr><td>Item</td><td>Qty</td><td>Price</td></tr>"
                 "<tr><td>Widget</td><td>2</td><td>$45</td></tr>"
                 "<tr><td>Cable</td><td>5</td><td>$20</td></tr></table>")
    add("ce_table", p, "Convert the table to HTML (<table> with <tr>/<td>).", gold_html,
        "table", "teds", add_concise=False)


def build_formula():
    im = Image.new("RGB", (520, 130), "white")
    ImageDraw.Draw(im).text((30, 40), "E = m c^2", fill="black", font=load_font(34, True))
    p = save_img("formula_en", im)
    add("ce_formula", p, "Write the formula in LaTeX.", ["E = m c^2", "E=mc^2", "E = mc^{2}"],
        "formula", "ned")


def build_chart():
    im = Image.new("RGB", (480, 360), "white")
    d = ImageDraw.Draw(im)
    for i, (name, val) in enumerate([("A", 30), ("B", 70), ("C", 45)]):
        x = 70 + i * 130
        d.rectangle([x, 300 - val * 3, x + 70, 300], fill=(70, 120, 200))
        d.text((x + 25, 308), name, fill="black", font=load_font(20))
        d.text((x + 15, 300 - val * 3 - 26), str(val), fill="black", font=load_font(18))
    p = save_img("chart_en", im)
    add("ce_chart", p, "What is the value of bar B?", "70", "chart", "relaxed_acc")


def build_qr():
    try:
        import qrcode

        payload = "DOCID-77-2025"
        im = qrcode.make(payload).get_image().resize((220, 220))
        canvas = Image.new("RGB", (300, 300), "white")
        canvas.paste(im, (40, 40))
        p = save_img("qr", canvas)
        add("ce_qr", p, "Read the QR code and output its text content.", payload, "qr", "exact")
    except Exception as e:
        print("[skip] qr:", e)


def build_barcode():
    try:
        import barcode
        from barcode.writer import ImageWriter

        payload = "0123456789"
        bc = barcode.get("code128", payload, writer=ImageWriter())
        tmp = IMG / "barcode"
        IMG.mkdir(parents=True, exist_ok=True)
        path = bc.save(str(tmp))  # writes barcode.png
        im = Image.open(path).convert("RGB").resize((420, 180))
        p = save_img("barcode", im)
        add("ce_barcode", p, "Read the barcode digits.", payload, "barcode", "exact")
    except Exception as e:
        print("[skip] barcode:", e)


def build_stamp_logo_figure():
    im = Image.new("RGB", (600, 300), "white")
    d = ImageDraw.Draw(im)
    d.text((30, 30), "Contract document body text ...", fill="black", font=load_font(22))
    # red circular stamp
    d.ellipse([420, 60, 560, 200], outline=(200, 0, 0), width=5)
    d.text((445, 115), "APPROVED", fill=(200, 0, 0), font=load_font(20, True))
    stamp_box = [420, 60, 560, 200]
    # logo block
    d.rectangle([30, 230, 170, 280], fill=(20, 80, 160))
    d.text((45, 242), "ACME", fill="white", font=load_font(24, True))
    logo_box = [30, 230, 170, 280]
    p = save_img("stamp_logo", im)
    add("ce_stamp", p, "Is there an approval stamp? Give its bounding box [x1,y1,x2,y2]. Image 600x300.",
        f"{stamp_box[0]},{stamp_box[1]},{stamp_box[2]},{stamp_box[3]};600,300", "stamp", "grounding",
        spotting=",".join(map(str, stamp_box)))
    add("ce_logo", p, "Give the bounding box of the company logo [x1,y1,x2,y2]. Image 600x300.",
        f"{logo_box[0]},{logo_box[1]},{logo_box[2]},{logo_box[3]};600,300", "logo", "grounding",
        spotting=",".join(map(str, logo_box)))


# ---------------------------------------------------------------- rotation robustness
def build_rotation():
    base = Image.new("RGB", (600, 160), "white")
    ImageDraw.Draw(base).text((30, 55), "Reference Code: RC-8842", fill="black", font=load_font(34, True))
    for ang in (0, 15, 90, 180, 270):
        im = base.rotate(-ang, expand=True, fillcolor="white")
        p = save_img(f"rot_{ang}", im)
        add(f"ce_rot{ang}_read", p, "Read the reference code.", "RC-8842",
            "text", "ned", rotation=ang)
        add(f"ce_rot{ang}_angle", p,
            "By how many degrees is this page rotated (0, 90, 180, or 270)?",
            str(ang if ang in (0, 90, 180, 270) else 0), "orientation", "exact", rotation=ang)


# ---------------------------------------------------------------- language + reading direction
LANGS = {
    "en": ("Total amount due", "Total amount due"),
    "ko": ("청구 금액 합계", "청구 금액 합계"),
    "ja": ("請求金額合計", "請求金額合計"),
    "zh": ("应付总金额", "应付总金额"),
}


def build_languages():
    for lang, (text, gold) in LANGS.items():
        font = load_font(34, True) if lang == "en" else load_cjk_font(34)
        im = Image.new("RGB", (560, 120), "white")
        ImageDraw.Draw(im).text((30, 40), text, fill="black", font=font)
        p = save_img(f"lang_{lang}", im)
        add(f"ce_lang_{lang}", p, "Transcribe the text in the image.", gold,
            "text", "ned", language=lang)


def build_reading_direction():
    if not have_cjk():
        print("[skip] vertical CJK (no CJK font)")
        return
    for lang, text in (("ja", "上から下"), ("zh", "从上到下")):
        im = Image.new("RGB", (140, 360), "white")
        d = ImageDraw.Draw(im)
        f = load_cjk_font(40)
        for i, ch in enumerate(text):
            d.text((50, 30 + i * 80), ch, fill="black", font=f)
        p = save_img(f"vertical_{lang}", im)
        add(f"ce_dir_{lang}", p,
            "Is the text laid out horizontally or vertically? Answer 'horizontal' or 'vertical'.",
            "vertical", "direction", "exact", language=lang, reading="vertical")
    # a horizontal control
    im = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(im).text((20, 35), "left to right", fill="black", font=load_font(30))
    p = save_img("horizontal_en", im)
    add("ce_dir_en", p,
        "Is the text laid out horizontally or vertically? Answer 'horizontal' or 'vertical'.",
        "horizontal", "direction", "exact", reading="ltr")


def main():
    build_text_and_spotting()
    build_table()
    build_formula()
    build_chart()
    build_qr()
    build_barcode()
    build_stamp_logo_figure()
    build_rotation()
    build_languages()
    build_reading_direction()

    save_jsonl(samples, OUT / "custom_eval.jsonl")
    # catalog preview
    if samples:
        Image.open(samples[0].image_path).save(OUT / "sample.png")
    (OUT / "sample.json").write_text(json.dumps({
        "benchmark": "custom_eval", "name": "Proposed custom evaluation set",
        "category": "F1. Custom capability axes", "metric": "ned/teds/exact/relaxed_acc/grounding",
        "purpose": "Our proposed real-world evaluation format: per content-class, per-language, "
                   "rotation-robustness, reading-direction, and spotting (basis-of-extraction), each "
                   "scored with a class-appropriate metric. See custom_eval/README.md.",
        "source": "SYNTHETIC (scripts/make_custom_eval.py)",
        "ground_truth": {"n_samples": len(samples),
                         "classes": sorted({s.answer_type for s in samples}),
                         "languages": sorted({s.meta["language"] for s in samples}),
                         "rotations": sorted({s.meta["rotation_deg"] for s in samples})},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(samples)} samples -> {OUT/'custom_eval.jsonl'}")
    from collections import Counter
    print("  by class:", dict(Counter(s.answer_type for s in samples)))
    print("  by lang :", dict(Counter(s.meta["language"] for s in samples)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Reading-order probe: controlled layouts where ONLY the reading order differs.

Renders the SAME ordered elements under layouts that stress reading-order conventions, with exact
GT, so order failures are measurable in isolation (see ``docvlm_eval.metrics.order``):

* ``ltr``       — horizontal left-to-right lines (baseline; any model should pass)
* ``tategaki``  — Japanese vertical writing: columns read top-to-bottom, columns advance
                  RIGHT-TO-LEFT (skipped with a warning if no CJK font is installed)
* ``twocol``    — two-column page with paragraph indents: column-major reading (finish the left
                  column, then the right), where row-major reading scrambles the order
* ``boxes``     — two parallel side-by-side boxes: the sentence STARTS in the left box and
                  CONTINUES in the right one — the order-discriminative "which box continues?" case

Each image emits FOUR samples: transcription scored by ``content_bag`` (did it read everything?)
and by ``order_tau`` (right sequence?), a "what is the k-th element?" QA (exact), and a
"how many text blocks?" segmentation QA (exact) — the last one tests the hypothesis that
paragraph-segmentation ability underlies reading order (correlate segmentation vs order per model).

``--sweep`` additionally renders the ``boxes`` layout at increasing horizontal gaps with fixed
content → the switch-threshold characterization: plot order_tau vs gap per model to find where its
reading logic flips between column-major and row-major (``scripts/analyze_reading_order.py``).

    python scripts/make_reading_order_probe.py                 # 4 layouts x --count images + sweep
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CJK_FONTS = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
             "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# order-discriminative element pools (numbered steps: any scramble is detectable AND incoherent)
_WORDS = ["mix the flour", "add two eggs", "whisk until smooth", "pour into the pan",
          "bake for twenty minutes", "let it cool down", "dust with sugar", "serve warm"]
_JP = ["春の朝に", "鳥が鳴いて", "風が吹く", "花が咲き", "人が集う", "日が沈む"]


def _font(path: str, size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return None


def _new_page(w=900, h=700):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def render_ltr(elements, out):
    img, d = _new_page()
    f = _font(LATIN_FONT, 26)
    for i, el in enumerate(elements):
        d.text((60, 60 + i * 70), f"{i + 1}. {el}", fill="black", font=f)
    img.save(out)
    return len(elements)


def render_tategaki(elements, out):
    """Vertical columns, top-to-bottom; columns advance right-to-left (Japanese tategaki)."""
    f = next((_font(p, 30) for p in CJK_FONTS if _font(p, 30)), None)
    if f is None:
        return 0
    img, d = _new_page(700, 700)
    x = 700 - 90                                    # FIRST column at the RIGHT edge
    for el in elements:
        y = 60
        for ch in el:
            d.text((x, y), ch, fill="black", font=f)
            y += 38
        x -= 90                                     # next column to the LEFT
    img.save(out)
    return len(elements)


def render_twocol(elements, out):
    """Column-major two-column layout with paragraph indents; row-major reading scrambles it."""
    img, d = _new_page()
    f = _font(LATIN_FONT, 24)
    half = (len(elements) + 1) // 2
    for i, el in enumerate(elements):
        col, row = (0, i) if i < half else (1, i - half)
        indent = 30 if (row % 2) else 0             # alternating paragraph indent
        d.text((60 + col * 440 + indent, 60 + row * 90), f"{i + 1}. {el}", fill="black", font=f)
    d.line([(450, 40), (450, 660)], fill="#cccccc", width=1)
    img.save(out)
    return len(elements)


def render_boxes(elements, out, gap=80):
    """Two parallel boxes; the text STARTS in the left box and CONTINUES in the right one."""
    from PIL import Image, ImageDraw
    w = 260 * 2 + gap + 120
    img = Image.new("RGB", (max(w, 700), 600), "white")
    d = ImageDraw.Draw(img)
    f = _font(LATIN_FONT, 22)
    half = (len(elements) + 1) // 2
    for b, chunk in enumerate((elements[:half], elements[half:])):
        x0 = 60 + b * (260 + gap)
        d.rectangle([x0 - 15, 45, x0 + 260, 60 + len(chunk) * 80 + 15], outline="black", width=2)
        for r, el in enumerate(chunk):
            idx = b * half + r
            d.text((x0, 60 + r * 80), f"{idx + 1}. {el}", fill="black", font=f)
    img.save(out)
    return len(elements)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "data" / "probes" / "reading_order"))
    p.add_argument("--count", type=int, default=6, help="images per layout")
    p.add_argument("--sweep", type=int, nargs="*", default=[20, 60, 120, 200, 320, 480],
                   help="box-layout gap sweep (px) for the switch-threshold characterization")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    samples = []

    def emit(layout, img_path, elements, n_blocks, meta=None):
        sid = Path(img_path).stem
        gold = "\n".join(elements)
        base = {"image_path": str(img_path), "answer_type": f"reading-order:{layout}",
                "meta": {"layout": layout, "n_elements": len(elements), **(meta or {})}}
        k = rng.randrange(1, len(elements) + 1)
        samples.extend([
            {**base, "sample_id": f"{sid}_content", "metric": "content_bag", "answers": [gold],
             "question": "Transcribe every text element on the page in reading order."},
            {**base, "sample_id": f"{sid}_order", "metric": "order_tau", "answers": [gold],
             "question": "Transcribe every text element on the page in reading order."},
            {**base, "sample_id": f"{sid}_kth", "metric": "exact",
             "answers": [elements[k - 1]],
             "question": f"What is the {k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'} "
                         f"text element in reading order? Answer with its text only."},
            {**base, "sample_id": f"{sid}_segments", "metric": "exact",
             "answers": [str(n_blocks)],
             "question": "How many distinct text blocks or paragraphs does the page contain? "
                         "Answer with a number."},
        ])

    layouts = {"ltr": render_ltr, "tategaki": render_tategaki,
               "twocol": render_twocol, "boxes": render_boxes}
    skipped_tategaki = False
    for layout, fn in layouts.items():
        for i in range(args.count):
            pool = _JP if layout == "tategaki" else _WORDS
            k = min(len(pool), 6)
            elements = rng.sample(pool, k)
            img_path = out / f"{layout}_{i:03d}.png"
            n = fn(elements, str(img_path))
            if n == 0:
                skipped_tategaki = True
                break
            # block count: numbered lines are blocks; boxes layout has 2 boxes
            emit(layout, img_path, [f"{j + 1}. {el}" for j, el in enumerate(elements)],
                 n_blocks=2 if layout == "boxes" else len(elements))
    if skipped_tategaki:
        print("[warn] tategaki skipped: no CJK font available")

    # switch-threshold sweep: SAME content, growing gap between the parallel boxes
    sweep_elements = _WORDS[:6]
    for gap in args.sweep:
        img_path = out / f"sweep_gap{gap:03d}.png"
        render_boxes(sweep_elements, str(img_path), gap=gap)
        gold = "\n".join(f"{j + 1}. {el}" for j, el in enumerate(sweep_elements))
        samples.append({"image_path": str(img_path), "sample_id": f"sweep_gap{gap:03d}_order",
                        "metric": "order_tau", "answers": [gold],
                        "question": "Transcribe every text element on the page in reading order.",
                        "answer_type": "reading-order:sweep",
                        "meta": {"layout": "boxes", "gap_px": gap,
                                 "n_elements": len(sweep_elements)}})

    (out / "probe.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples)
                                     + "\n", encoding="utf-8")
    n_imgs = len(list(out.glob("*.png")))
    print(f"[ok] reading-order probe: {n_imgs} images, {len(samples)} samples -> {out}")
    print("     metrics: content_bag + order_tau split, k-th element (exact), "
          "segmentation count (exact); sweep gaps:", args.sweep)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

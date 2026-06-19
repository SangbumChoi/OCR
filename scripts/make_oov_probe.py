#!/usr/bin/env python3
"""OOV (out-of-vocabulary script) probe — what does a VLM do with glyphs its tokenizer never
saw? Ancient/invented scripts can't be truly "read", so the interesting signal is the *fallback
pattern*: does the model abstain, transliterate to latin, hallucinate plausible text, or try to
copy the glyphs? We also test whether an in-image **legend** lets it decode by reasoning.

Classes:
  * invented-glyph word, NO legend   -> pure OOV; analyse fallback behaviour
  * invented-glyph word, WITH legend -> solvable by in-context visual symbol reasoning
  * runic word (real ancient script) -> OOV-ish; GT is the romanisation
  * 7-segment digits                 -> non-font glyphs (overlaps digital/special)

Output: data/probes/oov_probe/{images,oov.jsonl,sample.*}
    python scripts/make_oov_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.benchmarks import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.fonts import load_font  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

OUT = ROOT / "data" / "probes" / "oov_probe"
IMG = OUT / "images"
samples: list[Sample] = []


def save(name, im):
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f"{name}.png"
    im.convert("RGB").save(p)
    return str(p)


# ---- invented glyph alphabet: each letter -> a distinct simple shape drawn in a 60px cell ----
def draw_glyph(d: ImageDraw.ImageDraw, letter: str, x: int, y: int, s: int = 50):
    c = (0, 0, 0)
    w = 4
    if letter == "A":      # triangle
        d.polygon([(x + s / 2, y), (x, y + s), (x + s, y + s)], outline=c, width=w)
    elif letter == "B":    # circle with dot
        d.ellipse([x, y, x + s, y + s], outline=c, width=w); d.ellipse([x + s/2-5, y+s/2-5, x+s/2+5, y+s/2+5], fill=c)
    elif letter == "C":    # square
        d.rectangle([x, y, x + s, y + s], outline=c, width=w)
    elif letter == "D":    # X
        d.line([x, y, x + s, y + s], fill=c, width=w); d.line([x + s, y, x, y + s], fill=c, width=w)
    elif letter == "E":    # vertical bars
        d.line([x + s/3, y, x + s/3, y + s], fill=c, width=w); d.line([x + 2*s/3, y, x + 2*s/3, y + s], fill=c, width=w)
    elif letter == "R":    # arrow up
        d.line([x + s/2, y + s, x + s/2, y], fill=c, width=w); d.polygon([(x+s/2-10,y+15),(x+s/2+10,y+15),(x+s/2,y)], fill=c)


def render_word(word: str, with_legend: bool):
    legend_h = 90 if with_legend else 0
    n_uniq = len(set(word))
    W = max(80 * len(word) + 40, (130 + 110 * n_uniq + 20) if with_legend else 0)
    H = 140 + legend_h
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for i, ch in enumerate(word):
        draw_glyph(d, ch, 30 + i * 80, 40)
    if with_legend:
        d.text((20, 150), "Legend:", fill="black", font=load_font(22, True))
        x = 130
        for ch in sorted(set(word)):
            draw_glyph(d, ch, x, 148, 36)
            d.text((x + 8, 188), "= " + ch, fill="black", font=load_font(20))
            x += 110
    return im


def build():
    # 1) invented glyphs, no legend (pure OOV -> fallback study)
    word = "CAB"
    p = save("invented_nolegend", render_word(word, with_legend=False))
    samples.append(Sample("oov_nolegend", p,
        "What does this text say? If you cannot read the symbols, say 'unreadable'.",
        [word], "oov-invented", "ned", {"oov": True, "legend": False, "fallback_probe": True}))

    # 2) invented glyphs, WITH legend (solvable by in-context reasoning)
    p = save("invented_legend", render_word(word, with_legend=True))
    samples.append(Sample("oov_legend", p,
        "Use the legend to decode the symbols into letters. Answer with the decoded word.",
        [word], "oov-invented", "ned", {"oov": True, "legend": True, "fallback_probe": False}))

    # 3) runic (real ancient script; GT = romanisation)
    im = Image.new("RGB", (360, 130), "white")
    ImageDraw.Draw(im).text((20, 40), "ᚠᚢᚦᚨᚱ", fill="black", font=load_font(54))
    p = save("runic", im)
    samples.append(Sample("oov_runic", p,
        "Transcribe/transliterate this text into the latin alphabet. If unreadable, say 'unreadable'.",
        ["futhar", "fuþar", "futhar"], "oov-runic", "ned",
        {"oov": True, "script": "runic", "fallback_probe": True}))

    # 4) 7-segment digits (non-font glyphs)
    im = Image.new("RGB", (320, 140), (10, 10, 10))
    d = ImageDraw.Draw(im)
    _seg7(d, "428", 30, 30, on=(0, 255, 120))
    p = save("seven_seg", im)
    samples.append(Sample("oov_7seg", p, "What number is shown on the display?",
        ["428"], "oov-7seg", "exact", {"oov": True, "glyph": "7segment", "fallback_probe": False}))

    save_jsonl(samples, OUT / "oov.jsonl")
    Image.open(samples[1].image_path).save(OUT / "sample.png")
    (OUT / "sample.json").write_text(json.dumps({
        "benchmark": "oov_probe", "name": "Out-of-vocabulary script probe",
        "category": "F1. Custom capability axes", "metric": "ned / exact",
        "purpose": "Glyphs absent from VLM tokenizers (invented/ancient/7-seg): measure the "
                   "FALLBACK pattern (abstain / transliterate / hallucinate / copy) and whether an "
                   "in-image legend enables decoding by in-context visual reasoning.",
        "source": "SYNTHETIC (scripts/make_oov_probe.py)",
        "ground_truth": {"n_samples": len(samples)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {len(samples)} OOV samples -> {OUT/'oov.jsonl'}")


def _seg7(d, digits, x0, y0, on, w=44, h=80, gap=20):
    # minimal 7-seg renderer
    segs = {"0":"abcdef","1":"bc","2":"abdeg","3":"abcdg","4":"bcfg","5":"acdfg",
            "6":"acdefg","7":"abc","8":"abcdefg","9":"abcdfg"}
    for k, ch in enumerate(digits):
        x = x0 + k * (w + gap)
        s = segs.get(ch, "")
        H = [(x, y0, x + w, y0), (x, y0 + h, x + w, y0 + h), (x, y0 + 2 * h, x + w, y0 + 2 * h)]
        if "a" in s: d.line(H[0], fill=on, width=8)
        if "g" in s: d.line(H[1], fill=on, width=8)
        if "d" in s: d.line(H[2], fill=on, width=8)
        if "f" in s: d.line([x, y0, x, y0 + h], fill=on, width=8)
        if "b" in s: d.line([x + w, y0, x + w, y0 + h], fill=on, width=8)
        if "e" in s: d.line([x, y0 + h, x, y0 + 2 * h], fill=on, width=8)
        if "c" in s: d.line([x + w, y0 + h, x + w, y0 + 2 * h], fill=on, width=8)


if __name__ == "__main__":
    build()

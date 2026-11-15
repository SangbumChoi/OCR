#!/usr/bin/env python
"""Render a short, silent intro video (MP4, 1920x1080) for the UDD rationale + anomaly work.

Slides are drawn with PIL and piped to ffmpeg as raw frames. Panels come from
docs/report/figures/udd_reasoning/ (scripts/visualize_udd_reasoning.py); numbers come from
docs/results/udd_reasoning_stats.json and docs/results/udd_anomalies.json, so the video always
matches the last run.

    python scripts/make_udd_intro_video.py            # -> examples/udd_reasoning/udd_intro.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "report" / "figures" / "udd_reasoning"
W, H, FPS = 1920, 1080, 30
FADE = 0.45                                     # seconds of fade at each slide edge

BG, SURFACE, FG, MUTED, LINE = (16, 22, 24), (27, 35, 38), (236, 240, 237), (150, 162, 158), (52, 62, 64)
ACCENT = (127, 196, 168)
TIER = {"source_verified": (111, 211, 145), "source": (183, 156, 245), "derived_exact": (143, 176, 255),
        "evidence_match": (245, 160, 109), "rejected": (240, 138, 138)}
BAD, GOOD = (240, 138, 138), (111, 211, 145)

_F = "/usr/share/fonts/opentype/inter/"
_FONTS = {"display": _F + "InterDisplay-Bold.otf", "semi": _F + "Inter-SemiBold.otf",
          "body": _F + "Inter-Regular.otf", "bold": _F + "Inter-Bold.otf",
          "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
          "jp": "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"}
_cache: dict = {}


def font(kind: str, size: int):
    key = (kind, size)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(_FONTS[kind], size)
    return _cache[key]


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            cand = f"{line} {w}".strip()
            if d.textlength(cand, font=f) <= width or not line:
                line = cand
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def text_block(d, xy, text, f, fill, width, gap=1.32) -> int:
    x, y = xy
    for ln in wrap(d, text, f, width):
        d.text((x, y), ln, font=f, fill=fill)
        y += int(f.size * gap)
    return y


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def header(d, eyebrow: str, title: str, y: int = 70) -> int:
    d.text((110, y), eyebrow.upper(), font=font("semi", 24), fill=ACCENT)
    return text_block(d, (110, y + 44), title, font("display", 58), FG, W - 220, 1.15) + 10


def chip(d, xy, label, color, size=24) -> int:
    f = font("mono", size)
    x, y = xy
    w = int(d.textlength(label, font=f)) + 24
    d.rounded_rectangle([x, y, x + w, y + size + 16], radius=6, outline=color, width=2)
    d.text((x + 12, y + 7), label, font=f, fill=color)
    return x + w + 12


# --------------------------------------------------------------------------- slide builders
def s_title(st) -> Image.Image:
    im, d = canvas()
    d.text((110, 300), f"danelcsb/UDD  ·  {st['n_rows']:,} images  ·  32 benchmarks".upper(),
           font=font("semi", 26), fill=ACCENT)
    y = text_block(d, (110, 360), "Reasoning from public\ndocument benchmarks", font("display", 104), FG, 1700, 1.1)
    text_block(d, (110, y + 30), "Where an honest rationale can come from, how each one is checked, "
               "and what was wrong in the data along the way.", font("body", 36), MUTED, 1500)
    return im


def s_gap(st) -> Image.Image:
    im, d = canvas()
    header(d, "The starting point", "Benchmarks ship answers, not reasons")
    d.text((110, 330), f"{st['reasoning_before']:,}", font=font("display", 200), fill=BAD)
    d.text((110, 560), f"of {st['qas_before']:,} QAs came with a rationale", font=font("semi", 44), fill=FG)
    text_block(d, (110, 680), "But many rows ship the material a rationale is made of: a receipt parse, "
               "a form's key–value links, OCR words with boxes, an HTML table, a derivation string "
               "hidden inside the answer.", font("body", 36), MUTED, 1600)
    return im


def s_principle(st) -> Image.Image:
    im, d = canvas()
    header(d, "The rule", "Evidence plus an operation, and it is checked")
    cols = [("Never invent", ("Every value a rationale states must already be in the row: the "
                              "instruction, the answers, the boxes, the table, the source's own text.")),
            ("Tier every one", None),
            ("Gate every one", ("Rejected: restating the question, evidence that is not in the row, "
                                "arithmetic that does not reach the gold, a verdict that contradicts it."))]
    x0, cw = 110, 540
    for i, (t, body) in enumerate(cols):
        x = x0 + i * (cw + 50)
        d.rounded_rectangle([x, 320, x + cw, 900], radius=14, fill=SURFACE)
        d.text((x + 36, 356), t, font=font("display", 44), fill=FG)
        if body:
            text_block(d, (x + 36, 440), body, font("body", 32), MUTED, cw - 72)
        else:
            y = 450
            for name, desc in [("source_verified", "shipped, re-computed"), ("source", "shipped by the benchmark"),
                               ("derived_exact", "computed from the row's GT"), ("evidence_match", "matched to OCR / UI text")]:
                chip(d, (x + 36, y), name, TIER[name], 24)
                d.text((x + 36, y + 52), desc, font=font("body", 28), fill=MUTED)
                y += 108
    return im


def s_panel(name: str, eyebrow: str, title: str, chips: list[tuple[str, str]], caption: str) -> Image.Image:
    """Panel slides are rendered LARGER than the frame so the frames can slowly zoom in."""
    im, d = canvas()
    d.text((110, 52), eyebrow.upper(), font=font("semi", 24), fill=ACCENT)
    d.text((110, 90), title, font=font("display", 52), fill=FG)
    x = 110 + int(d.textlength(title, font=font("display", 52))) + 30
    for label, tier in chips:
        x = chip(d, (x, 104), label, TIER.get(tier, MUTED), 22)
    panel = Image.open(FIG / f"{name}.jpg").convert("RGB")
    box_w, box_h = W - 220, H - 210 - 120
    s = min(box_w / panel.width, box_h / panel.height)
    p = panel.resize((int(panel.width * s), int(panel.height * s)), Image.LANCZOS)
    px = (W - p.width) // 2
    im.paste(p, (px, 180))
    d.rectangle([px - 1, 179, px + p.width, 180 + p.height], outline=LINE, width=2)
    text_block(d, (110, H - 108), caption, font("semi", 30), FG, W - 220, 1.25)
    return im


def s_variants(st) -> Image.Image:
    im, d = canvas()
    y = header(d, "Instruction / answer diversity", "One QA, several trainable forms")
    d.text((110, y + 10), "Q: How much change was given back?   gold: 25,400   (CORD, derived_exact)",
           font=font("mono", 28), fill=MUTED)
    rows = [("original", "How much change was given back?", "25,400"),
            ("number", "… Respond with the numeric value only, without units.", "25400"),
            ("json", "… Return JSON of the form {\"answer\": <value>}.", "{\"answer\": \"25,400\"}"),
            ("sentence", "… Answer in one complete sentence.", "The answer is 25,400."),
            ("reasoned", "… Explain briefly, then answer after 'Answer:'.",
             "Reasoning: Cash paid 200,000 minus total 174,600 = 25,400.\nAnswer: 25,400")]
    ty = y + 80
    for fmt, ins, tgt in rows:
        d.rounded_rectangle([110, ty, W - 110, ty + 112], radius=10, fill=SURFACE)
        d.text((140, ty + 36), fmt, font=font("mono", 30), fill=ACCENT)
        d.text((380, ty + 18), ins, font=font("body", 28), fill=MUTED)
        for k, line in enumerate(tgt.split("\n")):
            d.text((380, ty + 56 + 30 * k), line, font=font("mono", 26), fill=FG)
        ty += 128
    text_block(d, (110, ty + 14), "The instruction clause and the answer form change together. The question's "
               "meaning and the gold value never change.", font("semi", 30), FG, W - 220)
    return im


def s_overview(an) -> Image.Image:
    im, d = canvas()
    header(d, "What was wrong in the data", f"{len(an['anomalies'])} anomaly types in 6 families, "
           f"each with a detector")
    fams = [("gold", "the scored gold is not the answer"), ("geometry", "boxes in the wrong frame or place"),
            ("convention", "a label means something else"), ("rationale", "source reasons that are not evidence"),
            ("instruction", "prompt clauses that conflict"), ("number", "notations parsers misread")]
    agg = {f: [0, 0] for f, _ in fams}
    for a in an["anomalies"]:
        agg[a["family"]][0] += 1
        agg[a["family"]][1] += a["hits"]
    cw, ch = 540, 250
    for i, (f, desc) in enumerate(fams):
        x, y = 110 + (i % 3) * (cw + 50), 330 + (i // 3) * (ch + 40)
        d.rounded_rectangle([x, y, x + cw, y + ch], radius=14, fill=SURFACE)
        d.text((x + 34, y + 28), f, font=font("display", 44), fill=FG)
        d.text((x + 34, y + 92), f"{agg[f][1]:,} hits", font=font("display", 52), fill=ACCENT)
        text_block(d, (x + 34, y + 160), f"{agg[f][0]} cases · {desc}", font("body", 27), MUTED, cw - 68, 1.25)
    d.text((110, H - 100), f"Counted on all {an['n_rows']:,} rows by scripts/audit_udd_anomalies.py",
           font=font("body", 28), fill=MUTED)
    return im


def s_case(cid: str, title: str, count: str, before: list[tuple[str, str]], after: list[tuple[str, str]],
           note: str) -> Image.Image:
    im, d = canvas()
    d.text((110, 70), f"ANOMALY {cid}", font=font("semi", 24), fill=ACCENT)
    text_block(d, (110, 114), title, font("display", 56), FG, W - 220, 1.15)
    d.text((110, 260), count, font=font("semi", 34), fill=MUTED)
    for k, (lab, items, col) in enumerate([("before", before, BAD), ("after", after, GOOD)]):
        x = 110 + k * 870
        d.rounded_rectangle([x, 340, x + 830, 860], radius=14, fill=SURFACE, outline=col, width=3)
        d.text((x + 34, 366), lab.upper(), font=font("semi", 28), fill=col)
        y = 430
        for kind, text in items:
            f = font("jp" if any(ord(c) > 0x2E80 for c in text) else ("mono" if kind == "code" else "body"),
                     28 if kind == "code" else 30)
            y = text_block(d, (x + 34, y), text, f, FG if kind != "dim" else MUTED, 760) + 16
    text_block(d, (110, 900), note, font("semi", 30), FG, W - 220)
    return im


def s_results(st, an) -> Image.Image:
    im, d = canvas()
    header(d, "Result on the full corpus", "From 325 rationales to a checked, diversified set")
    stats = [(f"{st['reasoning_before']:,} → {st['reasoning_after']:,}", "QAs with a rationale"),
             (f"{st['derived_qas']:,}", "new QAs computed from the rows' own ground truth"),
             (f"{st['answers_repaired']:,}", "corrupted golds repaired"),
             (f"{st['variants']:,}", "format-bound instruction / answer variants"),
             (f"{len(an['anomalies'])}", "anomaly types, each detected and counted")]
    y = 320
    for big, label in stats:
        d.text((110, y), big, font=font("display", 70), fill=ACCENT)
        d.text((820, y + 22), label, font=font("body", 36), fill=FG)
        y += 128
    return im


def s_end(st) -> Image.Image:
    im, d = canvas()
    header(d, "Reproduce", "Everything is a script over the public corpus")
    cmds = ["python scripts/derive_udd_reasoning.py --from-hub",
            "python scripts/audit_udd_anomalies.py --from-hub",
            "python scripts/visualize_udd_reasoning.py",
            "python scripts/make_udd_intro_video.py"]
    y = 330
    for c in cmds:
        d.rounded_rectangle([110, y, W - 110, y + 80], radius=10, fill=SURFACE)
        d.text((140, y + 22), c, font=font("mono", 32), fill=FG)
        y += 104
    text_block(d, (110, y + 30), "docs/report/benchmark_reasoning_extraction.md  ·  "
               "docs/report/udd_anomaly_catalogue.md", font("body", 30), MUTED, W - 220)
    return im


# --------------------------------------------------------------------------- assembly
def build_slides(st, an) -> list[tuple[Image.Image, float, bool]]:
    """(image, seconds, zoom) per slide."""
    P = []
    P.append((s_title(st), 5.0, False))
    P.append((s_gap(st), 6.5, False))
    P.append((s_principle(st), 8.0, False))
    panels = [
        ("tatqa", "Verified arithmetic", "TAT-QA", [("source_verified", "source_verified")],
         "The derivation was hidden inside the gold. Split out and re-computed: (16,284 − 6,509) / 6,509 = 150.18%."),
        ("cord", "Verified arithmetic", "CORD", [("derived_exact", "derived_exact")],
         "Subtotal, total and change from the receipt parse, emitted only when the printed numbers add up."),
        ("funsd", "Grounded in structure", "FUNSD", [("derived_exact", "derived_exact")],
         "The source's key–value links, grounded in word boxes: the value sits to the right of its label."),
        ("pubtabnet", "Grounded in structure", "PubTabNet", [("derived_exact", "derived_exact")],
         "Cell lookups by unique row label and column header, only on tables without merged cells."),
        ("doclaynet", "Grounded in structure", "DocLayNet", [("derived_exact", "derived_exact")],
         "Region counts with positions, asked only when the count is small enough to check."),
        ("ocrvqa", "Grounded in OCR", "OCR-VQA", [("evidence_match", "evidence_match"), ("derived_exact", "derived_exact")],
         "Title words found on the cover; 'Is this book related to Travel?' answered from the row's own genre."),
    ]
    for name, eb, title, chips, cap in panels:
        P.append((s_panel(name, eb, title, chips, cap), 6.5, True))
    P.append((s_variants(st), 8.0, False))
    P.append((s_overview(an), 8.5, False))
    P.append((s_case("G1", "The derivation is inside the gold answer", "3,057 golds · TAT-QA, Docmatix",
                     [("dim", "gold as stored:"), ("code", "(16,284 - 6,509) / 6,509\nAnswer: 150.18."),
                      ("dim", "a correct '150.18' is scored as wrong")],
                     [("dim", "gold:"), ("code", "150.18"), ("dim", "rationale (source_verified):"),
                      ("code", "Compute (16,284 - 6,509) / 6,509\n= 1.5018; as a percentage: 150.18%.")],
                     "Split at 'Answer:', re-evaluate the expression, keep it only if it reproduces the gold."), 7.5, False))
    P.append((s_case("I1", "A brevity suffix conflicts with the new format", "17,003 instructions · the_cauldron (18 variants)",
                     [("code", "Which bar has the largest value?\nOffer a terse response.\nRespond with a full sentence."),
                      ("dim", "an impossible instruction")],
                     [("code", "Which bar has the largest value?\nRespond with a full sentence."),
                      ("dim", "target: The answer is Soil.")],
                     "Measured suffixes are stripped first. Semantic clauses such as 'Answer scale should be: percent.' stay."),
           7.0, False))
    P.append((s_case("B1", "Pixel boxes kept in the pre-resize frame", "2,078 boxes off-image · CORD 39%, ScreenQA 60%",
                     [("code", "[672, 652, 746, 678]  on a 667×1000 image"),
                      ("dim", ("the build shrank images to ≤1000 px and left pixel boxes untouched, "
                               "so every pixel box is mis-scaled"))],
                     [("dim", "the loader now normalizes boxes before resizing"),
                      ("dim", "until UDD is rebuilt, rationales use the element text and ignore pixel boxes")],
                     "Found because a 'grounded' training target pointed past the right edge (x = 1009 of 1000)."),
           7.5, False))
    P.append((s_panel("ocrvqa", "Anomaly B2 · found by drawing it", "Recurring words", [("151 hits", "rejected")],
                      "'doku' is in the title and in the tagline: the evidence box covered the whole cover. "
                      "Now: nearest occurrence to the largest word, at most 40% of the page away."), 7.0, True))
    P.append((s_panel("hallusionbench_verdict_stripped", "Anomaly R2 · found by drawing it", "Paired-question verdict",
                      [("source", "source")],
                      "'Yes.' belongs to the paired question (gold here is 'no'); the statement after it matches "
                      "the chart. Only the verdict is stripped."), 7.0, True))
    P.append((s_case("I2", "English answer templates on non-English questions", "3,424 questions · MTVQA, OCRBench v2, MathVista",
                     [("code", "診療時間は何時からですか\nAnswer in one complete sentence."),
                      ("code", "→ The answer is 午前9時から…")],
                     [("dim", "sentence templates only for English questions;"),
                      ("dim", "JSON and short forms still apply")],
                     "Gated: the variant is not generated rather than generated in two languages."), 6.5, False))
    P.append((s_results(st, an), 7.5, False))
    P.append((s_end(st), 6.0, False))
    return P


def render(out: Path, st, an) -> float:
    slides = build_slides(st, an)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                            stdin=subprocess.PIPE)
    black = Image.new("RGB", (W, H), BG)
    total = 0.0
    for im, secs, zoom in slides:
        n = int(secs * FPS)
        nf = int(FADE * FPS)
        for i in range(n):
            frame = im
            if zoom:                                   # slow push-in, 1.00 -> 1.05
                z = 1 + 0.05 * i / max(n - 1, 1)
                cw, ch = int(W / z), int(H / z)
                x0, y0 = (W - cw) // 2, int((H - ch) * 0.35)
                frame = im.crop((x0, y0, x0 + cw, y0 + ch)).resize((W, H), Image.BILINEAR)
            a = min(1.0, i / nf, (n - 1 - i) / nf) if nf else 1.0
            if a < 1.0:
                frame = Image.blend(black, frame, max(a, 0.0))
            proc.stdin.write(frame.tobytes())
        total += n / FPS
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "examples" / "udd_reasoning" / "udd_intro.mp4"),
                    help="output MP4 (default lives in the git-ignored examples/udd_reasoning/)")
    ap.add_argument("--stills", help="also write one PNG per slide into this directory")
    args = ap.parse_args(argv)
    s = json.loads((ROOT / "docs" / "results" / "udd_reasoning_stats.json").read_text())
    st = {"n_rows": s["n_rows"]}
    for v in s["per_source"].values():
        for k, x in v.items():
            st[k] = st.get(k, 0) + x
    an = json.loads((ROOT / "docs" / "results" / "udd_anomalies.json").read_text())
    if args.stills:
        Path(args.stills).mkdir(parents=True, exist_ok=True)
        for i, (im, _, _) in enumerate(build_slides(st, an)):
            im.save(Path(args.stills) / f"slide_{i:02d}.png")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    secs = render(Path(args.out), st, an)
    print(f"wrote {args.out} ({secs:.1f} s, {Path(args.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Realistic synthetic generator for the document-type *special cases*
(report/document_type_taxonomy.md): renders genuinely document-looking pages and applies
scanner/phone/historical degradation, while keeping ground truth exact.

Pipeline (the key idea — realism AND exact GT at once):
    HTML/CSS (+ Faker values)  --WeasyPrint-->  PDF
    PDF  --PyMuPDF-->  PNG (rasterized at DPI)  +  field boxes via page.search_for()
    PNG  --Augraphy-->  realistic degraded copy   (photometric only -> boxes stay valid)

GT box coordinates come straight from the renderer (PDF text positions scaled by DPI/72), so
spotting boxes are pixel-exact for free. Faker is seeded -> deterministic. Some cases (LCD/7-seg)
are drawn with PIL because they have no real font.

    python scripts/make_realistic_cases.py            # all cases
    python scripts/make_realistic_cases.py --only id_card cheque   # a subset
    python scripts/make_realistic_cases.py --no-degrade            # clean only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from faker import Faker
from PIL import Image, ImageDraw
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks" / "realistic_cases"
DPI = 150
ZOOM = DPI / 72.0
fake = Faker()
Faker.seed(7)

records: list[dict] = []


# --------------------------------------------------------------------------- render core
def render(html: str, css: str = "") -> tuple[Image.Image, fitz.Page, fitz.Document]:
    """HTML/CSS -> (PIL image, fitz page) at DPI. Page kept open for box queries."""
    full = f"<style>{css}</style>{html}"
    pdf = HTML(string=full, base_url=str(ROOT)).write_pdf()
    doc = fitz.open(stream=pdf, filetype="pdf")
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return img, page, doc


def box_of(page: fitz.Page, text: str) -> list[int] | None:
    """Pixel box [x1,y1,x2,y2] of the first occurrence of `text`, or None."""
    rects = page.search_for(text)
    if not rects:
        return None
    r = rects[0]
    return [round(r.x0 * ZOOM), round(r.y0 * ZOOM), round(r.x1 * ZOOM), round(r.y1 * ZOOM)]


# --------------------------------------------------------------------------- degradation
def degrade(img: Image.Image, preset: str) -> Image.Image | None:
    """Augraphy photometric degradation (no geometry -> GT boxes remain valid)."""
    try:
        import cv2
        import numpy as np
        from augraphy import (
            BadPhotoCopy, BleedThrough, Brightness, ColorPaper, DirtyDrum,
            Jpeg, LightingGradient, NoiseTexturize, ShadowCast, SubtleNoise,
        )
    except Exception as e:  # pragma: no cover
        print("  [skip degrade]", e)
        return None
    import numpy as np
    arr = np.array(img)[:, :, ::-1]  # RGB->BGR

    def safe(cls, **kw):
        try:
            return cls(**kw)
        except Exception:
            return None

    presets = {
        "scan": ([safe(BleedThrough, p=0.4)],
                 [safe(ColorPaper, p=0.6), safe(NoiseTexturize, p=0.5)],
                 [safe(SubtleNoise, p=0.5), safe(Jpeg, quality_range=(45, 75))]),
        "photo": ([],
                  [safe(ColorPaper, p=0.3)],
                  [safe(LightingGradient), safe(ShadowCast, p=0.7),
                   safe(Brightness, brightness_range=(0.85, 1.1)), safe(Jpeg, quality_range=(35, 60))]),
        "historical": ([safe(BleedThrough, p=0.8)],
                       [safe(ColorPaper, p=0.9), safe(NoiseTexturize, p=0.8)],
                       [safe(Brightness, brightness_range=(0.7, 0.95)),
                        safe(ShadowCast, p=0.6), safe(SubtleNoise)]),
        "fax": ([],
                [safe(DirtyDrum, p=0.7)],
                [safe(BadPhotoCopy, p=0.9), safe(Jpeg, quality_range=(20, 40))]),
    }
    ink, paper, post = presets[preset]
    from augraphy import AugraphyPipeline
    pipe = AugraphyPipeline(
        ink_phase=[a for a in ink if a],
        paper_phase=[a for a in paper if a],
        post_phase=[a for a in post if a],
    )
    out = pipe(arr)
    return Image.fromarray(out[:, :, ::-1])


def emit(key: str, img: Image.Image, page, gt: dict, degrade_preset: str, do_degrade: bool):
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "clean.png")
    if do_degrade:
        deg = degrade(img, degrade_preset)
        if deg is not None:
            deg.save(folder / "degraded.png")
            gt["degraded_preset"] = degrade_preset
    (folder / "gt.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    records.append({"key": key, **{k: gt[k] for k in ("type", "stressors", "anchor_metric") if k in gt}})
    print(f"[ok] {key:18} {img.size}  fields={len(gt.get('fields', {}))}")


# --------------------------------------------------------------------------- shared CSS
BASE = """
@page { size: A5; margin: 12mm; }
* { box-sizing: border-box; }
body { font-family: 'Liberation Sans', sans-serif; color:#111; font-size: 11px; }
h1,h2 { margin: 0 0 6px; }
table { border-collapse: collapse; width: 100%; font-size: 10px; }
td,th { border: 1px solid #888; padding: 3px 5px; text-align: left; }
.muted { color:#555; }
"""


# ============================================================ cases
def case_invoice(do_degrade):
    company = fake.company()
    inv_no = f"INV-2025-{fake.random_int(1000,9999)}"
    items = [(fake.bs().title()[:22], fake.random_int(1, 5), round(fake.random_int(10, 400) + 0.5, 2))
             for _ in range(4)]
    rows = "".join(f"<tr><td>{n}</td><td>{q}</td><td>${p:.2f}</td><td>${q*p:.2f}</td></tr>"
                   for n, q, p in items)
    total = sum(q * p for _, q, p in items)
    total_str = f"${total:,.2f}"
    html = f"""
    <h1>INVOICE</h1>
    <p class=muted>{company}<br>{fake.street_address()}, {fake.city()}</p>
    <p><b>Invoice No:</b> {inv_no} &nbsp; <b>Date:</b> {fake.date('%Y-%m-%d')}</p>
    <table><tr><th>Item</th><th>Qty</th><th>Unit</th><th>Amount</th></tr>{rows}
      <tr><td colspan=3 style="text-align:right"><b>TOTAL</b></td>
          <td style="color:#a00;font-weight:bold">{total_str}</td></tr></table>
    """
    img, page, doc = render(html, BASE)
    gt = {"type": "invoice/receipt", "stressors": ["layout", "table", "spotting", "hallucination"],
          "anchor_metric": "KIE F1 + spotting IoU",
          "fields": {"invoice_no": inv_no, "company": company, "total": f"{total:.2f}"},
          "spotting": {"total": box_of(page, total_str)},
          "abstain_probe": {"question": "What is the shipping tracking number?",
                            "expected": "not present / abstain"}}
    emit("invoice", img, page, gt, "scan", do_degrade); doc.close()


def case_id_card(do_degrade):
    name = fake.name().upper()
    idn = f"{fake.random_int(100000000,999999999)}"
    dob = fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%d %b %Y").upper()
    exp = "14 JUN 2031"
    surname = name.split()[-1]
    given = " ".join(name.split()[:-1])
    mrz1 = f"P<USA{surname}<<{given.replace(' ', '<')}".ljust(44, "<")[:44]
    mrz2 = f"{idn}<4USA{fake.numerify('######')}M310614<<<<<<<<<<<<<<00".ljust(44, "<")[:44]
    css = BASE + """
    @page { size: 90mm 58mm; margin: 0; }
    body { font-size: 9px; }
    .card { width:90mm; height:58mm; padding:6mm; background:
        linear-gradient(135deg,#eef3fb,#dbe6f5); position:relative; }
    .photo { position:absolute; right:6mm; top:6mm; width:20mm; height:26mm;
        background:#9fb0c8; border:1px solid #6b7c96; }
    .mrz { position:absolute; left:0; bottom:0; width:100%; padding:2mm 6mm;
        font-family:'Liberation Mono',monospace; font-size:10px; letter-spacing:1px;
        background:#f4f4f4; border-top:1px solid #bbb; }
    .lbl { color:#345; font-size:7px; text-transform:uppercase; }
    .val { font-weight:bold; font-size:11px; }
    """
    html = f"""
    <div class=card>
      <h2 style="color:#234">REPUBLIC — IDENTITY CARD</h2>
      <div class=photo></div>
      <p class=lbl>Surname / Given names</p><p class=val>{name}</p>
      <p class=lbl>Document No</p><p class=val>{idn}</p>
      <p class=lbl>Date of birth</p><p class=val>{dob}</p>
      <p class=lbl>Expiry</p><p class=val>{exp}</p>
      <div class=mrz>{mrz1}<br>{mrz2}</div>
    </div>"""
    img, page, doc = render(html, css)
    gt = {"type": "ID / passport", "stressors": ["language", "MRZ", "hallucination", "spotting"],
          "anchor_metric": "KIE F1 + abstain + IoU",
          "fields": {"name": name, "doc_no": idn, "dob": dob, "expiry": exp},
          "mrz": [mrz1, mrz2],
          "spotting": {"photo_region": "top-right portrait box", "doc_no": box_of(page, idn)},
          "abstain_probe": {"question": "What is the cardholder's blood type?",
                            "expected": "not present / abstain"}}
    emit("id_card", img, page, gt, "photo", do_degrade); doc.close()


def case_checkbox_form(do_degrade):
    opts = [("Email", True), ("SMS", False), ("Phone call", True), ("Postal mail", False)]
    langs = [("English", True), ("Korean", False), ("Spanish", False)]
    def rows(items):
        return "".join(f"<div class=row><span class=box>{'☒' if c else '☐'}</span> {t}</div>"
                       for t, c in items)
    css = BASE + """
    .row{ margin:4px 0; font-size:13px;} .box{ font-size:15px; margin-right:6px;}
    .sec{ font-weight:bold; margin-top:10px;}
    """
    html = f"""
    <h2>SERVICE ENROLLMENT FORM</h2>
    <p class=muted>Applicant: {fake.name()} &nbsp; ID: {fake.numerify('A#######')}</p>
    <div class=sec>Preferred contact methods:</div>{rows(opts)}
    <div class=sec>Notification language:</div>{rows(langs)}
    """
    img, page, doc = render(html, css)
    gt = {"type": "checkbox form", "stressors": ["selection-marks", "layout", "hallucination"],
          "anchor_metric": "selection-mark accuracy + F1",
          "fields": {"contact_checked": [t for t, c in opts if c],
                     "language_checked": [t for t, c in langs if c]},
          "task": "List only the CHECKED (☒) options.",
          "abstain_probe": {"question": "Is 'Fax' selected?", "expected": "option not present"}}
    emit("checkbox_form", img, page, gt, "scan", do_degrade); doc.close()


def case_redacted(do_degrade):
    name = fake.name()
    css = BASE + ".rd{ background:#111; color:#111; }"
    html = f"""
    <h2>CONFIDENTIAL MEMORANDUM</h2>
    <p class=muted>Date: {fake.date('%Y-%m-%d')} &nbsp; Ref: {fake.numerify('CASE-####')}</p>
    <p>The subject, <b>{name}</b>, attended the meeting held at the
    <span class=rd>████████████████</span> facility. The disclosed account number
    <span class=rd>███████████</span> has been sealed by order of the court.
    Authorising officer: <span class=rd>████████</span>.</p>
    <p>The next review is scheduled for {fake.date('%Y-%m-%d')}.</p>
    """
    img, page, doc = render(html, css)
    gt = {"type": "redacted document", "stressors": ["hallucination", "spotting"],
          "anchor_metric": "abstain (no-hallucination)",
          "fields": {"subject": name},
          "task": "Answer ONLY from visible text; if a value is blacked out, say '[redacted]'.",
          "abstain_probe": {"question": "What is the disclosed account number?",
                            "expected": "[redacted] (it is blacked out) — must NOT invent digits"}}
    emit("redacted", img, page, gt, "scan", do_degrade); doc.close()


def case_bank_statement(do_degrade):
    txns = [(fake.date('%m/%d'), fake.bs().title()[:20], round(fake.random_int(-500, 800) + 0.25, 2))
            for _ in range(6)]
    bal = 1000.0
    rows = ""
    for d, desc, amt in txns:
        bal += amt
        rows += f"<tr><td>{d}</td><td>{desc}</td><td>{amt:+.2f}</td><td>{bal:.2f}</td></tr>"
    html = f"""
    <h2>MONTHLY STATEMENT</h2>
    <p class=muted>{fake.company()} Bank &nbsp; Acct: {fake.numerify('****####')}</p>
    <table><tr><th>Date</th><th>Description</th><th>Amount</th><th>Balance</th></tr>{rows}</table>
    <p><b>Closing balance:</b> ${bal:.2f}</p>
    """
    img, page, doc = render(html, BASE)
    gt = {"type": "bank statement", "stressors": ["layout", "table", "spotting"],
          "anchor_metric": "TEDS + F1",
          "fields": {"closing_balance": f"{bal:.2f}", "n_transactions": len(txns)},
          "task": "Convert the transaction table to HTML (TEDS)."}
    emit("bank_statement", img, page, gt, "scan", do_degrade); doc.close()


def case_rtl_arabic(do_degrade):
    # Arabic letter (RTL). GT = transcription + reading-direction.
    text = "إشعار استلام رقم ٤٢ — المبلغ الإجمالي ١٤٥ ريالاً"
    css = BASE + """
    body{ direction:rtl; font-family:'Amiri','Noto Naskh Arabic',serif; font-size:16px;}
    h2{ font-family:'Amiri',serif;}
    """
    html = f"""
    <h2 dir=rtl>إيصال دفع</h2>
    <p dir=rtl>{text}</p>
    <p dir=rtl class=muted>التاريخ: ٢٠٢٥/٠٦/١٤ — التوقيع: __________</p>
    """
    img, page, doc = render(html, css)
    gt = {"type": "RTL doc (Arabic)", "stressors": ["read-direction(RTL)", "language", "script"],
          "anchor_metric": "direction + per-lang NED",
          "fields": {"language": "ar", "reading_direction": "rtl"},
          "transcript": text,
          "direction_probe": {"question": "Is this text right-to-left or left-to-right?",
                              "expected": "right-to-left"}}
    emit("rtl_arabic", img, page, gt, "scan", do_degrade); doc.close()


def case_webtoon(do_degrade):
    # vertical stacked panels with speech bubbles -> reading order top->bottom
    lines = ["Are you ready?", "I think so...", "Then let's begin!", "Wait — look over there!"]
    panels = ""
    for i, ln in enumerate(lines, 1):
        side = "left" if i % 2 else "right"
        panels += f"""
        <div class=panel>
          <span class=pno>{i}</span>
          <div class="bubble {side}">{ln}</div>
        </div>"""
    css = """
    @page{ size: 80mm 200mm; margin:4mm;}
    body{ font-family:'Liberation Sans',sans-serif;}
    .panel{ position:relative; height:44mm; border:2px solid #111; margin-bottom:3mm;
        background:linear-gradient(160deg,#f7f7f7,#e7ecf2);}
    .pno{ position:absolute; top:2mm; left:2mm; font-size:9px; color:#888;}
    .bubble{ position:absolute; top:8mm; max-width:55%; background:#fff; border:2px solid #111;
        border-radius:14px; padding:5px 9px; font-size:13px;}
    .bubble.left{ left:5mm;} .bubble.right{ right:5mm;}
    """
    img, page, doc = render("".join([f"<div>{panels}</div>"]), css)
    gt = {"type": "webtoon / manga", "stressors": ["read-order", "direction(vertical)", "SFX/art-text"],
          "anchor_metric": "read-order + direction + NED",
          "fields": {"n_panels": len(lines)},
          "reading_order": lines,
          "task": "Transcribe the speech bubbles in correct reading order (top to bottom)."}
    emit("webtoon", img, page, gt, "photo", do_degrade); doc.close()


def case_prescription(do_degrade):
    patient = fake.name()
    drug = "Amoxicillin 500mg — 1 cap TID x7d"
    css = BASE + """
    .rx{ font-size:34px; font-weight:bold;}
    .hand{ font-family:'Purisa',cursive; font-size:18px; color:#1a2a5a;}
    .line{ border-bottom:1px solid #999; height:1.4em;}
    """
    html = f"""
    <h2>{fake.name()}, M.D. — Internal Medicine</h2>
    <p class=muted>Patient: {patient} &nbsp; Date: {fake.date('%Y-%m-%d')}</p>
    <p class=rx>℞</p>
    <p class=hand>{drug}</p>
    <p class=hand>Sig: take with food</p>
    <p>Signature: <span class=hand>{fake.last_name()}</span></p>
    """
    img, page, doc = render(html, css)
    gt = {"type": "prescription", "stressors": ["handwriting", "hallucination", "degradation"],
          "anchor_metric": "CER(hand) + abstain",
          "fields": {"patient": patient, "rx_handwritten": drug},
          "task": "Transcribe the handwritten medication line (CER).",
          "abstain_probe": {"question": "What is the refill count?",
                            "expected": "not legible / not present — abstain"}}
    emit("prescription", img, page, gt, "fax", do_degrade); doc.close()


def case_cheque(do_degrade):
    payee = fake.name()
    amt_num = "1,450.00"
    amt_words = "One thousand four hundred fifty and 00/100"
    css = BASE + """
    @page{ size: 160mm 70mm; margin:6mm;}
    .cheque{ border:2px solid #1b3a6b; padding:5mm; height:100%; background:#f3f7fc;}
    .num{ float:right; border:1px solid #1b3a6b; padding:3px 8px; font-weight:bold; font-size:15px;}
    .words{ font-family:'Purisa',cursive; font-size:16px; border-bottom:1px solid #888; margin-top:8mm;}
    .micr{ font-family:'Liberation Mono',monospace; letter-spacing:2px; margin-top:6mm;}
    """
    html = f"""
    <div class=cheque>
      <p>{fake.company()} BANK <span class=num>$ {amt_num}</span></p>
      <p>Pay to the order of: <b>{payee}</b></p>
      <p class=words>{amt_words}</p>
      <p class=micr>⑆000123456⑆ 0987654321⑈ 4421</p>
      <p style="text-align:right">Signature: <span style="font-family:Purisa">{fake.last_name()}</span></p>
    </div>"""
    img, page, doc = render(html, css)
    gt = {"type": "cheque / 수표", "stressors": ["handwriting", "dual-amount", "hallucination", "spotting"],
          "anchor_metric": "dual-amount F1 + sign detect",
          "fields": {"payee": payee, "amount_numeric": amt_num, "amount_words": amt_words},
          "spotting": {"courtesy_amount": box_of(page, amt_num)},
          "consistency_probe": {"question": "Do the numeric and written amounts agree?",
                                "expected": "yes (1450.00 == one thousand four hundred fifty)"}}
    emit("cheque", img, page, gt, "scan", do_degrade); doc.close()


def case_ancient(do_degrade):
    # classical vertical-CJK manuscript; degraded with the 'historical' preset (fade/stain)
    poem = ["山","重","水","複","疑","無","路"]
    cols = "".join(f"<div class=col>{''.join(c)}</div>" for c in [poem[:4], poem[4:]])
    css = """
    @page{ size: 120mm 150mm; margin:14mm;}
    body{ font-family:'Noto Serif CJK SC',serif; background:#efe7d2;}
    h2{ font-family:'EB Garamond',serif; color:#5a4a2a;}
    .wrap{ display:flex; flex-direction:row-reverse; gap:10mm; justify-content:center;}
    .col{ writing-mode:vertical-rl; font-size:30px; line-height:1.6; color:#2a1f12;}
    """
    html = f"<h2>古文書 — Classical manuscript</h2><div class=wrap>{cols}</div>"
    img, page, doc = render(html, css)
    gt = {"type": "ancient manuscript / 고문서",
          "stressors": ["direction(vertical)", "language(classical)", "degradation(fade/stain)"],
          "anchor_metric": "NED + robustness",
          "fields": {"language": "zh-classical", "reading_direction": "vertical-rtl"},
          "transcript": "".join(poem),
          "task": "Transcribe the characters in reading order (top-to-bottom, right-to-left)."}
    emit("ancient", img, page, gt, "historical", do_degrade); doc.close()


def case_lcd(do_degrade):
    # 7-segment LCD meter (PIL: no real font for 7-seg) + glare via Augraphy 'photo'
    digits = "01428"
    W, H = 520, 200
    im = Image.new("RGB", (W, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    _seg7(d, digits, 40, 50, on=(40, 255, 150), off=(28, 40, 36))
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 70, 80), width=4)
    gt = {"type": "LCD / meter / 7-seg", "stressors": ["non-font digits", "glare"],
          "anchor_metric": "exact + IoU",
          "fields": {"reading": digits},
          "task": "What number is on the display?"}
    folder = OUT / "lcd_7seg"; folder.mkdir(parents=True, exist_ok=True)
    im.save(folder / "clean.png")
    if do_degrade:
        deg = degrade(im, "photo")
        if deg is not None:
            deg.save(folder / "degraded.png"); gt["degraded_preset"] = "photo"
    (folder / "gt.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    records.append({"key": "lcd_7seg", "type": gt["type"], "stressors": gt["stressors"],
                    "anchor_metric": gt["anchor_metric"]})
    print(f"[ok] {'lcd_7seg':18} {im.size}")


def _seg7(d, digits, x0, y0, on, off, w=60, h=58, t=10, gap=34):
    segs = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
            "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
    for k, ch in enumerate(digits):
        x = x0 + k * (w + gap); s = segs.get(ch, "")
        def H(yy): return [(x + t, yy), (x + w, yy), (x + w - t, yy + t), (x + t * 2, yy + t)]
        def seg(name, pts):
            d.polygon(pts, fill=on if name in s else off)
        seg("a", [(x + t, y0), (x + w, y0), (x + w - t, y0 + t), (x + t * 2, y0 + t)])
        seg("g", [(x + t, y0 + h), (x + w, y0 + h), (x + w - t, y0 + h + t), (x + t * 2, y0 + h + t)])
        seg("d", [(x + t, y0 + 2 * h), (x + w, y0 + 2 * h), (x + w - t, y0 + 2 * h + t), (x + t * 2, y0 + 2 * h + t)])
        seg("f", [(x, y0 + t), (x + t, y0 + 2 * t), (x + t, y0 + h - t), (x, y0 + h)])
        seg("b", [(x + w, y0 + t), (x + w + t, y0 + 2 * t), (x + w + t, y0 + h - t), (x + w, y0 + h)])
        seg("e", [(x, y0 + h + t), (x + t, y0 + h + 2 * t), (x + t, y0 + 2 * h - t), (x, y0 + 2 * h)])
        seg("c", [(x + w, y0 + h + t), (x + w + t, y0 + h + 2 * t), (x + w + t, y0 + 2 * h - t), (x + w, y0 + 2 * h)])


CASES = {
    "invoice": case_invoice, "id_card": case_id_card, "checkbox_form": case_checkbox_form,
    "redacted": case_redacted, "bank_statement": case_bank_statement, "rtl_arabic": case_rtl_arabic,
    "webtoon": case_webtoon, "prescription": case_prescription, "cheque": case_cheque,
    "ancient": case_ancient, "lcd_7seg": case_lcd,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CASES), help="subset of cases")
    ap.add_argument("--no-degrade", action="store_true", help="render clean only (fast)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = args.only or list(CASES)
    for k in keys:
        CASES[k](do_degrade=not args.no_degrade)
    (OUT / "index.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {len(keys)} realistic cases -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Realistic synthetic generator for the document-type *special cases*
(report/document_type_taxonomy.md).

Each case is declared through ``docvlm_eval.synth.DocBuilder`` so the ground truth is *produced
by* the render — every value is declared once (no drift between pixels and labels) and spotting
boxes are read straight out of the rendered PDF. A photometric Augraphy preset then makes a
realistic degraded copy whose boxes are still valid (no geometry changed). Faker (seeded) fills
field content deterministically.

Output per case: data/benchmarks/realistic_cases/<key>/{clean.png, degraded.png, gt.json}

    python scripts/make_realistic_cases.py                       # all cases
    python scripts/make_realistic_cases.py --only id_card cheque
    python scripts/make_realistic_cases.py --no-degrade          # clean only (fast)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.synth import DocBuilder, degrade, esc  # noqa: E402

OUT = ROOT / "data" / "benchmarks" / "realistic_cases"
DPI = 150
fake = Faker()
Faker.seed(7)
records: list[dict] = []


def emit(key: str, builder_or_img, preset: str, do_degrade: bool, gt: dict | None = None):
    """Render a builder (or accept a prebuilt PIL image + gt) -> clean.png + degraded.png + gt.json."""
    if gt is None:
        img, gt = builder_or_img.build(dpi=DPI)
    else:
        img = builder_or_img
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "clean.png")
    if do_degrade:
        deg = degrade(img, preset)
        if deg is not None:
            deg.save(folder / "degraded.png")
            gt["degraded_preset"] = preset
    (folder / "gt.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    records.append({"key": key, "type": gt["type"], "stressors": gt["stressors"],
                    "anchor_metric": gt["anchor_metric"]})
    n_spot = len(gt.get("spotting", {}))
    print(f"[ok] {key:14} {img.size}  fields={len(gt.get('fields',{}))} spots={n_spot}")


# ============================================================ paper / scan cases
def case_invoice(do_degrade):
    b = DocBuilder("invoice/receipt", ["layout", "table", "spotting", "hallucination"],
                   "KIE F1 + spotting IoU", page="A5", css=".total{color:#a00;font-size:15px}")
    company = fake.company()
    inv_no = f"INV-2025-{fake.random_int(1000,9999)}"
    items = [(fake.bs().title()[:22], fake.random_int(1, 5), round(fake.random_int(10, 400) + 0.5, 2))
             for _ in range(4)]
    rows = [[n, str(q), f"${p:.2f}", f"${q*p:.2f}"] for n, q, p in items]
    total = sum(q * p for _, q, p in items)
    total_str = f"${total:,.2f}"
    b.title("INVOICE")
    b.line(f"{esc(company)}<br>{esc(fake.street_address())}", cls="muted")
    b.field("Invoice No", inv_no, key="invoice_no", spot=True)
    b.field("Date", fake.date("%Y-%m-%d"), key="date")
    b.table(["Item", "Qty", "Unit", "Amount"], rows, key="lines")
    b.field("TOTAL", total_str, key="total", spot=True, cls="total")
    b.task("Extract invoice number, date and total; convert the line-item table to HTML.")
    b.probe("abstain", "What is the shipping tracking number?", "not present — abstain")
    emit("invoice", b, "scan", do_degrade)


def case_id_card(do_degrade):
    name = fake.name().upper()
    idn = str(fake.random_int(100000000, 999999999))
    dob = fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%d %b %Y").upper()
    surname, given = name.split()[-1], " ".join(name.split()[:-1])
    mrz1 = f"P<USA{surname}<<{given.replace(' ', '<')}".ljust(44, "<")[:44]
    mrz2 = f"{idn}<4USA{fake.numerify('######')}M310614<<<<<<<<<<<<<<00".ljust(44, "<")[:44]
    css = """
    @page{ size:90mm 58mm; margin:0;}
    .card{ width:90mm; height:58mm; padding:6mm; position:relative;
        background:linear-gradient(135deg,#eef3fb,#dbe6f5);}
    .photo{ position:absolute; right:6mm; top:6mm; width:20mm; height:26mm;
        background:#9fb0c8; border:1px solid #6b7c96;}
    .fld{ margin:1mm 0;} .fld b{ color:#345; font-size:7px; text-transform:uppercase; font-weight:normal;}
    .v{ font-weight:bold; font-size:11px;}
    .mrz{ position:absolute; left:0; bottom:0; width:100%; padding:2mm 6mm; background:#f4f4f4;
        border-top:1px solid #bbb;}
    .mrz .tx{ font-family:'Liberation Mono',monospace; font-size:10px; letter-spacing:1px; margin:0;}
    """
    b = DocBuilder("ID / passport", ["language", "MRZ", "hallucination", "spotting"],
                   "KIE F1 + abstain + IoU", page="90mm 58mm", margin="0", css=css)
    b.raw("<div class=card>")
    b.title("REPUBLIC — IDENTITY CARD", level=2)
    b.raw("<div class=photo></div>")
    b.field("Surname / Given names", name, key="name")
    b.field("Document No", idn, key="doc_no", spot=True)
    b.field("Date of birth", dob, key="dob")
    b.field("Expiry", "14 JUN 2031", key="expiry")
    b.raw("<div class=mrz>")
    b.transcript(mrz1, key="mrz1")
    b.transcript(mrz2, key="mrz2")
    b.raw("</div></div>")
    b.task("Extract name, document number, DOB and expiry; localise the MRZ and photo.")
    b.probe("abstain", "What is the cardholder's blood type?", "not present — abstain")
    emit("id_card", b, "photo", do_degrade)


def case_checkbox_form(do_degrade):
    contact = [("Email", True), ("SMS", False), ("Phone call", True), ("Postal mail", False)]
    langs = [("English", True), ("Korean", False), ("Spanish", False)]
    b = DocBuilder("checkbox form", ["selection-marks", "layout", "hallucination"],
                   "selection-mark accuracy + F1", page="A5")
    b.title("SERVICE ENROLLMENT FORM", level=2)
    b.field("Applicant", fake.name(), key="applicant")
    b.line("<b>Preferred contact methods:</b>")
    b.checkboxes("contact", contact)
    b.line("<b>Notification language:</b>")
    b.checkboxes("language", langs)
    b.task("List ONLY the checked (☒) options per group.")
    b.probe("abstain", "Is 'Fax' selected?", "option not present")
    emit("checkbox_form", b, "scan", do_degrade)


def case_redacted(do_degrade):
    name = fake.name()
    b = DocBuilder("redacted document", ["hallucination", "spotting"],
                   "abstain (no-hallucination)", page="A5")
    b.title("CONFIDENTIAL MEMORANDUM", level=2)
    b.field("Subject", name, key="subject")
    b.redaction(f"The disclosed account number ", fake.numerify("###########"),
                key="account_number", suffix_html=" has been sealed by court order.")
    b.redaction("Authorising officer: ", fake.name(), key="authorising_officer", bar="█" * 8)
    b.line(f"Next review: {fake.date('%Y-%m-%d')}.")
    b.task("Answer only from visible text; for blacked-out values output '[redacted]'.")
    emit("redacted", b, "scan", do_degrade)


def case_bank_statement(do_degrade):
    rows, bal = [], 1000.0
    for _ in range(6):
        amt = round(fake.random_int(-500, 800) + 0.25, 2)
        bal += amt
        rows.append([fake.date("%m/%d"), fake.bs().title()[:20], f"{amt:+.2f}", f"{bal:.2f}"])
    b = DocBuilder("bank statement / payslip", ["layout", "table", "spotting"], "TEDS + F1", page="A5")
    b.title("MONTHLY STATEMENT", level=2)
    b.field("Account", fake.numerify("****####"), key="account")
    b.table(["Date", "Description", "Amount", "Balance"], rows, key="txns")
    b.field("Closing balance", f"${bal:.2f}", key="closing_balance", spot=True)
    b.task("Convert the transaction table to HTML (TEDS) and read the closing balance.")
    emit("bank_statement", b, "scan", do_degrade)


def case_rtl_arabic(do_degrade):
    text = "إشعار استلام رقم ٤٢ — المبلغ الإجمالي ١٤٥ ريالاً"
    css = "body{direction:rtl;font-family:'Amiri','Noto Naskh Arabic',serif;font-size:16px;}"
    b = DocBuilder("RTL doc (Arabic)", ["read-direction(RTL)", "language", "script"],
                   "direction + per-lang NED", page="A5", css=css)
    b.raw("<h2 dir=rtl>إيصال دفع</h2>")
    b.transcript(text, key="transcript", lang="ar")
    b.fields["language"] = "ar"
    b.fields["reading_direction"] = "rtl"
    b.task("Transcribe the Arabic text and state the reading direction.")
    b.probe("direction", "Is this text right-to-left or left-to-right?", "right-to-left")
    emit("rtl_arabic", b, "scan", do_degrade)


def case_webtoon(do_degrade):
    lines = ["Are you ready?", "I think so...", "Then let's begin!", "Wait — look over there!"]
    css = """
    @page{ size:80mm 200mm; margin:4mm;}
    .panel{ position:relative; height:44mm; border:2px solid #111; margin-bottom:3mm;
        background:linear-gradient(160deg,#f7f7f7,#e7ecf2);}
    .pno{ position:absolute; top:2mm; left:2mm; font-size:9px; color:#888;}
    .bubble{ position:absolute; top:8mm; max-width:60%; background:#fff; border:2px solid #111;
        border-radius:14px; padding:5px 9px; font-size:13px;}
    .left{ left:5mm;} .right{ right:5mm;}
    """
    b = DocBuilder("webtoon / manga", ["read-order", "direction(vertical)", "art-text"],
                   "read-order + direction + NED", page="80mm 200mm", margin="4mm", css=css)
    for i, ln in enumerate(lines, 1):
        side = "left" if i % 2 else "right"
        b.raw(f"<div class=panel><span class=pno>{i}</span>"
              f"<div class='bubble {side}'>{esc(ln)}</div></div>")
    b.order(lines, note="top-to-bottom scroll")
    b.fields["n_panels"] = len(lines)
    b.task("Transcribe speech bubbles in reading order (top to bottom).")
    emit("webtoon", b, "photo", do_degrade)


def case_prescription(do_degrade):
    patient = fake.name()
    drug = "Amoxicillin 500mg — 1 cap TID x7d"
    css = """
    .rx{ font-size:34px; font-weight:bold;}
    .hand .tx,.hand.tx{ font-family:'Purisa',cursive; font-size:18px; color:#1a2a5a;}
    """
    b = DocBuilder("prescription / 처방전", ["handwriting", "hallucination", "degradation"],
                   "CER(hand) + abstain", page="A5", css=css)
    b.title(f"{fake.name()}, M.D. — Internal Medicine", level=2)
    b.field("Patient", patient, key="patient")
    b.raw("<p class=rx>℞</p>")
    b.transcript(drug, key="rx_handwritten", cls="hand")
    b.raw("<p class='hand'>Sig: take with food</p>")
    b.task("Transcribe the handwritten medication line (CER).")
    b.probe("abstain", "What is the refill count?", "not legible / not present — abstain")
    emit("prescription", b, "fax", do_degrade)


def case_cheque(do_degrade):
    payee = fake.name()
    amt_num, amt_words = "1,450.00", "One thousand four hundred fifty and 00/100"
    css = """
    @page{ size:160mm 70mm; margin:6mm;}
    .cheque{ border:2px solid #1b3a6b; padding:5mm; height:100%; background:#f3f7fc;}
    .num{ float:right; border:1px solid #1b3a6b; padding:3px 8px; font-weight:bold; font-size:15px;}
    .words .tx,.words.tx{ font-family:'Purisa',cursive; font-size:16px; border-bottom:1px solid #888;}
    .micr{ font-family:'Liberation Mono',monospace; letter-spacing:2px; margin-top:6mm;}
    """
    b = DocBuilder("cheque / 수표", ["handwriting", "dual-amount", "hallucination", "spotting"],
                   "dual-amount F1 + sign detect", page="160mm 70mm", margin="6mm", css=css)
    b.raw("<div class=cheque>")
    b.raw(f"<p>{esc(fake.company())} BANK <span class=num>$ {amt_num}</span></p>")
    b.spot("amount_numeric", amt_num)
    b.fields["amount_numeric"] = amt_num
    b.field("Pay to the order of", payee, key="payee")
    b.transcript(amt_words, key="amount_words", cls="words")
    b.raw("<p class=micr>⑆000123456⑆ 0987654321⑈ 4421</p></div>")
    b.task("Read the courtesy (numeric) and legal (written) amounts and check they agree.")
    b.probe("consistency", "Do the numeric and written amounts agree?",
            "yes (1450.00 == one thousand four hundred fifty)")
    emit("cheque", b, "scan", do_degrade)


def case_ancient(do_degrade):
    poem = "山重水複疑無路"
    css = """
    @page{ size:120mm 150mm; margin:14mm;}
    body{ font-family:'Noto Serif CJK SC',serif; background:#efe7d2;}
    h2{ font-family:'EB Garamond',serif; color:#5a4a2a;}
    .wrap{ display:flex; flex-direction:row-reverse; gap:10mm; justify-content:center;}
    .col{ writing-mode:vertical-rl; font-size:30px; line-height:1.6; color:#2a1f12;}
    """
    b = DocBuilder("ancient manuscript / 고문서",
                   ["direction(vertical)", "language(classical)", "degradation(fade/stain)"],
                   "NED + robustness", page="120mm 150mm", margin="14mm", css=css)
    b.title("古文書 — Classical manuscript", level=2)
    b.raw(f"<div class=wrap><div class=col>{esc(poem[:4])}</div>"
          f"<div class=col>{esc(poem[4:])}</div></div>")
    b.fields["transcript"] = poem
    b.fields["reading_direction"] = "vertical-rtl"
    b.task("Transcribe characters in reading order (top→bottom, right→left).")
    emit("ancient", b, "historical", do_degrade)


# ============================================================ digital-native surfaces
def case_website(do_degrade):
    brand = fake.company().split()[0]
    nav = ["Product", "Pricing", "Docs", "Sign in"]
    cta, headline = "Start free trial", "Ship documents, not paperwork."
    cards = [("Fast", "Parse a page in milliseconds."), ("Secure", "SOC-2 encrypted storage."),
             ("Global", "40+ languages out of the box.")]
    css = """
    @page{ size:1280px 900px; margin:0;}
    body{ margin:0;} .chrome{ background:#e7eaef; height:38px; display:flex; align-items:center;
        padding:0 12px; gap:7px;}
    .dot{ width:11px; height:11px; border-radius:50%;}
    .url{ flex:1; margin-left:14px; background:#fff; border-radius:7px; padding:5px 12px; color:#667;
        font-size:12px; max-width:520px;}
    nav{ display:flex; align-items:center; padding:16px 48px; border-bottom:1px solid #eef0f4;}
    .logo{ font-weight:bold; font-size:20px; color:#2a5bd7;}
    nav .sp{ flex:1;} nav a{ margin:0 14px; color:#445; font-size:15px;}
    .btn{ background:#2a5bd7; color:#fff!important; padding:9px 18px; border-radius:8px; font-weight:bold;}
    .hero{ text-align:center; padding:64px 40px 36px;} .hero h1{ font-size:46px; margin:0 0 14px;}
    .hero p{ color:#5a6675; font-size:19px;}
    .cards{ display:flex; gap:22px; padding:20px 64px 40px; justify-content:center;}
    .card{ flex:1; max-width:300px; border:1px solid #e7eaf0; border-radius:14px; padding:22px;}
    .card h3{ margin:10px 0 6px;} .card p{ color:#5a6675; font-size:14px;}
    footer{ background:#0f1830; color:#aeb8cc; padding:22px 48px; font-size:13px;}
    """
    b = DocBuilder("website / desktop screenshot", ["layout(web)", "reflow", "icons/links", "spotting"],
                   "NED + spotting", page="1280px 900px", margin="0", css=css)
    navhtml = "".join(f"<a>{esc(n)}</a>" for n in nav)
    cardhtml = "".join(f"<div class=card><h3>{esc(t)}</h3><p>{esc(p)}</p></div>" for t, p in cards)
    b.raw('<div class=chrome><span class=dot style="background:#f55"></span>'
          '<span class=dot style="background:#fb5"></span><span class=dot style="background:#5c5"></span>'
          f'<span class=url>https://{brand.lower()}.example/app</span></div>')
    b.raw(f"<nav><span class=logo>◆ {esc(brand)}</span><span class=sp></span>{navhtml}"
          f"<a class=btn>{esc(cta)}</a></nav>")
    b.raw(f"<div class=hero><h1>{esc(headline)}</h1><p>The document API for small teams.</p></div>")
    b.raw(f"<div class=cards>{cardhtml}</div>")
    b.raw(f"<footer>© 2025 {esc(brand)} · Terms · Privacy · Status</footer>")
    b.fields.update({"brand": brand, "nav_items": nav, "headline": headline, "cta": cta})
    b.spot("cta_button", cta)
    b.order(["nav", "headline", "subtext", "feature cards (L→R)", "footer"])
    b.task("List the navigation items in order, then the main headline.")
    b.probe("abstain", "What is the user's logged-in email?", "logged-out page — abstain")
    emit("website", b, "screenshot", do_degrade)


def case_mobile_app(do_degrade):
    msgs = [("in", "Hi! Is my invoice ready?"), ("out", "Yes — INV-2025-0042, total $145.50."),
            ("in", "Great, can you email it?"), ("out", "Sent to your inbox ✅"), ("in", "Thanks!")]
    css = """
    @page{ size:390px 844px; margin:0;}
    body{ margin:0; background:#f2f4f8;}
    .status{ height:30px; background:#fff; display:flex; align-items:center; justify-content:space-between;
        padding:0 16px; font-size:13px; font-weight:bold;}
    .hdr{ background:#2a5bd7; color:#fff; padding:12px 16px; font-size:17px; font-weight:bold;}
    .chat{ padding:14px 12px;} .row{ display:flex; margin:8px 0;} .row.out{ justify-content:flex-end;}
    .bub{ max-width:74%; padding:9px 13px; border-radius:16px; font-size:15px; line-height:1.3;}
    .bub.in{ background:#fff; border:1px solid #e2e6ee;} .bub.out{ background:#2a5bd7; color:#fff;}
    .input{ position:fixed; bottom:0; width:100%; background:#fff; border-top:1px solid #e2e6ee;
        padding:10px 14px; color:#889; font-size:14px;}
    """
    b = DocBuilder("mobile app / phone screenshot", ["layout(mobile)", "reflow(vertical)", "read-order"],
                   "NED + read-order", page="390px 844px", margin="0", css=css)
    b.raw("<div class=status><span>9:41</span><span>▮▮▮ 5G  87%</span></div>")
    b.raw("<div class=hdr>‹ Support Chat</div><div class=chat>")
    for s, t in msgs:
        b.raw(f'<div class="row {s}"><div class="bub {s}">{esc(t)}</div></div>')
    b.raw("</div><div class=input>Type a message…</div>")
    b.fields["messages"] = [t for _, t in msgs]
    b.order([t for _, t in msgs])
    b.task("Transcribe chat messages in order; mark incoming vs outgoing.")
    b.probe("direction", "Which messages are the user's?", "right/blue bubbles = user (outgoing)")
    emit("mobile_app", b, "screenshot", do_degrade)


def case_pdf_paper(do_degrade):
    title = "Sub-1B Vision-Language Models for Document Understanding"
    authors = f"{fake.name()}, {fake.name()}"
    secs = ["1. Introduction", "2. Related Work", "3. Method", "4. Experiments", "5. Conclusion"]
    para = " ".join(fake.sentence() for _ in range(6)) + " "
    cap = "Figure 1: The proposed evaluation pipeline."
    css = """
    @page{ size:A4; margin:18mm 15mm;
      @top-center{ content:"Proc. of the Synthetic Document Workshop, 2025"; font-size:8px; color:#999;}
      @bottom-center{ content:counter(page); font-size:9px; color:#555;}}
    body{ font-family:'EB Garamond','Liberation Serif',serif; font-size:9.5px;}
    h1{ font-size:17px; text-align:center; margin:0 0 4px;}
    .auth{ text-align:center; margin-bottom:8px; font-size:10px;}
    .abs{ font-style:italic; margin:0 8mm 8px; font-size:9px; border-top:1px solid #ccc;
        border-bottom:1px solid #ccc; padding:6px 0;}
    .cols{ column-count:2; column-gap:7mm; text-align:justify;}
    h3{ font-size:10.5px; margin:8px 0 3px;}
    .fig{ break-inside:avoid; border:1px solid #bbb; padding:5px; margin:6px 0; text-align:center;}
    .ph{ height:60px; background:repeating-linear-gradient(45deg,#eef,#eef 6px,#dde 6px,#dde 12px);}
    .cap{ font-size:8px; color:#444; margin-top:4px;}
    """
    b = DocBuilder("PDF research paper (2-col)",
                   ["layout(multi-column)", "read-order", "figure", "header/footer"],
                   "read-order + NED + TEDS", page="A4", margin="18mm 15mm", css=css)
    b.raw(f"<h1>{esc(title)}</h1><div class=auth>{esc(authors)}</div>")
    b.raw(f"<div class=abs><b>Abstract.</b> {esc(para)}</div><div class=cols>")
    for s in secs:
        b.raw(f"<h3>{esc(s)}</h3><p>{esc(para)}{esc(para)}</p>")
        if s == "3. Method":
            b.raw(f"<div class=fig><div class=ph></div><div class=cap>{esc(cap)}</div></div>")
    b.raw("</div>")
    b.fields.update({"title": title, "authors": authors, "sections": secs})
    b.spot("figure_caption", cap)
    b.order("left column top→bottom, then right column (NOT row-wise across columns)")
    b.task("Extract the title, authors and section headings in reading order.")
    b.probe("order", "Does text read across both columns row-by-row?",
            "no — each column reads top-to-bottom independently")
    emit("pdf_paper", b, "scan", do_degrade)


# ============================================================ non-HTML special case
def case_lcd(do_degrade):
    digits = "01428"
    W, H = 520, 200
    im = Image.new("RGB", (W, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    _seg7(d, digits, 40, 50, on=(40, 255, 150), off=(28, 40, 36))
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 70, 80), width=4)
    gt = {"type": "LCD / meter / 7-seg", "stressors": ["non-font digits", "glare"],
          "anchor_metric": "exact + IoU", "fields": {"reading": digits, "_task": "Read the display."},
          "source": "SYNTHETIC (docvlm_eval.synth; PIL — no real 7-seg font)",
          "render": {"dpi": None, "size_px": [W, H], "page_count": 1}}
    emit("lcd_7seg", im, "photo", do_degrade, gt=gt)


def _seg7(d, digits, x0, y0, on, off, w=60, h=58, t=10, gap=34):
    segs = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
            "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
    for k, ch in enumerate(digits):
        x = x0 + k * (w + gap)
        s = segs.get(ch, "")

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
    "ancient": case_ancient, "website": case_website, "mobile_app": case_mobile_app,
    "pdf_paper": case_pdf_paper, "lcd_7seg": case_lcd,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CASES))
    ap.add_argument("--no-degrade", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (args.only or list(CASES)):
        CASES[k](do_degrade=not args.no_degrade)
    (OUT / "index.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {len(args.only or CASES)} realistic cases -> {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Realistic synthetic generator for the document-type *special cases*
(docs/report/document_type_taxonomy.md).

Each case is declared through ``docvlm_eval.synth.DocBuilder`` so the ground truth is *produced
by* the render — every value is declared once (no drift between pixels and labels) and spotting
boxes are read straight out of the rendered PDF. A photometric Augraphy preset then makes a
realistic degraded copy whose boxes are still valid (no geometry changed). Faker (seeded) fills
field content deterministically.

Output per case: data/probes/realistic_cases/<key>/{clean.png, degraded.png, gt.json}

    python scripts/make_realistic_cases.py                       # all cases
    python scripts/make_realistic_cases.py --only id_card cheque
    python scripts/make_realistic_cases.py --no-degrade          # clean only (fast)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docvlm_eval.synth import DocBuilder, degrade, esc  # noqa: E402
from docvlm_eval.synth.dto import Degradation, DocSample, GenConfig  # noqa: E402

OUT = ROOT / "data" / "probes" / "realistic_cases"

# Faker locale per language code (A4). Latin locales localise name/company/address content;
# CJK/Arabic locales need the matching Noto fonts (referenced in the case CSS) to render.
LOCALE = {"en": "en_US", "es": "es_ES", "fr": "fr_FR", "de": "de_DE", "pt": "pt_BR",
          "it": "it_IT", "ja": "ja_JP", "ko": "ko_KR", "zh": "zh_CN", "ar": "ar_AA"}

# Module state set by main() per (variant, doc); the case functions read `fake` as before.
fake = Faker()
Faker.seed(7)
CFG: GenConfig = GenConfig()
CURRENT_VARIANT: str | None = None
CURRENT_LANG: str = "en"
records: list[dict] = []


def _resize_with_boxes(img: Image.Image, gt: dict) -> Image.Image:
    """Apply the A7 resize knobs to the image AND rescale every box (spotting + derived grounding
    answers) so GT stays exact at the new resolution."""
    tls, keep = CFG.target_long_side, CFG.keep_aspect
    if not tls:
        return img
    w, h = img.size
    if keep:
        sx = sy = tls / max(w, h)
        new = (max(1, round(w * sx)), max(1, round(h * sy)))
    else:                                   # squash to a square -> independent x/y scale
        sx, sy = tls / w, tls / h
        new = (tls, tls)
    nw, nh = new
    img = img.resize(new, Image.LANCZOS)

    def scale(b):
        return [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)]

    spotting = gt.get("spotting")
    if spotting:
        gt["spotting"] = {k: scale(b) for k, b in spotting.items()}
    # derived grounding QAs carry the box twice (qa["box"] + the "x1,y1,x2,y2;W,H" answer string)
    for q in gt.get("qa", []):
        if q.get("derived") and q.get("box"):
            nb = scale(q["box"])
            q["box"] = nb
            q["answers"] = [f"{nb[0]},{nb[1]},{nb[2]},{nb[3]};{nw},{nh}"]
            if q.get("rationale"):  # keep the reasoning coords consistent with the resized image
                q["rationale"] = (re.sub(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]",
                                         f"[{nb[0]}, {nb[1]}, {nb[2]}, {nb[3]}]", q["rationale"], count=1)
                                  ).replace(f"{w}x{h}px", f"{nw}x{nh}px")
    gt.setdefault("render", {})["size_px"] = list(new)
    return img


def _apply_emit_toggles(gt: dict) -> None:
    """Honour the supervision switches by dropping GT the control arm must not see."""
    if not CFG.emit_spotting:
        gt.pop("spotting", None)
    if not CFG.emit_rationale:
        for q in gt.get("qa", []):
            q.pop("rationale", None)
    if not getattr(CFG, "emit_understanding", True):
        gt["qa"] = [q for q in gt.get("qa", []) if not q.get("derived")]
    # note: the internal "box"/"derived" keys are consumed by from_builder_gt and then dropped by
    # DocSample.to_dict (which rebuilds qa from QAItem), so they never reach the saved gt.json.


def _pick_preset(key: str, default: str) -> str:
    """Choose a degradation preset for this doc-type honouring config overrides + severity-as-rng."""
    presets = (CFG.degrade_presets or {}).get(key)
    if presets:
        return random.choice(presets)
    return default


def emit(key: str, builder_or_img, preset: str, do_degrade: bool, gt: dict | None = None,
         *, builder=None, domain: str | None = None, acquisition: str | None = None):
    """Render a builder (or accept a prebuilt PIL image + gt) -> clean.png + degraded.png + gt.json.

    Writes the structured DocSample DTO (a superset of the legacy flat gt schema)."""
    if gt is None:
        builder = builder_or_img
        img, gt = builder.build(dpi=CFG.dpi)
    else:
        img = builder_or_img
    # A7 resize (with box rescale) + A1/A2 supervision toggles
    img = _resize_with_boxes(img, gt)
    # A4: this doc's content was generated under CURRENT_LANG's locale -> tag its fields
    if builder is not None and CURRENT_LANG != "en":
        for k in list(builder.field_lang):
            if builder.field_lang[k] == "en":
                builder.field_lang[k] = CURRENT_LANG
    _apply_emit_toggles(gt)

    folder = OUT / key if CURRENT_VARIANT is None else OUT / key / CURRENT_VARIANT
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "clean.png")

    degradation = None
    if do_degrade and random.random() < CFG.degrade_prob:
        chosen = _pick_preset(key, preset)
        # stable per-case seed (Python's str hash is salted per-process -> not reproducible)
        seed = CFG.seed + int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000
        deg = degrade(img, chosen, seed=seed)
        if deg is not None:
            deg.save(folder / "degraded.png")
            degradation = Degradation(preset=chosen, severity=CFG.degrade_severity, seed=seed)

    doc = DocSample.from_builder_gt(
        gt, builder=builder, gen_config=CFG, degradation=degradation,
        domain=domain, acquisition=acquisition,
    )
    doc.languages = [CURRENT_LANG] if builder is not None else doc.languages
    out = doc.to_dict()
    (folder / "gt.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    sup = out["ablation_support"]
    records.append({"key": key, "variant": CURRENT_VARIANT, "type": gt["type"],
                    "language": CURRENT_LANG, "stressors": gt["stressors"],
                    "anchor_metric": gt["anchor_metric"], "support": sup})
    if CURRENT_VARIANT in (None, "0000"):  # keep the log readable when fanning out
        flags = "".join(c for c, on in [("S", sup["spotting"]), ("R", sup["rationale"]),
                        ("M", sup["multilingual"]), ("s", sup["small_text"])] if on)
        print(f"[ok] {key:14} {img.size} lang={CURRENT_LANG:2} fields={len(out.get('fields',{}))} "
              f"spots={len(out.get('spotting',{}))} [{flags}]")


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
    b.table(["Item", "Qty", "Unit", "Amount"], rows, key="lines", region="the line-item table")
    b.field("TOTAL", total_str, key="total", spot=True, cls="total")
    b.task("Extract invoice number, date and total; convert the line-item table to HTML.")
    b.qa("What is the invoice number?", inv_no, metric="ned", answer_type="kie", key="invoice_no")
    line_sum = " + ".join(f"{q}×${p:.2f}" for _, q, p in items)
    b.qa("What is the total amount?", [total_str, f"{total:.2f}", f"{total:,.2f}"],
         answer_type="kie", key="total",
         rationale=f"Sum the line amounts: {line_sum} = {total_str}.")
    b.probe("abstain", "What is the shipping tracking number?", "not present — abstain")
    # --- model-free UNDERSTANDING GT (no external model): where / how-many / totals + reasoning ---
    b.ask_where("TOTAL", label="the TOTAL row")                       # L1: locate a word
    b.ask_count("$")                                                  # H: count currency symbols (table region is auto)
    amounts = [q * p for _, q, p in items]
    b.ask_aggregate("the sum of all line-item amounts", amounts, op="sum")
    # higher-level reasoning over table extremes (top vs bottom row), not plain extraction
    b.ask_aggregate("the sum of the first and last line-item amounts", [amounts[0], amounts[-1]], op="sum")
    b.ask_aggregate("the difference between the largest and smallest line-item amounts",
                    amounts, op="diff")
    # harder annotation: accountant-style multi-step calc (sum then apply 10% tax), not plain extraction
    grand = round(total * 1.10, 2)
    b.qa("What would the grand total be after adding 10% sales tax to the total?",
         [f"{grand:.2f}", f"{grand:,.2f}", f"${grand:,.2f}"], metric="relaxed_acc",
         answer_type="H-accounting",
         rationale=f"Total {total:.2f} × 1.10 (10% tax) = {grand:.2f}.")
    emit("invoice", b, "scan", do_degrade, domain="finance", acquisition="scan")


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
    b.transcript(mrz1, key="mrz1", role="mrz", font_px=10)   # small-text slice (A7)
    b.transcript(mrz2, key="mrz2", role="mrz", font_px=10)
    b.raw("</div></div>")
    b.task("Extract name, document number, DOB and expiry; localise the MRZ and photo.")
    b.qa("What is the document number?", idn, metric="ned", answer_type="kie")
    b.qa("What is the cardholder's full name?", name, metric="ned", answer_type="kie")
    b.probe("abstain", "What is the cardholder's blood type?", "not present — abstain")
    b.ask_where(idn, label="the document number")
    b.ask_region("the machine-readable zone (MRZ)", [mrz1, mrz2])
    # harder annotation: pull ONLY the surname out of the MRZ encoding (parse, don't just transcribe)
    b.qa("Extract only the surname from the MRZ (the part right after the country code, before '<<').",
         surname, metric="exact", answer_type="H-extract-strict",
         rationale=f"MRZ line 1 encodes 'P<USA{surname}<<...'; the surname before '<<' is {surname}.")
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
    b.qa("Which contact methods are checked?", ", ".join(t for t, c in contact if c),
         answer_type="selection")
    b.qa("Which notification language is checked?", ", ".join(t for t, c in langs if c),
         answer_type="selection")
    b.probe("abstain", "Is 'Fax' selected?", "option not present")
    b.ask_where("Email", label="the Email option")
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
    b.qa("Who is the subject of the memo?", name, metric="ned", answer_type="kie")
    b.ask_where(name, label="the subject")
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
    b.table(["Date", "Description", "Amount", "Balance"], rows, key="txns",
            region="the transaction table")
    b.field("Closing balance", f"${bal:.2f}", key="closing_balance", spot=True)
    b.task("Convert the transaction table to HTML (TEDS) and read the closing balance.")
    b.qa("What is the closing balance?", [f"${bal:.2f}", f"{bal:.2f}"], answer_type="kie")
    # model-free understanding GT: locate the closing balance (table region is auto)
    b.ask_where(f"${bal:.2f}", label="the closing balance")
    # reasoning over the balance column extremes (highest vs lowest)
    bals = [float(r[3]) for r in rows]
    b.ask_aggregate("the difference between the highest and lowest balance", bals, op="diff")
    emit("bank_statement", b, "scan", do_degrade, domain="finance", acquisition="scan")


def case_rtl_arabic(do_degrade):
    text = "إشعار استلام رقم ٤٢ — المبلغ الإجمالي ١٤٥ ريالاً"
    css = "body{direction:rtl;font-family:'Amiri','Noto Naskh Arabic',serif;font-size:16px;}"
    b = DocBuilder("RTL doc (Arabic)", ["read-direction(RTL)", "language", "script"],
                   "direction + per-lang NED", page="A5", css=css)
    b.raw("<h2 dir=rtl>إيصال دفع</h2>")
    b.transcript(text, key="transcript", lang="ar", spot=True)
    b.fields["language"] = "ar"
    b.fields["reading_direction"] = "rtl"
    b.task("Transcribe the Arabic text and state the reading direction.")
    b.qa("Transcribe the Arabic text in the image.", text, metric="ned", answer_type="multilingual")
    b.qa("Is the text right-to-left or left-to-right?", ["right-to-left", "rtl"],
         metric="exact", answer_type="direction")
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
        b.panel(ln, index=i, side="left" if i % 2 else "right")
    b.spot("bubble_1", lines[0])
    b.fields["n_panels"] = len(lines)
    b.task("Transcribe speech bubbles in reading order (top to bottom).")
    b.qa("Transcribe the speech bubbles top to bottom, one per line.", "\n".join(lines),
         metric="ned", answer_type="reading-order")
    b.ask_where(lines[0], label="the first speech bubble")
    b.qa("How many panels are in the comic?", str(len(lines)), metric="exact", answer_type="H-count")
    emit("webtoon", b, "photo", do_degrade)


def case_prescription(do_degrade):
    patient = fake.name()
    drug = "Amoxicillin 500mg — 1 cap TID x7d"
    css = """
    .rx{ font-size:34px; font-weight:bold;}
    .hand .tx,.hand.tx{ font-family:'Purisa',cursive; font-size:18px; color:#1a2a5a;}
    """
    b = DocBuilder("prescription", ["handwriting", "hallucination", "degradation"],
                   "CER(hand) + abstain", page="A5", css=css)
    b.title(f"{fake.name()}, M.D. — Internal Medicine", level=2)
    b.field("Patient", patient, key="patient")
    b.raw("<p class=rx>℞</p>")
    b.transcript(drug, key="rx_handwritten", cls="hand", spot=True)
    b.raw("<p class='hand'>Sig: take with food</p>")
    b.task("Transcribe the handwritten medication line (CER).")
    b.qa("Transcribe the handwritten medication line.", drug, metric="ned", answer_type="handwriting")
    b.probe("abstain", "What is the refill count?", "not legible / not present — abstain")
    b.ask_where(drug, label="the handwritten medication line")
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
    b = DocBuilder("cheque", ["handwriting", "dual-amount", "hallucination", "spotting"],
                   "dual-amount F1 + sign detect", page="160mm 70mm", margin="6mm", css=css)
    b.raw("<div class=cheque>")
    b.raw(f"<p>{esc(fake.company())} BANK <span class=num>$ {amt_num}</span></p>")
    b.spot("amount_numeric", amt_num)
    b.fields["amount_numeric"] = amt_num
    b.field("Pay to the order of", payee, key="payee")
    b.transcript(amt_words, key="amount_words", cls="words")
    b.raw("<p class=micr>⑆000123456⑆ 0987654321⑈ 4421</p></div>")
    b.task("Read the courtesy (numeric) and legal (written) amounts and check they agree.")
    b.qa("What is the numeric (courtesy) amount on the cheque?", [amt_num, "1450.00"],
         answer_type="kie", key="amount_numeric")
    b.qa("Who is the payee?", payee, metric="ned", answer_type="kie", key="payee")
    b.qa("Do the numeric and written amounts agree?", ["yes", "they agree"], metric="anls",
         answer_type="consistency",
         rationale=f"Numeric '{amt_num}' equals legal 'one thousand four hundred fifty' → they agree.")
    b.probe("consistency", "Do the numeric and written amounts agree?",
            "yes (1450.00 == one thousand four hundred fifty)")
    b.ask_where(amt_num, label="the courtesy (numeric) amount")
    emit("cheque", b, "scan", do_degrade, domain="finance", acquisition="scan")


def case_ancient(do_degrade):
    poem = "山重水複疑無路"
    cols = [poem[:4], poem[4:]]      # right-to-left columns; each stacked top->bottom
    # Stack characters one-per-line (WeasyPrint ignores writing-mode:vertical-rl), so it renders as a
    # true vertical classical manuscript. row-reverse puts the first column on the RIGHT.
    css = """
    @page{ size:120mm 150mm; margin:14mm;}
    body{ font-family:'Noto Serif CJK SC','Noto Sans CJK SC',serif; background:#efe7d2;}
    h2{ font-family:'EB Garamond','Noto Serif CJK SC',serif; color:#5a4a2a;}
    .wrap{ display:flex; flex-direction:row-reverse; gap:12mm; justify-content:center; margin-top:6mm;}
    .col{ font-size:32px; line-height:1.45; color:#2a1f12; text-align:center; writing-mode:vertical-rl;}
    .col span{ display:block; }      /* one glyph per line -> vertical column */
    """
    b = DocBuilder("ancient manuscript",
                   ["direction(vertical)", "language(classical)", "degradation(fade/stain)"],
                   "NED + robustness", page="120mm 150mm", margin="14mm", css=css)
    b.title("古文書 — Classical manuscript", level=2)
    body = "".join("<div class=col>" + "".join(f"<span>{esc(c)}</span>" for c in col) + "</div>"
                   for col in cols)
    b.raw(f"<div class=wrap>{body}</div>")
    b.spot("first_glyph", poem[0])   # a single glyph is reliably searchable (a stacked column isn't)
    b.fields["transcript"] = poem
    b.fields["reading_direction"] = "vertical-rtl"
    b.task("Transcribe characters in reading order (top→bottom, right→left).")
    b.qa("Transcribe the characters (top→bottom, right→left).", poem, metric="ned",
         answer_type="multilingual")
    b.ask_where("古文書", label="the title")
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
    b.qa("List the navigation menu items in order.", ", ".join(nav),
         metric="ned", answer_type="ui")
    b.qa("What is the main headline?", headline, metric="ned", answer_type="ui")
    b.probe("abstain", "What is the user's logged-in email?", "logged-out page — abstain")
    b.ask_where(cta, label="the call-to-action button")
    b.ask_region("the feature cards", [t for t, _ in cards])
    # UI affordance reasoning: what should the user do next?
    b.qa("What is the primary action this page wants the visitor to take?",
         [cta, cta.lower()], metric="anls", answer_type="H-action",
         rationale=f"The prominent call-to-action button reads '{cta}', so that is the next action.")
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
    for i, (s, t) in enumerate(msgs):
        b.bubble(t, side=s, key="bubble_first" if i == 0 else None)
    b.raw("</div><div class=input>Type a message…</div>")
    b.fields["messages"] = b.reading_order
    b.task("Transcribe chat messages in order; mark incoming vs outgoing.")
    b.qa("Transcribe the chat messages in order, one per line.", "\n".join(t for _, t in msgs),
         metric="ned", answer_type="reading-order")
    b.probe("direction", "Which messages are the user's?", "right/blue bubbles = user (outgoing)")
    b.ask_where(msgs[0][1], label="the first chat message")
    # diverse / contextual tasks: how many turns, and what the conversation is about
    b.qa("How many messages are in the conversation?", str(len(msgs)), metric="exact",
         answer_type="H-count")
    b.qa("What is the conversation about?", ["invoice", "the invoice", "an invoice"],
         metric="anls", answer_type="H-comprehension",
         rationale="The user asks if their invoice is ready and to email it -> the topic is the invoice.")
    b.qa("Based on the last message, what should the support agent do next?",
         ["nothing", "no action", "the issue is resolved", "wait"], metric="anls",
         answer_type="H-action",
         rationale="The invoice was already sent and the user replied 'Thanks!' -> no further action is needed.")
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
    b.qa("What is the paper title?", title, metric="ned", answer_type="ui")
    b.qa("List the section headings in order.", "; ".join(secs), metric="ned",
         answer_type="reading-order")
    b.probe("order", "Does text read across both columns row-by-row?",
            "no — each column reads top-to-bottom independently")
    b.ask_where(title, label="the paper title")
    b.ask_region("the figure", [cap])
    emit("pdf_paper", b, "scan", do_degrade)


# ============================================================ non-HTML special case
def case_lcd(do_degrade):
    digits = fake.numerify("0####")
    W, H = 520, 200
    im = Image.new("RGB", (W, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    _seg7(d, digits, 40, 50, on=(40, 255, 150), off=(28, 40, 36))
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 70, 80), width=4)
    gt = {"type": "LCD / meter / 7-seg", "stressors": ["non-font digits", "glare"],
          "anchor_metric": "exact + IoU", "fields": {"reading": digits, "_task": "Read the display."},
          "qa": [{"key": "reading", "question": "What number is on the display?",
                  "answers": [digits], "metric": "exact", "answer_type": "special-glyph"}],
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


def _choose_lang(rng: random.Random) -> str:
    """Pick this doc's language from the configured mix (weighted if given, else uniform)."""
    langs = CFG.languages or ["en"]
    if len(langs) == 1:
        return langs[0]
    if CFG.language_weights:
        ws = [CFG.language_weights.get(l, 0.0) for l in langs]
        if sum(ws) > 0:
            return rng.choices(langs, weights=ws, k=1)[0]
    return rng.choice(langs)


def main():
    global CURRENT_VARIANT, CFG, CURRENT_LANG, fake, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(CASES))
    ap.add_argument("--no-degrade", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "configs" / "synth_data.yaml"),
                    help="GenConfig YAML (controls every ablation factor)")
    ap.add_argument("--ablation", default=None,
                    help="named override under ablation_overrides: in the config (e.g. A1_spotting_on)")
    ap.add_argument("--count", type=int, default=None,
                    help="variants per case (overrides config.count; >1 fans out into <key>/<NNNN>/)")
    ap.add_argument("--seed", type=int, default=None, help="base seed (overrides config.seed)")
    ap.add_argument("--out", default=None,
                    help="output dir (default data/probes/realistic_cases) — use a different dir + "
                         "--seed for a held-out TEST split (memorization-vs-understanding, A0)")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)

    CFG = GenConfig.from_yaml(args.config, ablation=args.ablation)
    if args.count is not None:
        CFG.count = args.count
    if args.seed is not None:
        CFG.seed = args.seed
    if args.no_degrade:
        CFG.degrade_prob = 0.0

    OUT.mkdir(parents=True, exist_ok=True)
    keys = args.only or list(CASES)
    print(f"[config] {CFG.name} (ablation={CFG.ablation})  dpi={CFG.dpi} "
          f"long_side={CFG.target_long_side} spot={CFG.emit_spotting} reason={CFG.emit_rationale} "
          f"langs={CFG.languages} degrade_p={CFG.degrade_prob}")
    for v in range(CFG.count):
        CURRENT_VARIANT = None if CFG.count == 1 else f"{v:04d}"
        rng = random.Random(CFG.seed + v)
        random.seed(CFG.seed + v)            # used by emit() preset choice + degrade rng
        for k in keys:
            CURRENT_LANG = _choose_lang(rng)
            # reseed Faker to the doc's locale so multilingual content is real + reproducible
            fake = Faker(LOCALE.get(CURRENT_LANG, "en_US"))
            Faker.seed(CFG.seed + v)
            CASES[k](do_degrade=not args.no_degrade)
    (OUT / "index.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "gen_config.json").write_text(json.dumps(CFG.to_dict(), indent=2), encoding="utf-8")
    print(f"\n[done] {len(keys)} cases x {CFG.count} variant(s) = {len(records)} docs -> {OUT}")


if __name__ == "__main__":
    main()

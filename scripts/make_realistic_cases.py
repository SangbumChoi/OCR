#!/usr/bin/env python3
"""Realistic synthetic generator for the document-type *special cases*
(docs/report/document_type_taxonomy.md).

Each case is declared through ``docvlm_eval.synth.DocBuilder`` so the ground truth is *produced
by* the render — every value is declared once (no drift between pixels and labels) and spotting
boxes are read straight out of the rendered PDF. Photo-style cases may receive a deterministic
perspective warp whose exact homography is applied to every box before a photometric Augraphy
preset makes a realistic degraded copy. Faker (seeded) fills field content deterministically.

Output per case: data/probes/realistic_cases/<key>/{clean.png, degraded.png, gt.json}

    python scripts/make_realistic_cases.py                       # all cases
    python scripts/make_realistic_cases.py --only id_card cheque
    python scripts/make_realistic_cases.py --no-degrade          # clean only (fast)
"""

from __future__ import annotations

import argparse
import copy
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
from docvlm_eval.synth import (  # noqa: E402
    BundleDocument,
    DifficultySpec,
    DocBuilder,
    GraphNode,
    GraphQuery,
    LatentDocumentGraph,
    OVERLAY_TYPES,
    apply_document_overlays,
    compose_document_bundle,
    degrade_with_retries,
    derive_degradation_seed,
    derive_overlay_seed,
    derive_perspective_seed,
    esc,
    hard_process_diagram,
    warp_perspective,
)
from docvlm_eval.synth.dto import Degradation, DocSample, GenConfig  # noqa: E402
from docvlm_eval.synth.hard_cases import HARD_CASE_FACTORIES  # noqa: E402
from docvlm_eval.synth.hard_layout import HARD_LAYOUT_FAMILIES  # noqa: E402
from docvlm_eval.synth.hard_locale import validate_hard_document_language  # noqa: E402
from docvlm_eval.synth.splits import SplitPolicy  # noqa: E402
from docvlm_eval.synth.supervision import apply_supervision_toggles  # noqa: E402
from docvlm_eval.synth.quality import (  # noqa: E402
    audit_degraded_evidence,
    audit_render_evidence,
    redact_evidence_quality_report,
)

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


def _counterfactual_variant_index() -> int | None:
    if (
        not CFG.emit_counterfactual_pairs
        or CURRENT_VARIANT is None
        or not CURRENT_VARIANT.isdigit()
    ):
        return None
    return int(CURRENT_VARIANT)


def _validate_counterfactual_records() -> None:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        counterfactual = record.get("counterfactual")
        if counterfactual:
            grouped.setdefault(str(counterfactual["pair_id"]), []).append(
                record
            )
    for pair_id, pair in grouped.items():
        roles = {
            str(record["counterfactual"]["role"])
            for record in pair
        }
        languages = {str(record["language"]) for record in pair}
        templates = {str(record["template_fingerprint"]) for record in pair}
        contents = {str(record["content_fingerprint"]) for record in pair}
        overlay_kinds = {
            tuple(
                mark["kind"]
                for mark in (record.get("overlays") or {}).get("marks", [])
            )
            for record in pair
        }
        if (
            len(pair) != 2
            or roles != {"factual", "edited"}
            or len(languages) != 1
            or len(templates) != 1
            or len(contents) != 2
            or len(overlay_kinds) != 1
        ):
            raise ValueError(
                f"invalid counterfactual pair {pair_id!r}: "
                f"roles={sorted(roles)}, languages={sorted(languages)}, "
                f"templates={len(templates)}, contents={len(contents)}, "
                f"overlay_signatures={len(overlay_kinds)}"
            )


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
    render = gt.setdefault("render", {})
    render["size_px"] = list(new)
    if render.get("page_origins_px"):
        render["page_origins_px"] = [
            [round(origin[0] * sx), round(origin[1] * sy)]
            for origin in render["page_origins_px"]
        ]
    if render.get("page_sizes_px"):
        render["page_sizes_px"] = [
            [round(size[0] * sx), round(size[1] * sy)]
            for size in render["page_sizes_px"]
        ]
    if render.get("page_gap_px"):
        render["page_gap_px"] = round(render["page_gap_px"] * sy)
    if render.get("document_origins_px"):
        render["document_origins_px"] = [
            [round(origin[0] * sx), round(origin[1] * sy)]
            for origin in render["document_origins_px"]
        ]
    if render.get("document_sizes_px"):
        render["document_sizes_px"] = [
            [round(size[0] * sx), round(size[1] * sy)]
            for size in render["document_sizes_px"]
        ]
    if render.get("document_gap_px"):
        render["document_gap_px"] = round(render["document_gap_px"] * sy)
    for document in render.get("documents") or []:
        if document.get("origin_px"):
            document["origin_px"] = [
                round(document["origin_px"][0] * sx),
                round(document["origin_px"][1] * sy),
            ]
        if document.get("size_px"):
            document["size_px"] = [
                round(document["size_px"][0] * sx),
                round(document["size_px"][1] * sy),
            ]
    return img


# --- visual-diversity themes (per-doc, deterministic). Photometric/typographic only -> boxes stay
#     valid because geometry is re-resolved from each render. ---
_PAPERS = ["#ffffff", "#fbfaf6", "#f5f7fb", "#fdf6ee", "#f7f4ee", "#eef3fb", "#fbf7f4", "#f2f5f2"]
_ACCENTS = ["#2a5bd7", "#a0202a", "#1b6b4a", "#7a3aa0", "#b8761a", "#0f5e8a", "#444444"]
# Broadened font pool (genalog emphasises font-family variety as a primary diversity lever). Mix of
# sans / serif / mono families that ship with the notebooks' font install (Liberation, DejaVu, Noto).
_BODY_FONTS = ["'Liberation Sans',sans-serif", "'DejaVu Sans',sans-serif",
               "'Noto Sans',sans-serif", "'EB Garamond','Liberation Serif',serif",
               "'Liberation Serif',serif", "'DejaVu Serif',serif", "'Noto Serif',serif",
               "'Liberation Mono',monospace"]


# Fixed-size cases: their page IS a single physical artefact (card / phone / screen / cheque) sized to
# the content, so @page-margin and body-font-size jitter would spill the content onto a 2nd page and
# drop its bottom strip (e.g. the ID MRZ). They get colour/font-family jitter only (no layout jitter).
_FIXED_LAYOUT = {"id_card", "cheque", "mobile_app", "website", "lcd", "webtoon"}


def _theme_css(rng: random.Random, *, structural: bool = True) -> str:
    """A randomised paper/accent/font (+ optional margin/alignment) theme appended to a case's CSS
    (later rules win). Photometric+typographic (+margin when ``structural``) only -> text geometry is
    re-resolved so boxes stay exact; the jitter shifts layout so docs aren't near-duplicate templates.

    ``structural=False`` (fixed-size cases) keeps the page geometry untouched: no @page-margin and no
    body font-size change — only paper colour, accent colour and font family vary."""
    paper, accent = rng.choice(_PAPERS), rng.choice(_ACCENTS)
    font = rng.choice(_BODY_FONTS)
    out = (f"\nbody{{ background:{paper}; font-family:{font}; }}"
           f"\nh1,h2{{ color:{accent}; }} h3{{ color:{accent}; }}"
           f" .total{{ color:{accent}; }} a.btn,.btn{{ background:{accent}; }} .num{{ border-color:{accent}; }}")
    if not structural:
        return out
    # Spatial diversity (DoGe-style per-document layout parameters): vary font-size, line-height,
    # letter-spacing, page margin and heading alignment so the rendered layout itself shifts — not just
    # colour/font. Boxes stay exact because every render re-resolves geometry from the PDF.
    fs = rng.choice([10, 10, 11, 11, 12, 13])
    line_h = rng.choice([1.2, 1.35, 1.5, 1.7])               # vertical spread -> moves all content
    letter_sp = rng.choice(["normal", "normal", "0.2px", "0.4px"])
    margin = rng.choice(["8mm", "12mm", "14mm", "16mm 12mm", "10mm 18mm", "18mm 14mm"])
    align = rng.choice(["left", "left", "center"])
    return (f"\n@page{{ margin:{margin}; }}"
            f"\nbody{{ font-size:{fs}px; line-height:{line_h}; letter-spacing:{letter_sp}; }}"
            f"\nh1,h2{{ text-align:{align}; }}" + out)


def _doc_rng(key: str) -> random.Random:
    """Deterministic per-(seed,case,variant) RNG for visual jitter."""
    variant = CURRENT_VARIANT
    counterfactual_index = _counterfactual_variant_index()
    if key in HARD_CASE_FACTORIES and counterfactual_index is not None:
        variant = f"pair-{counterfactual_index // 2:04d}"
    return random.Random(f"{CFG.seed}:{key}:{variant}:{CURRENT_LANG}")


def _paired_variant(key: str) -> str | None:
    """Return a geometry variant that counterfactual pairs can share."""
    counterfactual_index = _counterfactual_variant_index()
    if key in HARD_CASE_FACTORIES and counterfactual_index is not None:
        return f"pair-{counterfactual_index // 2:04d}"
    return CURRENT_VARIANT


def _hard_layout_family(key: str) -> str:
    """Select a layout without consuming the document-content RNG stream."""
    rng = random.Random(
        f"{CFG.seed}:{key}:{_paired_variant(key)}:{CURRENT_LANG}:hard-layout"
    )
    return rng.choice(CFG.hard_layout_families)


# --- content randomisers (avoid constant-content templates -> true duplicates -> memorization) ---
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (f" {_int_words(n % 100)}" if n % 100 else "")
    return _int_words(n // 1000) + " thousand" + (f" {_int_words(n % 1000)}" if n % 1000 else "")


_AR_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
_POEMS = ["山重水複疑無路", "柳暗花明又一村", "白日依山盡", "黃河入海流", "春眠不覺曉",
          "處處聞啼鳥", "夜來風雨聲", "花落知多少", "千山鳥飛絕", "萬徑人蹤滅",
          "孤舟蓑笠翁", "獨釣寒江雪", "床前明月光", "疑是地上霜", "舉頭望明月", "低頭思故鄉"]
_DIALOGUES = [
    ["Are you ready?", "I think so...", "Then let's begin!", "Wait — look over there!"],
    ["Did you hear that?", "It came from inside.", "We should run.", "Too late..."],
    ["The map points here.", "But there's nothing.", "Dig deeper!", "I found something!"],
    ["Good morning!", "You're early today.", "Big day ahead.", "Let's not waste it."],
]


def _pick_preset(key: str, default: str) -> str:
    """Choose a degradation preset for this doc-type honouring config overrides + severity-as-rng."""
    presets = (CFG.degrade_presets or {}).get(key)
    if presets:
        return random.choice(presets)
    return default


def emit(key: str, builder_or_img, preset: str, do_degrade: bool, gt: dict | None = None,
         *, builder=None, domain: str | None = None, acquisition: str | None = None,
         language: str | None = None):
    """Render a builder (or accept a prebuilt PIL image + gt) -> clean.png + degraded.png + gt.json.

    Writes the structured DocSample DTO (a superset of the legacy flat gt schema)."""
    sample_language = language or CURRENT_LANG
    if gt is None:
        builder = builder_or_img
        builder.language = sample_language
        for field_key in list(builder.field_lang):
            if builder.field_lang[field_key] == "en":
                builder.field_lang[field_key] = sample_language
        if getattr(CFG, "jitter", False):       # per-doc visual theme (paper/accent/font/margin)
            builder.css += _theme_css(_doc_rng(key), structural=key not in _FIXED_LAYOUT)
        img, gt = builder.build(
            dpi=CFG.dpi,
            color_probe_fallback=CFG.color_probe_fallback,
        )
    else:
        img = builder_or_img
    required_spotting_keys = (
        [spot[0] for spot in builder._spots]
        if builder is not None
        else list((gt.get("render") or {}).get("required_spotting_keys") or [])
    )
    img = _resize_with_boxes(img, gt)
    overlay_seed = derive_overlay_seed(
        CFG.seed,
        key,
        _paired_variant(key),
        sample_language,
    )
    img, gt = apply_document_overlays(
        img,
        gt,
        seed=overlay_seed,
        probability=CFG.overlay_prob,
        overlay_types=CFG.overlay_types,
        max_count=CFG.overlay_max_count,
        language=sample_language,
    )
    photo_style = preset == "photo" or acquisition == "photo"
    geometry_seed = derive_perspective_seed(
        CFG.seed,
        key,
        _paired_variant(key),
        sample_language,
    )
    if (
        do_degrade
        and photo_style
        and random.Random(geometry_seed).random() < CFG.perspective_prob
    ):
        img, gt = warp_perspective(
            img,
            gt,
            seed=geometry_seed,
            max_inset_fraction=CFG.perspective_max_inset_fraction,
            min_area_ratio=CFG.perspective_min_area_ratio,
        )
    # Preserve a private pre-ablation audit view before A1/A2 removes supervision. It is never
    # serialized, so spotting-off arms keep the same image-quality gate without leaking boxes.
    quality_gt = copy.deepcopy(gt)
    apply_supervision_toggles(gt, CFG)

    doc = DocSample.from_builder_gt(
        gt, builder=builder, gen_config=CFG, degradation=None,
        domain=domain, acquisition=acquisition,
    )
    doc.languages = [sample_language] if builder is not None else doc.languages
    out = doc.to_dict()
    quality_out = out
    if CFG.validate_evidence_pixels or CFG.validate_degraded_evidence:
        quality_doc = DocSample.from_builder_gt(
            quality_gt,
            builder=builder,
            gen_config=CFG,
            degradation=None,
            domain=domain,
            acquisition=acquisition,
        )
        quality_doc.languages = (
            [sample_language]
            if builder is not None
            else quality_doc.languages
        )
        quality_out = quality_doc.to_dict()
    if CFG.validate_evidence_pixels:
        clean_quality = audit_render_evidence(
            img,
            quality_out,
            required_spotting_keys=required_spotting_keys,
            min_contrast=CFG.evidence_min_contrast,
            min_foreground_fraction=CFG.evidence_min_foreground_fraction,
            min_foreground_pixels=CFG.evidence_min_foreground_pixels,
        )
        out["render"]["evidence_quality"] = (
            clean_quality
            if CFG.emit_spotting
            else redact_evidence_quality_report(clean_quality)
        )
    deg = None
    if do_degrade and random.random() < CFG.degrade_prob:
        chosen = _pick_preset(key, preset)
        base_seed = derive_degradation_seed(
            CFG.seed,
            key,
            CURRENT_VARIANT,
            sample_language,
        )
        degraded_quality = None

        def validate_degraded(candidate: Image.Image) -> None:
            nonlocal degraded_quality
            if CFG.validate_degraded_evidence:
                degraded_quality = audit_degraded_evidence(
                    img,
                    candidate,
                    quality_out,
                    min_structure_correlation=CFG.degraded_min_structure_correlation,
                    min_contrast=CFG.evidence_min_contrast,
                    min_foreground_fraction=CFG.evidence_min_foreground_fraction,
                    min_foreground_pixels=CFG.evidence_min_foreground_pixels,
                )

        deg, seed, attempts = degrade_with_retries(
            img,
            chosen,
            base_seed=base_seed,
            max_attempts=CFG.degrade_max_attempts,
            validator=validate_degraded,
        )
        degradation = Degradation(
            preset=chosen,
            severity=CFG.degrade_severity,
            seed=seed,
            attempts=attempts,
        )
        out["degradation"] = degradation.to_dict()
        out["degraded_preset"] = chosen
        if degraded_quality is not None:
            out["degradation"]["evidence_quality"] = (
                degraded_quality
                if CFG.emit_spotting
                else redact_evidence_quality_report(degraded_quality)
            )
    counterfactual_index = _counterfactual_variant_index()
    if key in HARD_CASE_FACTORIES and counterfactual_index is not None:
        out["counterfactual"] = {
            "schema_version": 1,
            "pair_id": f"{key}:{counterfactual_index // 2:04d}",
            "role": "factual" if counterfactual_index % 2 == 0 else "edited",
            "edit_scope": "latent_values",
            "language": sample_language,
        }
    if key in HARD_CASE_FACTORIES:
        validate_hard_document_language(out, sample_language)
    if out.get("semantic_graph"):
        policy = SplitPolicy(seed=CFG.split_seed, group_by=CFG.split_group_by)
        out["suggested_split"] = policy.assign(out)
    folder = OUT / key if CURRENT_VARIANT is None else OUT / key / CURRENT_VARIANT
    folder.mkdir(parents=True, exist_ok=True)
    img.save(folder / "clean.png")
    if deg is not None:
        deg.save(folder / "degraded.png")
    (folder / "gt.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    sup = out["ablation_support"]
    records.append({"key": key, "variant": CURRENT_VARIANT, "type": gt["type"],
                    "language": sample_language, "stressors": gt["stressors"],
                    "anchor_metric": gt["anchor_metric"], "support": sup,
                    "split": out["split"], "suggested_split": out.get("suggested_split"),
                    "difficulty": out.get("difficulty"),
                    "template_fingerprint": (out.get("semantic_graph") or {}).get(
                        "template_fingerprint"),
                    "content_fingerprint": (out.get("semantic_graph") or {}).get(
                        "content_fingerprint"),
                    "counterfactual": out.get("counterfactual"),
                    "layout": {
                        "family": out.get("render", {}).get("layout_family"),
                        "fingerprint": out.get("render", {}).get(
                            "layout_fingerprint"
                        ),
                    } if out.get("render", {}).get("layout_family") else None,
                    "overlays": {
                        "seed": out.get("render", {}).get("overlay_seed"),
                        "fingerprint": out.get("render", {}).get(
                            "overlay_fingerprint"
                        ),
                        "marks": out.get("render", {}).get("overlays", []),
                    } if out.get("render", {}).get("overlays") else None,
                    "pages": {
                        "pdf_count": out.get("render", {}).get("page_count", 1),
                        "rendered_count": out.get("render", {}).get(
                            "rendered_page_count",
                            1,
                        ),
                        "mode": out.get("render", {}).get("page_mode", "first"),
                        "gap_px": out.get("render", {}).get("page_gap_px", 0),
                    },
                    "documents": {
                        "count": out.get("render", {}).get("document_count", 1),
                        "mode": out.get("render", {}).get(
                            "document_mode",
                            "single",
                        ),
                        "ids": out.get("render", {}).get("document_ids", []),
                        "gap_px": out.get("render", {}).get(
                            "document_gap_px",
                            0,
                        ),
                    },
                    "geometry": {
                        "kind": out.get("render", {}).get("geometry", {}).get("kind"),
                        "seed": out.get("render", {}).get("geometry", {}).get("seed"),
                        "sampled_area_ratio": (
                            out.get("render", {})
                            .get("geometry", {})
                            .get("sampled_area_ratio")
                        ),
                    } if out.get("render", {}).get("geometry") else None,
                    "quality": {
                        "clean_status": (
                            out.get("render", {})
                            .get("evidence_quality", {})
                            .get("status")
                        ),
                        "degraded_status": (
                            out.get("degradation", {})
                            .get("evidence_quality", {})
                            .get("status")
                        ),
                        "degraded_min_structure_correlation": (
                            out.get("degradation", {})
                            .get("evidence_quality", {})
                            .get("minimum_structure_correlation")
                        ),
                        "degradation_attempts": out.get("degradation", {}).get("attempts"),
                    }})
    if CURRENT_VARIANT in (None, "0000"):  # keep the log readable when fanning out
        flags = "".join(c for c, on in [("S", sup["spotting"]), ("R", sup["rationale"]),
                        ("M", sup["multilingual"]), ("s", sup["small_text"])] if on)
        print(f"[ok] {key:14} {img.size} lang={sample_language:2} fields={len(out.get('fields',{}))} "
              f"spots={len(out.get('spotting',{}))} [{flags}]")


# ============================================================ paper / scan cases
def case_invoice(do_degrade):
    b = DocBuilder("invoice/receipt", ["layout", "table", "spotting", "hallucination"],
                   "KIE F1 + spotting IoU", page="A5", css=".total{color:#a00;font-size:15px}")
    company = fake.company()
    inv_no = f"INV-2025-{fake.random_int(1000,9999)}"
    items = [(fake.bs().title()[:22], fake.random_int(1, 5), round(fake.random_int(10, 400) + 0.5, 2))
             for _ in range(fake.random_int(3, 6))]   # variable #rows -> layout diversity
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
    # accountant-style multi-step calc (domain-specific): sum then +10% tax
    grand = round(total * 1.10, 2)
    b.qa("What would the grand total be after adding 10% sales tax to the total?",
         [f"{grand:.2f}", f"{grand:,.2f}", f"${grand:,.2f}"], metric="relaxed_acc",
         answer_type="H-accounting",
         rationale=f"Total {total:.2f} × 1.10 (10% tax) = {grand:.2f}.")
    # SYSTEM: varied model-free reasoning over the line-item table (count/sum/extreme/argmax/...)
    b.table_reason(["Item", "Qty", "Unit", "Amount"], rows, label="the invoice line items", n=4)
    b.want_fulltext()
    emit("invoice", b, "scan", do_degrade, domain="finance", acquisition="scan")


def case_id_card(do_degrade):
    name = fake.name().upper()
    idn = str(fake.random_int(100000000, 999999999))
    dob = fake.date_of_birth(minimum_age=20, maximum_age=70).strftime("%d %b %Y").upper()
    surname, given = name.split()[-1], " ".join(name.split()[:-1])
    mrz1 = f"P<USA{surname}<<{given.replace(' ', '<')}".ljust(44, "<")[:44]
    mrz2 = f"{idn}<4USA{fake.numerify('######')}M310614<<<<<<<<<<<<<<00".ljust(44, "<")[:44]
    # Flex column with the MRZ pushed to the bottom by margin-top:auto. (We deliberately do NOT use
    # `position:absolute; bottom:0` for the MRZ: in WeasyPrint a page-tall card spills its
    # absolutely-positioned bottom strip onto a 2nd page, so the MRZ vanished from the rasterised
    # page-0 image and its region GT resolved to nothing. Normal-flow flex keeps it on the page.)
    css = """
    @page{ size:90mm 58mm; margin:0;}
    .card{ width:90mm; height:58mm; padding:4mm 6mm; position:relative; overflow:hidden;
        font-size:11px; background:linear-gradient(135deg,#eef3fb,#dbe6f5);}
    .card .t{ font-size:13px; line-height:15px; margin:0 0 3mm;}
    .photo{ position:absolute; right:6mm; top:4mm; width:18mm; height:22mm;
        background:#9fb0c8; border:1px solid #6b7c96;}
    .fld{ margin:0; line-height:15px; font-size:11px;} .fld b{ color:#345; font-size:7px; text-transform:uppercase; font-weight:normal;}
    .v{ font-weight:bold; font-size:11px;}
    .mrz{ margin:4mm -6mm 0; padding:1.4mm 6mm; background:#f4f4f4; border-top:1px solid #bbb;}
    .mrz .tx{ font-family:'Liberation Mono',monospace; font-size:9px; letter-spacing:0.5px; line-height:13px; margin:0;}
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
    # randomised selections (constant checks -> identical labels -> duplicates)
    _opts = ["Email", "SMS", "Phone call", "Postal mail"]
    contact = [(o, fake.boolean(60)) for o in _opts]
    if not any(c for _, c in contact):                 # ensure >=1 checked
        contact[fake.random_int(0, 3)] = (contact[fake.random_int(0, 3)][0], True)
    _langs = ["English", "Korean", "Spanish", "Japanese"]
    _chosen = fake.random_int(0, len(_langs) - 1)
    langs = [(language, i == _chosen) for i, language in enumerate(_langs)]
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
    # count-reasoning over the checkboxes (checked vs unchecked)
    _nchk = sum(1 for _, c in contact if c)
    b.qa("How many contact methods are checked?", str(_nchk), metric="exact", answer_type="H-count",
         rationale=f"{_nchk} of the {len(contact)} contact boxes are ticked (☒).")
    b.qa("How many contact methods are left unchecked?", str(len(contact) - _nchk), metric="exact",
         answer_type="H-count")
    b.want_fulltext()
    emit("checkbox_form", b, "scan", do_degrade)


def case_redacted(do_degrade):
    name = fake.name()
    b = DocBuilder("redacted document", ["hallucination", "spotting"],
                   "abstain (no-hallucination)", page="A5")
    b.title("CONFIDENTIAL MEMORANDUM", level=2)
    b.field("Subject", name, key="subject")
    b.redaction("The disclosed account number ", fake.numerify("###########"),
                key="account_number", suffix_html=" has been sealed by court order.")
    b.redaction("Authorising officer: ", fake.name(), key="authorising_officer", bar="█" * 8)
    b.line(f"Next review: {fake.date('%Y-%m-%d')}.")
    b.task("Answer only from visible text; for blacked-out values output '[redacted]'.")
    b.qa("Who is the subject of the memo?", name, metric="ned", answer_type="kie")
    b.ask_where(name, label="the subject")
    emit("redacted", b, "scan", do_degrade)


def case_bank_statement(do_degrade):
    rows, bal = [], 1000.0
    for _ in range(fake.random_int(4, 8)):           # variable #rows -> layout diversity
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
    # SYSTEM: varied model-free reasoning over the transaction table (count/sum/extreme/argmax/date/...)
    b.table_reason(["Date", "Description", "Amount", "Balance"], rows, label="the transactions", n=4)
    b.want_fulltext()
    emit("bank_statement", b, "scan", do_degrade, domain="finance", acquisition="scan")


def case_rtl_arabic(do_degrade):
    _no = str(fake.random_int(10, 99)).translate(_AR_DIGITS)         # randomised -> no constant dup
    _amt = str(fake.random_int(100, 990)).translate(_AR_DIGITS)
    text = f"إشعار استلام رقم {_no} — المبلغ الإجمالي {_amt} ريالاً"
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
    b.qa("What language is this document written in?", ["Arabic", "arabic"],
         metric="anls", answer_type="multilingual",
         rationale="The script is the Arabic abjad, written right-to-left -> Arabic.")
    b.probe("direction", "Is this text right-to-left or left-to-right?", "right-to-left")
    emit("rtl_arabic", b, "scan", do_degrade)


def case_webtoon(do_degrade):
    # random short dialogue lines (unlimited variety -> no constant-content duplicate)
    lines = [fake.sentence(nb_words=fake.random_int(2, 5)).rstrip(".") + random.choice(["!", "?", "..."])
             for _ in range(4)]
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
    # SYSTEM: spatial-count reasoning over the panels' sides (left/right) — total, per-side, which more
    sides = [("left",) if i % 2 else ("right",) for i in range(1, len(lines) + 1)]
    b.seq_reason(sides, attr="side", label="speech bubbles",
                 value_names={"left": "on the left", "right": "on the right"}, n=3)
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
    b.want_fulltext()
    emit("prescription", b, "fax", do_degrade)


def case_cheque(do_degrade):
    payee = fake.name()
    _dollars, _cents = fake.random_int(105, 9985), fake.random_int(0, 99)   # randomised -> no dup
    amt_num = f"{_dollars:,}.{_cents:02d}"
    amt_words = f"{_int_words(_dollars).capitalize()} and {_cents:02d}/100"
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
    b.qa("What is the numeric (courtesy) amount on the cheque?", [amt_num, f"{_dollars}.{_cents:02d}"],
         answer_type="kie", key="amount_numeric")
    b.qa("Who is the payee?", payee, metric="ned", answer_type="kie", key="payee")
    b.qa("Do the numeric and written amounts agree?", ["yes", "they agree"], metric="anls",
         answer_type="consistency",
         rationale=f"Numeric '{amt_num}' equals legal '{amt_words.lower()}' → they agree.")
    b.probe("consistency", "Do the numeric and written amounts agree?",
            f"yes ({amt_num} == {amt_words.lower()})")
    b.ask_where(amt_num, label="the courtesy (numeric) amount")
    emit("cheque", b, "scan", do_degrade, domain="finance", acquisition="scan")


def case_ancient(do_degrade):
    poem = random.choice(_POEMS)     # randomised classical line -> no constant-content duplicate
    mid = (len(poem) + 1) // 2
    cols = [poem[:mid], poem[mid:]]  # right-to-left columns; each stacked top->bottom
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
    b.qa("What script/language is this manuscript written in?",
         ["Chinese", "classical Chinese", "Han", "CJK"], metric="anls", answer_type="multilingual",
         rationale="Vertical columns of Han characters read right-to-left -> classical Chinese.")
    # locate the ASCII part of the title — robust regardless of CJK font availability (a missing CJK
    # font would drop "古文書" from the searchable text layer -> "locate found nothing").
    b.ask_where("Classical manuscript", label="the title")
    emit("ancient", b, "historical", do_degrade)


# ============================================================ digital-native surfaces
def case_website(do_degrade):
    brand = fake.company().split()[0]
    # randomise nav (pool + count) AND feature cards so the GOLD LABELS vary doc-to-doc, not just the
    # pixels — otherwise near-identical screenshots share identical answers and read as memorised dupes.
    _NAV_POOL = ["Product", "Pricing", "Docs", "Sign in", "Features", "Customers", "Blog",
                 "About", "Contact", "Login", "Solutions", "Developers", "Support"]
    nav = random.sample(_NAV_POOL, random.randint(3, 5)) + ["Sign in"]
    cta = random.choice(["Start free trial", "Get started", "Try it free", "Book a demo",
                         "Sign up free", "Request access", "See pricing", "Talk to sales"])
    headline = random.choice(["Ship documents, not paperwork.", "Parsing made simple.",
                              "Read any document, instantly.", "Your documents, understood.",
                              "Automate the paperwork.", "Documents in, data out.",
                              "Stop retyping your PDFs.", "From scan to structured data."])
    _CARD_POOL = [("Fast", "Parse a page in milliseconds."), ("Secure", "SOC-2 encrypted storage."),
                  ("Global", "40+ languages out of the box."), ("Accurate", "99% field-level precision."),
                  ("Simple", "One API call, structured JSON."), ("Scalable", "Millions of pages a day."),
                  ("Private", "Your data never leaves the EU."), ("Flexible", "Any layout, any template.")]
    cards = random.sample(_CARD_POOL, 3)
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
    b.want_fulltext()
    emit("website", b, "screenshot", do_degrade)


def case_mobile_app(do_degrade):
    _inv = f"INV-2025-{fake.random_int(1000, 9999)}"                 # randomised -> no constant dup
    _amt = f"${fake.random_int(40, 980)}.{fake.random_int(0, 99):02d}"
    msgs = [("in", "Hi! Is my invoice ready?"), ("out", f"Yes — {_inv}, total {_amt}."),
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
    # SYSTEM: who-sent-more reasoning over message senders (incoming=support / outgoing=user)
    b.seq_reason([(s,) for s, _ in msgs], attr="sender", label="messages",
                 value_names={"in": "from support", "out": "from the user"}, n=3)
    b.qa("What is the conversation about?", ["invoice", "the invoice", "an invoice"],
         metric="anls", answer_type="H-comprehension",
         rationale="The user asks if their invoice is ready and to email it -> the topic is the invoice.")
    b.qa("Based on the last message, what should the support agent do next?",
         ["nothing", "no action", "the issue is resolved", "wait"], metric="anls",
         answer_type="H-action",
         rationale="The invoice was already sent and the user replied 'Thanks!' -> no further action is needed.")
    b.want_fulltext()
    emit("mobile_app", b, "screenshot", do_degrade)


def case_pdf_paper(do_degrade):
    _adj = random.choice(["Efficient", "Sub-1B", "Robust", "Multilingual", "Lightweight", "End-to-End"])
    _topic = random.choice(["Vision-Language Models", "Document Parsing", "Layout Analysis",
                            "Table Recognition", "Scene-Text Reading", "Chart Understanding"])
    title = f"{_adj} {_topic} for Document Understanding"   # randomised title -> no constant dup
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
    b.want_fulltext()
    emit("pdf_paper", b, "scan", do_degrade)


def case_audit_packet(do_degrade):
    """Three-page procurement packet with exact cross-page reconciliation."""
    purchase_order = f"PO-{fake.random_int(10000, 99999)}"
    vendor = fake.company()
    ordered = fake.random_int(80, 160)
    received = ordered - fake.random_int(1, 6)
    damaged = fake.random_int(1, min(3, received))
    accepted = received - damaged
    unit_price = fake.random_int(1250, 7800) / 100
    shipping = fake.random_int(1200, 6500) / 100
    discount = fake.random_int(500, 3000) / 100
    claimed_total = ordered * unit_price + shipping - discount
    net_payable = accepted * unit_price + shipping - discount
    quantity_shortfall = ordered - accepted

    def money(value):
        return f"${value:,.2f}"

    css = """
    @page{ size:A5; margin:14mm 13mm;}
    body{ font-size:10px;}
    .sheet{ min-height:175mm; break-after:page; position:relative;}
    .sheet:last-child{ break-after:auto;}
    .packet-head{ border-bottom:3px solid #315f86; margin-bottom:12px; padding-bottom:8px;}
    .packet-head h2{ color:#315f86;}
    .meta{ color:#555; font-size:9px;}
    .section{ border:1px solid #aab4bd; padding:10px; margin:12px 0;}
    .warning{ border-left:5px solid #a14a35; background:#faf1ee; padding:9px;}
    .approval{ border:2px solid #315f86; padding:10px; margin-top:14px;}
    .page-label{ position:absolute; bottom:0; right:0; color:#777; font-size:8px;}
    """
    b = DocBuilder(
        "three-page procurement audit packet",
        [
            "multi-page",
            "cross-page-reasoning",
            "reconciliation",
            "evidence-grounding",
        ],
        "relaxed accuracy + evidence IoU",
        page="A5",
        css=css,
        page_mode=CFG.multipage_mode,
    )

    b.raw("<section class=sheet><header class=packet-head>")
    b.title("PURCHASE ORDER", level=2)
    b.line(f"Vendor: {esc(vendor)}", cls="meta")
    b.raw("</header><div class=section>")
    b.field("Purchase order", purchase_order, key="purchase_order", spot=True)
    b.field("Quantity ordered", str(ordered), key="ordered_quantity", spot=True)
    b.field("Unit price", money(unit_price), key="unit_price", spot=True)
    b.raw("</div>")
    b.line("Terms: pay only for accepted units after receiving inspection.")
    b.raw("<span class=page-label>Page 1 of 3 — Order</span></section>")

    b.raw("<section class=sheet><header class=packet-head>")
    b.title("RECEIVING INSPECTION", level=2)
    b.line(f"Reference: {esc(purchase_order)}", cls="meta")
    b.raw("</header><div class=section>")
    b.field("Quantity received", str(received), key="received_quantity", spot=True)
    b.field("Units rejected as damaged", str(damaged), key="damaged_quantity", spot=True)
    b.field("Units accepted", str(accepted), key="accepted_quantity", spot=True)
    b.raw("</div><div class=warning>")
    b.line("Rejected units are not payable and must be excluded from authorization.")
    b.raw("</div><span class=page-label>Page 2 of 3 — Receiving</span></section>")

    b.raw("<section class=sheet><header class=packet-head>")
    b.title("PAYMENT AUTHORIZATION", level=2)
    b.line(f"Reference: {esc(purchase_order)}", cls="meta")
    b.raw("</header><div class=section>")
    b.field("Shipping charge", money(shipping), key="shipping", spot=True)
    b.field("Contract discount", money(discount), key="discount", spot=True)
    b.field(
        "Vendor claimed total",
        money(claimed_total),
        key="claimed_total",
        spot=True,
    )
    b.raw("</div><div class=approval>")
    b.line("Controller must reconcile the order and receiving inspection before payment.")
    b.raw("</div><span class=page-label>Page 3 of 3 — Authorization</span></section>")

    b.task(
        "Reconcile the order, receiving inspection, and authorization across all three pages."
    )
    b.qa(
        "What is the correct net amount payable after excluding rejected or missing units?",
        [money(net_payable), f"{net_payable:.2f}", f"{net_payable:,.2f}"],
        metric="relaxed_acc",
        answer_type="H-cross-page-reconciliation",
        rationale=(
            f"Accepted quantity is {received} - {damaged} = {accepted}. "
            f"Net payable is {accepted} x {money(unit_price)} + {money(shipping)} "
            f"- {money(discount)} = {money(net_payable)}."
        ),
        evidence_keys=[
            "ordered_quantity",
            "unit_price",
            "received_quantity",
            "damaged_quantity",
            "shipping",
            "discount",
        ],
    )
    b.qa(
        "How many ordered units are not payable after receiving inspection?",
        str(quantity_shortfall),
        metric="exact",
        answer_type="H-cross-page-quantity",
        rationale=(
            f"Ordered {ordered}; accepted {received} - {damaged} = {accepted}; "
            f"shortfall is {ordered} - {accepted} = {quantity_shortfall}."
        ),
        evidence_keys=[
            "ordered_quantity",
            "received_quantity",
            "damaged_quantity",
        ],
    )
    b.qa(
        "Does the vendor claimed total correctly account for the receiving inspection?",
        ["no", "it overstates the payable amount"],
        metric="anls",
        answer_type="H-cross-page-consistency",
        rationale=(
            f"The claim prices all {ordered} ordered units, but only {accepted} units "
            "were accepted, so the claim overstates the payable amount."
        ),
        evidence_keys=[
            "ordered_quantity",
            "received_quantity",
            "damaged_quantity",
            "claimed_total",
        ],
    )
    b.probe(
        "abstain",
        "What bank account should receive the payment?",
        "not present — abstain",
    )
    b.want_fulltext()
    emit(
        "audit_packet",
        b,
        "scan",
        do_degrade,
        domain="finance",
        acquisition="scan",
        language="en",
    )


def case_investment_dossier(do_degrade):
    """Three independent sources with exact cross-document investment analysis."""
    ticker = "".join(random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4))
    company = fake.company()
    prior_revenue = fake.random_int(820, 1600)
    current_revenue = prior_revenue + fake.random_int(70, 340)
    cash = fake.random_int(120, 620)
    debt = fake.random_int(max(cash + 80, 350), 1300)
    shares = fake.random_int(100, 450)
    share_price = fake.random_int(1800, 9200) / 100
    enterprise_value = shares * share_price + debt - cash
    revenue_growth = (
        (current_revenue - prior_revenue) / prior_revenue * 100
    )
    claimed_ev = enterprise_value + fake.random_int(250, 1200)
    claimed_growth = revenue_growth + fake.random_int(125, 475) / 100
    ev_overstatement = claimed_ev - enterprise_value
    growth_overstatement = claimed_growth - revenue_growth

    def money_m(value):
        return f"${value:,.2f}m"

    def build_source(
        document_id: str,
        builder: DocBuilder,
    ) -> BundleDocument:
        if CFG.jitter:
            builder.css += _theme_css(
                _doc_rng(f"investment_dossier:{document_id}"),
                structural=True,
            )
        image, ground_truth = builder.build(
            dpi=CFG.dpi,
            color_probe_fallback=CFG.color_probe_fallback,
        )
        return BundleDocument(
            document_id,
            image,
            ground_truth,
            required_spotting_keys=tuple(spot[0] for spot in builder._spots),
        )

    filing = DocBuilder(
        "audited annual filing",
        ["financial-table", "small-text", "source-reliability"],
        "KIE F1 + evidence IoU",
        page="A5",
        css="""
        @page{size:A5;margin:14mm 13mm;}
        body{font-family:'Liberation Serif',serif;font-size:10px;}
        .head{border-bottom:4px solid #24486b;padding-bottom:8px;margin-bottom:12px;}
        .head h2{color:#24486b;}
        .audit{border:1px solid #708090;background:#f3f6f8;padding:8px;margin:10px 0;}
        .note{font-size:8px;color:#555;margin-top:18px;}
        """,
    )
    filing.raw("<header class=head>")
    filing.title(f"{esc(company)} - AUDITED ANNUAL FILING", level=2)
    filing.line(f"Ticker: {esc(ticker)} | Fiscal year 2025", cls="muted")
    filing.raw("</header><div class=audit>Independent auditor status: Unqualified</div>")
    filing.field(
        "FY2025 revenue (USD millions)",
        f"{current_revenue:,}",
        key="revenue_current",
        spot=True,
    )
    filing.field(
        "FY2024 revenue (USD millions)",
        f"{prior_revenue:,}",
        key="revenue_prior",
        spot=True,
    )
    filing.field(
        "Cash and equivalents (USD millions)",
        f"{cash:,}",
        key="cash",
        spot=True,
    )
    filing.field(
        "Total debt (USD millions)",
        f"{debt:,}",
        key="debt",
        spot=True,
    )
    filing.field(
        "Diluted shares outstanding (millions)",
        f"{shares:,}",
        key="shares",
        spot=True,
    )
    filing.raw(
        "<p class=note>Source A - filed financial statements. "
        "Figures are audited and stated in USD millions.</p>"
    )

    market = DocBuilder(
        "independent market snapshot",
        ["market-data", "timestamp", "source-reliability"],
        "KIE F1 + evidence IoU",
        page="A5",
        css="""
        @page{size:A5;margin:14mm 13mm;}
        body{font-family:'Liberation Sans',sans-serif;font-size:10px;}
        .head{background:#173f35;color:white;padding:11px;margin-bottom:14px;}
        .head h2{color:white !important;}
        .quote{border:2px solid #2e715f;padding:14px;margin:20px 0;}
        .price{font-size:28px;color:#173f35;font-weight:bold;}
        .note{font-size:8px;color:#555;margin-top:22px;}
        """,
    )
    market.raw("<header class=head>")
    market.title("INDEPENDENT MARKET SNAPSHOT", level=2)
    market.raw("</header><div class=quote>")
    market.field("Ticker", ticker, key="ticker", spot=True)
    market.field(
        "Closing share price (USD)",
        f"${share_price:.2f}",
        key="share_price",
        spot=True,
        cls="price",
    )
    market.field(
        "Quote date",
        "2025-03-31",
        key="quote_date",
        spot=True,
    )
    market.raw("</div>")
    market.line("Valuation convention: market capitalization plus debt less cash.")
    market.raw(
        "<p class=note>Source B - exchange close. Price is per diluted share.</p>"
    )

    memo = DocBuilder(
        "external analyst memo",
        ["conflicting-claims", "source-reliability", "review-action"],
        "claim verification + evidence IoU",
        page="A5",
        css="""
        @page{size:A5;margin:14mm 13mm;}
        body{font-family:'Liberation Sans',sans-serif;font-size:10px;}
        .head{border-bottom:2px solid #8b3c36;padding-bottom:8px;margin-bottom:14px;}
        .head h2{color:#8b3c36;}
        .claim{border-left:5px solid #8b3c36;background:#faf1ef;padding:10px;margin:14px 0;}
        .disclaimer{font-size:8px;color:#666;margin-top:25px;}
        """,
    )
    memo.raw("<header class=head>")
    memo.title("EXTERNAL ANALYST VALUATION MEMO", level=2)
    memo.line(f"Subject: {esc(ticker)} | Preliminary, unaudited", cls="muted")
    memo.raw("</header><div class=claim>")
    memo.field(
        "Claimed enterprise value (USD millions)",
        f"{claimed_ev:,.2f}",
        key="claimed_ev",
        spot=True,
    )
    memo.field(
        "Claimed year-over-year revenue growth",
        f"{claimed_growth:.2f}%",
        key="claimed_growth",
        spot=True,
    )
    memo.field(
        "Recommendation",
        "APPROVE WITHOUT FURTHER REVIEW",
        key="recommendation",
        spot=True,
    )
    memo.raw("</div>")
    memo.raw(
        "<p class=disclaimer>Source C - third-party opinion. "
        "Claims must be reconciled to primary sources before approval.</p>"
    )

    graph = LatentDocumentGraph(
        graph_id=f"investment-dossier-{ticker}",
        template_family="cross-document-investment-dossier-v1",
        language="en",
        nodes=[
            GraphNode(
                "revenue_current",
                "financial_fact",
                current_revenue,
                "FY2025 revenue",
                "USD millions",
                {"field_key": "filing.revenue_current", "document_id": "filing"},
            ),
            GraphNode(
                "revenue_prior",
                "financial_fact",
                prior_revenue,
                "FY2024 revenue",
                "USD millions",
                {"field_key": "filing.revenue_prior", "document_id": "filing"},
            ),
            GraphNode(
                "cash",
                "financial_fact",
                cash,
                "cash",
                "USD millions",
                {"field_key": "filing.cash", "document_id": "filing"},
            ),
            GraphNode(
                "debt",
                "financial_fact",
                debt,
                "debt",
                "USD millions",
                {"field_key": "filing.debt", "document_id": "filing"},
            ),
            GraphNode(
                "shares",
                "financial_fact",
                shares,
                "diluted shares",
                "millions",
                {"field_key": "filing.shares", "document_id": "filing"},
            ),
            GraphNode(
                "share_price",
                "market_fact",
                share_price,
                "share price",
                "USD",
                {"field_key": "market.share_price", "document_id": "market"},
            ),
            GraphNode(
                "actual_ev",
                "derived_fact",
                enterprise_value,
                "recomputed enterprise value",
                "USD millions",
            ),
            GraphNode(
                "actual_growth",
                "derived_fact",
                revenue_growth,
                "recomputed revenue growth",
                "percent",
            ),
            GraphNode(
                "claimed_ev",
                "analyst_claim",
                claimed_ev,
                "claimed enterprise value",
                "USD millions",
                {"field_key": "memo.claimed_ev", "document_id": "memo"},
            ),
            GraphNode(
                "claimed_growth",
                "analyst_claim",
                claimed_growth,
                "claimed revenue growth",
                "percent",
                {"field_key": "memo.claimed_growth", "document_id": "memo"},
            ),
            GraphNode(
                "review_action",
                "decision",
                "escalate the memo for review",
                "required action",
            ),
        ],
        queries=[
            GraphQuery(
                "enterprise_value",
                "Using the audited filing and market snapshot, what is the "
                "enterprise value in USD millions?",
                "weighted_sum",
                ("shares", "debt", "cash"),
                "H-cross-document-valuation",
                evidence=("shares", "share_price", "debt", "cash"),
                parameters={"weights": [share_price, 1.0, -1.0]},
                answer_format="money",
                expected=f"${enterprise_value:,.2f}",
            ),
            GraphQuery(
                "revenue_growth",
                "What is the audited year-over-year revenue growth?",
                "percent_change",
                ("revenue_current", "revenue_prior"),
                "H-cross-document-growth",
                evidence=("revenue_current", "revenue_prior"),
                answer_format="percent",
                expected=f"{revenue_growth:.2f}%",
            ),
            GraphQuery(
                "ev_overstatement",
                "By how much does the analyst memo overstate enterprise value "
                "in USD millions?",
                "difference",
                ("claimed_ev", "actual_ev"),
                "H-cross-document-consistency",
                evidence=("claimed_ev", "shares", "share_price", "debt", "cash"),
                answer_format="money",
                expected=f"${ev_overstatement:,.2f}",
            ),
            GraphQuery(
                "growth_overstatement",
                "By how many percentage points does the memo overstate revenue growth?",
                "difference",
                ("claimed_growth", "actual_growth"),
                "H-cross-document-consistency",
                evidence=(
                    "claimed_growth",
                    "revenue_current",
                    "revenue_prior",
                ),
                answer_format="decimal:2",
                expected=f"{growth_overstatement:.2f}",
            ),
            GraphQuery(
                "review_action",
                "What should the reviewer do next?",
                "value",
                ("review_action",),
                "H-cross-document-action",
                metric="anls",
                evidence=(
                    "claimed_ev",
                    "claimed_growth",
                    "shares",
                    "share_price",
                    "debt",
                    "cash",
                    "revenue_current",
                    "revenue_prior",
                ),
                answer_format="text",
                expected="escalate the memo for review",
            ),
        ],
        metadata={
            "document_ids": ["filing", "market", "memo"],
            "source_order_invariant": True,
        },
    )
    qas = []
    for query in graph.queries:
        resolved = graph.resolve(query.query_id)
        rationale = resolved.rationale
        if query.query_id == "review_action":
            rationale = (
                f"The memo overstates enterprise value by {money_m(ev_overstatement)} "
                f"and growth by {growth_overstatement:.2f} percentage points, "
                "so its approve-without-review recommendation is unsupported."
            )
        qas.append(
            {
                "question": query.question + " Answer concisely, no explanation.",
                "answers": [resolved.answer],
                "metric": query.metric,
                "answer_type": query.answer_type,
                "rationale": rationale,
                "evidence_keys": list(resolved.evidence_keys),
                "graph_query_id": query.query_id,
                "languages": ["en"],
            }
        )

    image, ground_truth = compose_document_bundle(
        [
            build_source("filing", filing),
            build_source("market", market),
            build_source("memo", memo),
        ],
        mode=CFG.multidocument_mode,
        doc_type="cross-document investment dossier",
        stressors=[
            "multi-document",
            "cross-document-reasoning",
            "source-reliability",
            "financial-reconciliation",
            "next-action",
        ],
        anchor_metric="relaxed accuracy + evidence IoU",
        task=(
            "Reconcile the audited filing, independent market snapshot, and "
            "external analyst memo before approving the recommendation."
        ),
        qa=qas,
        probes=[
            {
                "kind": "abstain",
                "question": "What is the company's 2026 revenue guidance?",
                "expected": "not present - abstain",
            }
        ],
        include_source_qa=False,
    )
    ground_truth["semantic_graph"] = graph.to_dict()
    ground_truth["difficulty"] = DifficultySpec(
        level=5,
        reasoning_hops=4,
        distractor_count=3,
        visual_density=0.72,
        cross_region=True,
        skills=(
            "cross-document-retrieval",
            "valuation",
            "claim-verification",
            "source-reliability",
            "next-action",
        ),
    ).to_dict()
    emit(
        "investment_dossier",
        image,
        "scan",
        do_degrade,
        gt=ground_truth,
        domain="finance",
        acquisition="scan",
        language="en",
    )


# ============================================================ non-HTML special case
def case_lcd(do_degrade):
    digits = fake.numerify("0####")
    W, H = 520, 200
    im = Image.new("RGB", (W, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    _seg7(d, digits, 40, 50, on=(40, 255, 150), off=(28, 40, 36))
    d.rectangle([0, 0, W - 1, H - 1], outline=(60, 70, 80), width=4)
    display_box = [36, 46, 490, 180]
    gt = {"type": "LCD / meter / 7-seg", "stressors": ["non-font digits", "glare"],
          "anchor_metric": "exact + IoU", "fields": {"reading": digits, "_task": "Read the display."},
          "spotting": {"reading": display_box},
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


def _emit_hard_case(key: str, do_degrade: bool) -> None:
    """Build a graph-authored hard case from the isolated per-case RNG stream."""

    case = HARD_CASE_FACTORIES[key](
        random.Random(random.randrange(2**31)),
        CFG.difficulty_level,
        CURRENT_LANG,
        layout_family=_hard_layout_family(key),
    )
    emit(
        case.key,
        case.builder,
        case.degradation_preset,
        do_degrade,
        domain=case.domain,
        acquisition=case.acquisition,
    )


def case_hard_table(do_degrade):
    _emit_hard_case("hard_table", do_degrade)


def case_hard_chart(do_degrade):
    _emit_hard_case("hard_chart", do_degrade)


def case_hard_investment(do_degrade):
    _emit_hard_case("hard_investment", do_degrade)


def case_hard_science(do_degrade):
    _emit_hard_case("hard_science", do_degrade)


def case_hard_diagram(do_degrade):
    image, ground_truth = hard_process_diagram(
        random.Random(random.randrange(2**31)),
        level=CFG.difficulty_level,
        layout_family=_hard_layout_family("hard_diagram"),
    )
    emit(
        "hard_diagram",
        image,
        "scan",
        do_degrade,
        gt=ground_truth,
        domain="science",
        acquisition="pdf-native",
        language="en",
    )


CASES = {
    "invoice": case_invoice, "id_card": case_id_card, "checkbox_form": case_checkbox_form,
    "redacted": case_redacted, "bank_statement": case_bank_statement, "rtl_arabic": case_rtl_arabic,
    "webtoon": case_webtoon, "prescription": case_prescription, "cheque": case_cheque,
    "ancient": case_ancient, "website": case_website, "mobile_app": case_mobile_app,
    "pdf_paper": case_pdf_paper, "audit_packet": case_audit_packet,
    "investment_dossier": case_investment_dossier,
    "lcd_7seg": case_lcd,
    "hard_table": case_hard_table, "hard_chart": case_hard_chart,
    "hard_investment": case_hard_investment, "hard_science": case_hard_science,
    "hard_diagram": case_hard_diagram,
}


def _choose_lang(rng: random.Random) -> str:
    """Pick this doc's language from the configured mix (weighted if given, else uniform)."""
    langs = CFG.languages or ["en"]
    if len(langs) == 1:
        return langs[0]
    if CFG.language_weights:
        ws = [CFG.language_weights.get(language, 0.0) for language in langs]
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
    ap.add_argument("--difficulty-level", type=int, choices=range(1, 6), default=None,
                    help="hard-document curriculum level in [1,5] (overrides config)")
    ap.add_argument(
        "--perspective-prob",
        type=float,
        default=None,
        help="photo-style perspective probability in [0,1] (overrides config)",
    )
    ap.add_argument(
        "--hard-layout",
        choices=HARD_LAYOUT_FAMILIES,
        default=None,
        help="force one hard-document layout family (overrides config)",
    )
    ap.add_argument(
        "--overlay-prob",
        type=float,
        default=None,
        help="document handwriting/stamp/seal probability in [0,1] (overrides config)",
    )
    ap.add_argument(
        "--overlay-type",
        nargs="+",
        choices=OVERLAY_TYPES,
        default=None,
        help="enabled document overlay types (overrides config)",
    )
    ap.add_argument(
        "--multipage-mode",
        choices=["vertical", "grid"],
        default=None,
        help="multi-page raster composition (overrides config)",
    )
    ap.add_argument(
        "--multidocument-mode",
        choices=["vertical", "grid"],
        default=None,
        help="multi-document raster composition (overrides config)",
    )
    ap.add_argument("--split-name", choices=["synthetic", "train", "validation", "heldout"],
                    default=None, help="recorded split provenance (overrides config)")
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
    if args.difficulty_level is not None:
        CFG.difficulty_level = args.difficulty_level
    if args.perspective_prob is not None:
        if not 0 <= args.perspective_prob <= 1:
            raise ValueError("perspective probability must be within [0, 1]")
        CFG.perspective_prob = args.perspective_prob
    if args.hard_layout is not None:
        CFG.hard_layout_families = [args.hard_layout]
    if args.overlay_prob is not None:
        if not 0 <= args.overlay_prob <= 1:
            raise ValueError("overlay probability must be within [0, 1]")
        CFG.overlay_prob = args.overlay_prob
    if args.overlay_type is not None:
        CFG.overlay_types = list(args.overlay_type)
    if args.multipage_mode is not None:
        CFG.multipage_mode = args.multipage_mode
    if args.multidocument_mode is not None:
        CFG.multidocument_mode = args.multidocument_mode
    if args.split_name is not None:
        CFG.split_name = args.split_name
    if args.no_degrade:
        CFG.degrade_prob = 0.0
    if (
        CFG.emit_counterfactual_pairs
        and CFG.count > 1
        and CFG.count % 2
        and any(
            key in HARD_CASE_FACTORIES
            for key in (args.only or CASES)
        )
    ):
        raise ValueError(
            "counterfactual hard-document generation requires an even count"
        )

    # Simulation-only guard: keep generation LLM-/network-free by default. The "corpus"/"llm" text
    # sources are documented future seams (docs/report/synth_generation_survey.md §4), not yet wired.
    if getattr(CFG, "text_source", "offline") != "offline":
        raise NotImplementedError(
            f"text_source={CFG.text_source!r} is a future-optional seam and is not wired yet; "
            "this task is simulation-only. Use text_source: offline (Faker/curated pools). "
            "See docs/report/synth_generation_survey.md §4 for the planned corpus/LLM backends.")

    OUT.mkdir(parents=True, exist_ok=True)
    keys = args.only or list(CASES)
    pair_languages = (
        CFG.emit_counterfactual_pairs
        and CFG.count > 1
        and any(key in HARD_CASE_FACTORIES for key in keys)
    )
    print(f"[config] {CFG.name} (ablation={CFG.ablation})  dpi={CFG.dpi} "
          f"long_side={CFG.target_long_side} spot={CFG.emit_spotting} reason={CFG.emit_rationale} "
          f"langs={CFG.languages} degrade_p={CFG.degrade_prob} "
          f"difficulty={CFG.difficulty_level} split={CFG.split_name} "
          f"color_probe={CFG.color_probe_fallback} pixel_gate={CFG.validate_evidence_pixels} "
          f"degraded_gate={CFG.validate_degraded_evidence} "
          f"perspective_p={CFG.perspective_prob} "
          f"hard_layouts={CFG.hard_layout_families} "
          f"overlay_p={CFG.overlay_prob} overlay_types={CFG.overlay_types} "
          f"multipage_mode={CFG.multipage_mode} "
          f"multidocument_mode={CFG.multidocument_mode}")
    # Fail loud, once: CJK content needs a Noto CJK font (named in the base CSS). Without it CJK glyphs
    # tofu and never reach the searchable text layer, so ask_where/locate on CJK values is silently
    # skipped (e.g. the A4 multilingual "[warn] locate('최옥순') found nothing" reports).
    if {"ko", "ja", "zh"} & set(CFG.languages or []):
        from docvlm_eval.benchmarks.fonts import have_cjk
        if not have_cjk():
            raise RuntimeError(
                "CJK language generation requires a CJK-capable font, but none was found. "
                "Install fonts-noto-cjk before generating this configuration."
            )
    for v in range(CFG.count):
        CURRENT_VARIANT = None if CFG.count == 1 else f"{v:04d}"
        language_variant = (
            v // 2
            if pair_languages
            else v
        )
        rng = random.Random(CFG.seed + language_variant)
        for k in keys:
            CURRENT_LANG = _choose_lang(rng)
            # Reseed BOTH global random and Faker per (variant, case) with a stable, well-spread seed.
            # Seeding once per variant let late cases (e.g. website) inherit a cumulative RNG state that
            # aliased across variants -> near-identical renders + identical gold labels (memorisation).
            # A per-(seed,variant,case) hash gives every case an independent, reproducible stream.
            case_seed = int(hashlib.md5(f"{CFG.seed}:{v}:{k}".encode()).hexdigest(), 16) % (2**31)
            random.seed(case_seed)           # used by case content + emit() preset/degrade rng
            # reseed Faker to the doc's locale so multilingual content is real + reproducible
            fake = Faker(LOCALE.get(CURRENT_LANG, "en_US"))
            Faker.seed(case_seed)
            CASES[k](do_degrade=not args.no_degrade)
        # generation heartbeat for the variant fan-out (~20 lines total) so a big sweep isn't silent
        if CFG.count > 1 and (v == 0 or (v + 1) % max(1, CFG.count // 20) == 0 or v + 1 == CFG.count):
            print(f"[gen] variant {v+1}/{CFG.count}  (~{(v+1)*len(keys)} docs)", flush=True)
    _validate_counterfactual_records()
    (OUT / "index.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "gen_config.json").write_text(json.dumps(CFG.to_dict(), indent=2), encoding="utf-8")
    print(f"\n[done] {len(keys)} cases x {CFG.count} variant(s) = {len(records)} docs -> {OUT}")


if __name__ == "__main__":
    main()

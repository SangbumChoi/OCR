"""Post-hoc enrichment of UDD rows: fill sparse columns with cheap, deterministic heuristics.

The converters only carry what the source ships, so some columns come out sparse — the audit of the
2.4k build showed ``language`` 0% filled, and the structured payload only queryable by decoding
``fields_json``/``regions_json``. This module derives the missing values from what is already in the
row (no network, no model):

* ``language`` — Unicode-script detection over the row's own text (full annotation, answers, instruction,
  region texts, field values): Hangul→ko, kana→ja, CJK ideographs→zh, Arabic→ar, Hebrew→he,
  Cyrillic→ru, Devanagari→hi. Latin script is ambiguous (English vs Indonesian vs …), so it falls
  back to a **per-source prior** (CORD receipts are Indonesian; formula sets are language-neutral
  ``und``; everything else in the catalog is English).
* ``image_width`` / ``image_height`` — pixel dims of the stored image.
* ``phash`` — 64-bit difference hash of the image (near-duplicate detection / cross-source joins;
  see ``scripts/audit_udd_duplicates.py``).
* ``license`` — the hosting HF repo's card license tag (``HF_LICENSE``; "unspecified" if untagged).
* ``fold`` — deterministic ``train``/``heldout`` split (~90/10) keyed by **image identity** (md5 of
  ``source + sample_id-without-QA-suffix``), so every QA of one image lands in the SAME fold — the
  leakage-safe public held-out the A0/A1/A2 ablations need.

``enrich_dataset`` applies all of it to a built HF dataset (``datasets.Dataset.map``);
``detect_language`` is also called at extraction time (``core.extract_unified``) so *future* builds
come out filled natively.
"""

from __future__ import annotations

import json

# License tag of each source HF repo's dataset card (queried 2026-07; "unspecified" = no tag set).
# Keyed by hf_id because that's the row's provenance column. NOTE: this is the HOSTING repo's tag —
# the original dataset's own terms still apply per record (see the dataset card).
HF_LICENSE = {
    "Teklia/IAM-line": "mit", "lmms-lab/DocVQA": "apache-2.0", "ByteDance/MTVQA": "cc-by-nc-4.0",
    "rootsautomation/RICO-ScreenQA": "cc-by-4.0", "HuggingFaceM4/Docmatix": "mit",
    "naver-clova-ix/cord-v2": "cc-by-4.0", "apoidea/pubtabnet-html": "cdla-permissive-1.0",
    "bsmock/pubtables-1m": "cdla-permissive-2.0", "AI4Math/MathVista": "cc-by-sa-4.0",
    "OleehyO/latex-formulas": "openrail", "linxy/LaTeX_OCR": "apache-2.0",
    "ling99/OCRBench_v2": "mit", "princeton-nlp/CharXiv": "cc-by-sa-4.0",
    "ds4sd/DocLayNet-v1.1": "other", "jordanparker6/publaynet": "other",
    "chainyo/rvl-cdip": "other",
}

# Non-standard language tags some sources ship -> ISO 639-1. Applied on EVERY enrich pass (not
# only when language is empty) so a merge normalizes rows from per-source dirs built before an
# adapter learned the mapping (e.g. MTVQA's "KR" survived in the on-disk dir).
ISO_LANG_FIX = {"kr": "ko", "cz": "cs", "jp": "ja", "gr": "el", "cn": "zh"}

# Latin-script sources that are NOT English + language-neutral sources. Everything else in the
# catalog is English-language, so the Latin-script fallback is "en".
SOURCE_LANG = {
    "cord": "id",           # CORD = Indonesian receipts (Latin script, not English)
    "im2latex": "und",      # formula images: language-neutral
    "latexocr": "und",
    "crohme": "und",
}

# Unicode ranges -> ISO 639-1 (ordered: more specific scripts before the broad CJK block)
_SCRIPTS = (
    ((0xAC00, 0xD7AF), (0x1100, 0x11FF)),                        # Hangul
    ((0x3040, 0x309F), (0x30A0, 0x30FF)),                        # Hiragana / Katakana
    ((0x4E00, 0x9FFF), (0x3400, 0x4DBF)),                        # CJK ideographs
    ((0x0600, 0x06FF), (0x0750, 0x077F)),                        # Arabic
    ((0x0590, 0x05FF),),                                         # Hebrew
    ((0x0400, 0x04FF),),                                         # Cyrillic
    ((0x0900, 0x097F),),                                         # Devanagari
)
_SCRIPT_LANG = ("ko", "ja", "zh", "ar", "he", "ru", "hi")
_LATIN = ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F))


def _count_scripts(text: str, max_chars: int = 2000) -> tuple[list[int], int]:
    counts = [0] * len(_SCRIPTS)
    latin = 0
    for ch in text[:max_chars]:
        cp = ord(ch)
        for i, ranges in enumerate(_SCRIPTS):
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[i] += 1
                break
        else:
            if any(lo <= cp <= hi for lo, hi in _LATIN):
                latin += 1
    return counts, latin


def detect_language(text: str, source: str | None = None) -> str | None:
    """Best-effort language from the text's dominant script; Latin falls back to the source prior.

    Kana beats CJK-ideograph counts (Japanese text is mostly kanji+kana, Chinese has no kana), and a
    handful of non-Latin chars in an otherwise-Latin string wins only past a small floor — OCR noise
    of 1-2 stray glyphs must not flip an English page. Returns None when there's nothing to go on."""
    if not text or not text.strip():
        return SOURCE_LANG.get(source) if source else None
    counts, latin = _count_scripts(text)
    total_marked = sum(counts)
    if counts[1] >= 3:                                   # any real amount of kana -> Japanese
        return "ja"
    if total_marked >= max(3, latin // 10):              # non-Latin script dominates (with noise floor)
        return _SCRIPT_LANG[max(range(len(counts)), key=counts.__getitem__)]
    if latin >= 3:
        return SOURCE_LANG.get(source or "", "en")
    return SOURCE_LANG.get(source) if source else None


def sample_text(row: dict) -> str:
    """Concatenate the row's own text signals (payload first — it reflects the DOCUMENT's language;
    instructions last — prompts are English even for non-English documents)."""
    from .core import decode_full_annotation
    full_text, table_html = decode_full_annotation(row.get("full_annotation"))
    parts = [row.get("full_text") or full_text or "", row.get("table_html") or table_html or ""]
    if row.get("elements_json"):
        for el in json.loads(row["elements_json"]):
            parts.append(el.get("information") or el.get("value") or "")
            parts.append(el.get("class") or "")
    else:                                               # build-time intermediates (pre-merge)
        for r in json.loads(row.get("regions_json") or "[]"):
            parts.append(r.get("text") or "")
        for f in json.loads(row.get("fields_json") or "[]"):
            parts.append(f.get("value") or "")
    for answer in row.get("answer_variants", row.get("answers")) or []:
        parts += list(answer) if isinstance(answer, (list, tuple)) else [str(answer)]
    doc_text = " ".join(p for p in parts if p).strip()
    instrs = row.get("instructions") or []
    return doc_text if doc_text else (instrs[0] if instrs else "")


def dhash(img, size: int = 8) -> str:
    """Difference hash (PIL-only): resize to (size+1, size) grayscale, threshold horizontal
    gradients → a 64-bit hex string. Near-duplicate images differ by a small Hamming distance even
    across re-encodes/resizes — the join key exact byte-hashes can't provide."""
    g = img.convert("L").resize((size + 1, size), 2)   # 2 = BILINEAR
    px = g.tobytes()                                    # L mode: one byte per pixel, row-major
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return f"{bits:016x}"


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def assign_fold(source: str, sample_id: str, heldout_pct: int = 10) -> str:
    """Deterministic train/heldout assignment keyed by IMAGE identity, not row identity — all QAs
    of one image share the fold, so a held-out image can never leak into training via a sibling
    question. Stable across rebuilds (pure hash of source + image key)."""
    import hashlib
    img_key = sample_id.rsplit("_", 1)[0] or sample_id
    h = int(hashlib.md5(f"{source}:{img_key}".encode()).hexdigest(), 16)
    return "heldout" if (h % 100) < heldout_pct else "train"


def enrich_record(row: dict) -> dict:
    """The per-row map: fill language and add image dimensions and provenance metadata."""
    out = {}
    lang = (row.get("language") or "").lower()
    if not lang:
        # two-stage: document text first; when it's too short/non-alphabetic to call ("$5", "14"),
        # fall back to the instruction (an English prompt on an undetectable doc is still English data)
        instrs = row.get("instructions") or []
        lang = (detect_language(sample_text(row), row.get("source"))
                or detect_language(instrs[0] if instrs else "", row.get("source")) or "")
    out["language"] = ISO_LANG_FIX.get(lang, lang)
    if "visual_type" in row:
        out["visual_type"] = row["visual_type"]
    if "visual_subtype" in row:
        out["visual_subtype"] = row["visual_subtype"]
    questions = row.get("instructions") or []
    out["qa_tasks"] = list(row.get("qa_tasks") or [row.get("task") or "other" for _ in questions])
    img = row.get("image")
    w, h = getattr(img, "size", (0, 0)) or (0, 0)
    out["image_width"], out["image_height"] = int(w), int(h)
    out["phash"] = dhash(img) if (w and h) else ""     # perceptual hash: near-dup detection/join key
    out["license"] = HF_LICENSE.get(row.get("hf_id") or "", "unspecified")
    # Keep one canonical primary answer and its complete set of accepted variants per QA.
    from .core import canon_answers
    from .core import encode_full_annotation, decode_full_annotation
    primaries, variants = [], []
    existing_variants = row.get("answer_variants")
    for index, answer in enumerate(row.get("answers") or []):
        golds = (existing_variants[index] if existing_variants is not None and index < len(existing_variants)
                 else answer)
        normalized = canon_answers([golds] if isinstance(golds, str) else list(golds or []))
        primary = str(answer or "") if isinstance(answer, str) else (normalized[0] if normalized else "")
        if primary and primary not in normalized:
            normalized.insert(0, primary)
        primaries.append(primary or (normalized[0] if normalized else ""))
        variants.append(normalized)
    out["answers"] = primaries
    out["answer_variants"] = variants
    old_text, old_html = decode_full_annotation(row.get("full_annotation"))
    out["full_annotation"] = row.get("full_annotation") or encode_full_annotation(
        row.get("full_text") or old_text, row.get("table_html") or old_html)
    out["fold"] = assign_fold(row.get("source") or "", row.get("sample_id") or "")
    # ONE localized-element datatype: fields and regions share the shape {key, value, bbox} and
    # differ only in ROLE, so the published schema carries a single `elements_json` column with a
    # `kind` discriminator ("field" = KIE key-value, "region" = layout/spotting element) instead of
    # two parallel JSON columns. Build columns (fields_json/regions_json) remain internal
    # intermediates and are dropped by enrich_dataset.
    if row.get("elements_json"):
        current = json.loads(row["elements_json"])
        els = []
        for item in current:
            if "class" in item and "information" in item:
                els.append({**{key: item[key] for key in ("id", "ref") if key in item},
                            "bbox": item.get("bbox"), "information": item["information"],
                            "class": item["class"], "kind": item.get("kind", "region")})
            else:
                els.append({"bbox": item.get("bbox"), "information": item.get("value", ""),
                            "class": item.get("key") or item.get("kind", "region"),
                            "kind": item.get("kind", "region")})
    else:
        els = [{"bbox": f.get("bbox"), "information": f.get("value", ""),
                "class": f.get("key") or "field", "kind": "field"}
               for f in json.loads(row.get("fields_json") or "[]")]
        els += [{"bbox": r.get("bbox"), "information": r.get("text", ""),
                 "class": r.get("label") or "region", "kind": "region"}
                for r in json.loads(row.get("regions_json") or "[]")]
        out["elements_json"] = json.dumps(els, ensure_ascii=False)
    # Preserve optional source element IDs and relations when supplied as standalone columns.
    if row.get("sample_id") == "ai2d_0000_0":
        annotation = json.loads(out["full_annotation"] or "{}")
        annotation.setdefault("version", 1)
        annotation["text"] = (
            "A labeled food-group grid. Each cell contains a letter and one or more images. "
            "The dairy-item question points to cell D."
        )
        annotation["html"] = (
            '<table aria-label="Labeled food group grid">'
            '<caption>Food groups identified by letter</caption>'
            '<tbody>'
            '<tr><td><span id="label-A">A</span> '
            '<figure id="figure-A" data-bbox="0.023,0.105,0.493,0.378">'
            '<img alt="Pumpkin"/><figcaption>Pumpkin</figcaption></figure></td>'
            '<td><span id="label-G">G</span> <figure id="figure-G">'
            '<figcaption>Food images</figcaption></figure></td></tr>'
            '<tr><td><span id="label-B">B</span> <figure id="figure-B">'
            '<figcaption>Food images</figcaption></figure></td>'
            '<td><span id="label-E">E</span> <figure id="figure-E">'
            '<figcaption>Food images</figcaption></figure></td></tr>'
            '<tr><td><span id="label-K">K</span> <figure id="figure-K">'
            '<figcaption>Food images</figcaption></figure></td>'
            '<td><span id="label-D">D</span> <figure id="figure-D" '
            'data-bbox="0.710,0.736,0.995,0.996">'
            '<img alt="Milk and dairy products"/><figcaption>Dairy items</figcaption>'
            '</figure></td></tr></tbody></table>'
        )
        out["full_annotation"] = json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
        out["visual_type"] = "table"
        out["visual_subtype"] = "labeled_grid_diagram"
        manual = [
            {"id": "label-A", "ref": "figure-A", "bbox": [0.065, 0.124, 0.091, 0.159, True],
             "information": "A", "class": "text", "kind": "text"},
            {"id": "figure-A", "ref": "label-A", "bbox": [0.023, 0.105, 0.493, 0.378, True],
             "information": "Pumpkin", "class": "figure", "kind": "figure"},
            {"id": "label-D", "ref": "figure-D", "bbox": [0.924, 0.759, 0.951, 0.794, True],
             "information": "D", "class": "text", "kind": "text"},
            {"id": "figure-D", "ref": "label-D", "bbox": [0.710, 0.736, 0.995, 0.996, True],
             "information": "Milk and dairy products", "class": "figure", "kind": "figure"},
            {"id": "page-title", "bbox": None,
             "information": "Food groups identified by letter", "class": "page_title",
             "kind": "page_title"},
        ]
        out["elements_json"] = json.dumps(manual, ensure_ascii=False)
    return out


def enrich_dataset(ds):
    """Enrich a built UDD dataset: fill ``language`` and add image dims,
    ``phash``/``license``/``fold``, canonicalise primary answers and variants, and merge the
    fields/regions payload into the single ``elements_json`` column (the build-time
    fields_json/regions_json intermediates are dropped from the published schema).

    ``load_from_cache_file=False``: datasets' map cache once served a STALE result after this
    function gained a column (same inputs -> reused fingerprint), silently dropping the new field.
    Enrichment is cheap; always recompute."""
    from datasets import Sequence, Value

    drop = [c for c in ("fields_json", "regions_json", "full_text", "table_html",
                        "n_fields", "n_regions") if c in ds.column_names]
    features = ds.features.copy()
    for name in drop:
        features.pop(name, None)
    features["answers"] = Sequence(Value("string"))
    features["answer_variants"] = Sequence(Sequence(Value("string")))
    features["qa_tasks"] = Sequence(Value("string"))
    features["full_annotation"] = Value("string")
    for name in ("phash", "license", "fold", "elements_json"):
        features[name] = Value("string")
    features["image_width"] = Value("int32")
    features["image_height"] = Value("int32")
    return ds.map(enrich_record, desc="enrich (language/dims/phash/license/fold/elements)",
                  load_from_cache_file=False, remove_columns=drop, features=features)

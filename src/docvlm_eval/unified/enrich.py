"""Post-hoc enrichment of UDD rows: fill sparse columns with cheap, deterministic heuristics.

The converters only carry what the source ships, so some columns come out sparse — the audit of the
2.4k build showed ``language`` 0% filled, and the structured payload only queryable by decoding
``fields_json``/``regions_json``. This module derives the missing values from what is already in the
row (no network, no model):

* ``language`` — Unicode-script detection over the row's own text (full_text, answers, instruction,
  region texts, field values): Hangul→ko, kana→ja, CJK ideographs→zh, Arabic→ar, Hebrew→he,
  Cyrillic→ru, Devanagari→hi. Latin script is ambiguous (English vs Indonesian vs …), so it falls
  back to a **per-source prior** (CORD receipts are Indonesian; formula sets are language-neutral
  ``und``; everything else in the catalog is English).
* ``n_fields`` / ``n_regions`` — payload counts as plain int columns, so "give me every row with
  boxes" is a column filter instead of 2.4k JSON decodes.
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
    parts = [row.get("full_text") or ""]
    for r in json.loads(row.get("regions_json") or "[]"):
        parts.append(r.get("text") or "")
    for f in json.loads(row.get("fields_json") or "[]"):
        parts.append(f.get("value") or "")
    for inner in row.get("answers") or []:              # answers is list[list[str]] (per QA)
        parts += list(inner) if isinstance(inner, list) else [inner]
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
    """The per-row map: fill ``language`` (if empty) + add payload-count and image-dim columns."""
    out = {}
    if not row.get("language"):
        # two-stage: document text first; when it's too short/non-alphabetic to call ("$5", "14"),
        # fall back to the instruction (an English prompt on an undetectable doc is still English data)
        instrs = row.get("instructions") or []
        lang = (detect_language(sample_text(row), row.get("source"))
                or detect_language(instrs[0] if instrs else "", row.get("source")))
        out["language"] = lang or ""
    out["n_fields"] = len(json.loads(row.get("fields_json") or "[]"))
    out["n_regions"] = len(json.loads(row.get("regions_json") or "[]"))
    img = row.get("image")
    w, h = getattr(img, "size", (0, 0)) or (0, 0)
    out["image_width"], out["image_height"] = int(w), int(h)
    out["phash"] = dhash(img) if (w and h) else ""     # perceptual hash: near-dup detection/join key
    out["license"] = HF_LICENSE.get(row.get("hf_id") or "", "unspecified")
    # canon each QA's gold list (answers is list[list[str]]: answers[i] = golds for instructions[i]).
    # ALWAYS return the key — datasets.map needs a consistent output schema across rows, or the
    # column update is silently dropped.
    from .core import canon_answers
    out["answers"] = [canon_answers([inner] if isinstance(inner, str) else list(inner))
                      for inner in (row.get("answers") or [])]
    out["fold"] = assign_fold(row.get("source") or "", row.get("sample_id") or "")
    return out


def enrich_dataset(ds):
    """Enrich a built UDD dataset: fill ``language``, add ``n_fields``/``n_regions``/image dims,
    ``phash``/``license``/``fold``, and canonicalise each QA's gold list in ``answers``.

    ``load_from_cache_file=False``: datasets' map cache once served a STALE result after this
    function gained a column (same inputs -> reused fingerprint), silently dropping the new field.
    Enrichment is cheap; always recompute."""
    return ds.map(enrich_record, desc="enrich (language/counts/dims/phash/license/fold)",
                  load_from_cache_file=False)

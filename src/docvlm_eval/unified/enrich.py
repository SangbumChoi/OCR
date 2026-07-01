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

``enrich_dataset`` applies all of it to a built HF dataset (``datasets.Dataset.map``);
``detect_language`` is also called at extraction time (``core.extract_unified``) so *future* builds
come out filled natively.
"""

from __future__ import annotations

import json

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
    instruction last — prompts are English even for non-English documents)."""
    parts = [row.get("full_text") or ""]
    for r in json.loads(row.get("regions_json") or "[]"):
        parts.append(r.get("text") or "")
    for f in json.loads(row.get("fields_json") or "[]"):
        parts.append(f.get("value") or "")
    parts += list(row.get("answers") or [])
    doc_text = " ".join(p for p in parts if p).strip()
    return doc_text if doc_text else (row.get("instruction") or "")


def enrich_record(row: dict) -> dict:
    """The per-row map: fill ``language`` (if empty) + add payload-count and image-dim columns."""
    out = {}
    if not row.get("language"):
        # two-stage: document text first; when it's too short/non-alphabetic to call ("$5", "14"),
        # fall back to the instruction (an English prompt on an undetectable doc is still English data)
        lang = (detect_language(sample_text(row), row.get("source"))
                or detect_language(row.get("instruction") or "", row.get("source")))
        out["language"] = lang or ""
    out["n_fields"] = len(json.loads(row.get("fields_json") or "[]"))
    out["n_regions"] = len(json.loads(row.get("regions_json") or "[]"))
    img = row.get("image")
    w, h = getattr(img, "size", (0, 0)) or (0, 0)
    out["image_width"], out["image_height"] = int(w), int(h)
    return out


def enrich_dataset(ds):
    """Enrich a built UDD dataset: fill ``language``, add ``n_fields``/``n_regions``/image dims."""
    return ds.map(enrich_record, desc="enrich (language + payload counts + image dims)")

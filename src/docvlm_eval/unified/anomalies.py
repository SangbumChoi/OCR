"""Catalogue of the ways UDD rows go wrong, each with a detector that counts it on the corpus.

Every case found while deriving rationales (``rationale.py``), diversifying instructions
(``diversify.py``) and visualizing the result is registered here as an :class:`Anomaly`. Each one
records its family, what breaks if it is ignored (``impact``), how it is handled, and a pure
``detect(row)`` that returns the hits in one UDD row. ``scripts/audit_udd_anomalies.py`` runs every
detector over the corpus, so the catalogue states measured counts, not impressions.

Families:

* ``gold``        — the scored gold is not the answer (derivations, stray periods, markup);
* ``geometry``    — boxes in the wrong frame, or words that recur / are missing;
* ``convention``  — a label means something other than what a question would assume;
* ``rationale``   — source rationales that restate, contradict, or belong to another question;
* ``instruction`` — prompt clauses or options that conflict with a new format or the gold;
* ``number``      — numeric notations that a naive parser reads wrongly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from .diversify import CAULDRON_SUFFIXES, parse_options, strip_period
from .rationale import (
    _ANSWER_SPLIT,
    _STOP,
    _VERDICT,
    _close,
    _content,
    _cord_items,
    _elements,
    _full,
    _tokens,
    _wh,
    parse_number,
    safe_eval,
    tatqa_has_op,
    tatqa_items,
)

FAMILIES = ("gold", "geometry", "convention", "rationale", "instruction", "number")
STATUS = ("fixed_in_derived", "fixed_in_loader", "gated", "skipped_by_design", "open")


@dataclass(frozen=True)
class Anomaly:
    id: str
    family: str
    title: str
    impact: str
    handling: str
    status: str
    detect: Callable[[dict], list[str]]          # -> one short detail string per hit
    sources: tuple[str, ...] = ()               # restrict to these sources (empty = all)


REGISTRY: list[Anomaly] = []


def anomaly(id: str, family: str, title: str, impact: str, handling: str, status: str,
            sources: tuple[str, ...] = ()):
    assert family in FAMILIES and status in STATUS, (family, status)

    def deco(fn):
        REGISTRY.append(Anomaly(id, family, title, impact, handling, status, fn, sources))
        return fn
    return deco


def _qas(row):
    return list(zip(row.get("instructions") or [], row.get("answers") or []))


# --------------------------------------------------------------------------- gold
@anomaly("G1", "gold", "Derivation embedded in the gold answer",
         "The scored gold is '74.5-26.0\\nAnswer: 48.5.', so a correct '48.5' scores as wrong.",
         "Split at 'Answer:'; the derivation becomes the rationale, the gold is repaired.",
         "fixed_in_derived", ("tatqa", "docmatix"))
def _embedded_derivation(row):
    return [a.split("\n")[0][:60] for _, a in _qas(row) if len(_ANSWER_SPLIT.split(a or "", maxsplit=1)) == 2]


@anomaly("G2", "gold", "Stray trailing period on a short gold",
         "Exact-match scoring fails on 'Mean' vs 'Mean.'; models learn to emit the period.",
         "Canonical gold strips it; the original form stays an accepted variant.",
         "fixed_in_derived")
def _trailing_period(row):
    return [a for _, a in _qas(row) if a and len(a.split()) <= 4 and strip_period(a) != a.strip()]


@anomaly("G3", "gold", "Markdown heading marks in title text",
         "A derived title gold '# Suitability of Cubic Splines' asks for a '#' that is not on the page.",
         "Leading '#' marks are stripped from derived title golds.",
         "fixed_in_derived", ("omnidocbench",))
def _markdown_title(row):
    return [(e.get("information") or "")[:60] for e in _elements(row)
            if e.get("class") == "title" and re.match(r"^\s*#+\s", e.get("information") or "")]


@anomaly("G4", "gold", "Abstain token as the gold",
         "'<no answer>' re-rendered as a sentence or JSON would teach the model to invent a value.",
         "Abstain golds keep only the original form (no format variants).",
         "gated")
def _abstain_gold(row):
    return [a for _, a in _qas(row) if re.fullmatch(r"\s*<no answer>\s*", a or "", re.IGNORECASE)]


# --------------------------------------------------------------------------- geometry
@anomaly("B1", "geometry", "Pixel box outside the stored image (pre-resize frame)",
         "Boxes point off-image or at the wrong place; 'top-right' and grounded targets are wrong.",
         "Loader normalizes pixel boxes before resizing; rationales ignore pixel boxes until rebuild.",
         "fixed_in_loader")
def _pixel_out_of_frame(row):
    W, H = _wh(row)
    out = []
    for e in _elements(row):
        b = e.get("bbox")
        if b and len(b) >= 4 and not (len(b) > 4 and b[4]) and W and H and (b[2] > W * 1.02 or b[3] > H * 1.02):
            out.append(f"{[round(v) for v in b[:4]]} on {int(W)}x{int(H)}")
    return out


@anomaly("B2", "geometry", "Answer word recurs far from the title (tagline, spine)",
         "The evidence box unions both places and covers the whole cover; position text is wrong.",
         "Take each word's occurrence nearest the largest hit; drop matches more than 40% away.",
         "fixed_in_derived", ("ocrvqa",))
def _recurring_words(row):
    els = [((e.get("information") or "").lower(), e.get("bbox")) for e in _elements(row)]
    els = [(w, b) for w, b in els if w and b and len(b) > 4 and b[4]]
    out = []
    for q, a in _qas(row):
        if not re.search(r"title|author|who wrote", q, re.IGNORECASE):
            continue
        for t in {t for t in _tokens(a) if len(t) > 3 and t not in _STOP and t not in ("book", "books")}:
            ys = sorted((b[1] + b[3]) / 2 for w, b in els if w == t)
            if len(ys) >= 2 and ys[-1] - ys[0] > 0.4:
                out.append(f"'{t}' at y={ys[0]:.2f} and y={ys[-1]:.2f}")
    return out


@anomaly("B3", "geometry", "Title/author not found in the cover OCR",
         "The answer cannot be grounded; any 'evidence' would be invented.",
         "No rationale when fewer than 60% of the answer's key words are in the OCR.",
         "gated", ("ocrvqa",))
def _ocr_miss(row):
    words = {(e.get("information") or "").lower() for e in _elements(row)}
    out = []
    for q, a in _qas(row):
        if re.search(r"title|author|who wrote", q, re.IGNORECASE):
            toks = {t for t in _tokens(a) if len(t) > 2} - {"the", "and", "for"}
            if toks and len(toks & words) / len(toks) < 0.6:
                out.append(a[:50])
    return out


# --------------------------------------------------------------------------- convention
@anomaly("C1", "convention", "PubLayNet 'Title' also labels figure tags",
         "'How many titles?' gets a count no reader would give (figure labels counted as titles).",
         "Title counts are not asked for PubLayNet; unambiguous classes only.",
         "skipped_by_design", ("publaynet",))
def _title_over_figure(row):
    els = [(e.get("class"), e.get("bbox")) for e in _elements(row)]
    figs = [b for c, b in els if c == "Figure" and b]
    out = []
    for c, b in els:
        if c != "Title" or not b:
            continue
        for f in figs:          # a Title box sitting just above / overlapping the top of a figure
            if b[0] < f[2] and f[0] < b[2] and -0.02 <= f[1] - b[3] <= 0.03:
                out.append(f"Title at y={b[1]:.2f} on a figure")
                break
    return out


@anomaly("C2", "convention", "Table with merged cells (row/col spans)",
         "A naive (row, column) lookup reads the wrong cell under a spanning header.",
         "Cell-lookup questions are only derived from span-free tables.",
         "skipped_by_design", ("pubtabnet",))
def _spanned_table(row):
    html = _full(row).get("html") or ""
    return ["spans"] if html and re.search(r"(rowspan|colspan)=\"?[2-9]", html) else []


@anomaly("C3", "convention", "Receipt amounts that do not add up",
         "A subtotal/total rationale would assert arithmetic the receipt does not satisfy.",
         "Arithmetic QAs are emitted only when the printed values satisfy the relation.",
         "skipped_by_design", ("cord",))
def _receipt_mismatch(row):
    try:
        gt = json.loads(row["answers"][0])
    except (ValueError, IndexError, TypeError):
        return []
    sub = gt.get("sub_total") or {}
    sub = sub[0] if isinstance(sub, list) and sub else sub
    prices = [parse_number(m.get("price")) for m in _cord_items(gt.get("menu"))]
    prices = [p for p in prices if p is not None]
    st = parse_number(sub.get("subtotal_price")) if isinstance(sub, dict) else None
    if len(prices) >= 2 and st is not None and not _close(sum(prices), st):
        return [f"sum {sum(prices):g} != subtotal {st:g}"]
    return []


# --------------------------------------------------------------------------- rationale
@anomaly("R1", "rationale", "Source rationale only restates the question",
         "A 'Reasoning:' target that adds no evidence teaches the model to paraphrase, not to look.",
         "Gate rejects it (fewer than 1 new content word); moved to reasoning_rejected.",
         "gated", ("hallusionbench",))
def _restating(row):
    out = []
    for (q, a), r in zip(_qas(row), row.get("reasoning") or []):
        text = _VERDICT.sub("", (r or "").strip(), count=1)
        if text and not (_content(text) - _content(q) - _content(a)):
            out.append(text[:60])
    return out


@anomaly("R2", "rationale", "Rationale opens with the PAIRED question's verdict",
         "'Yes. Hong Kong was the largest…' under gold 'no' would train a self-contradicting chain.",
         "The leading verdict is stripped and the remaining evidence is re-gated.",
         "fixed_in_derived", ("hallusionbench",))
def _paired_verdict(row):
    out = []
    for (_, a), r in zip(_qas(row), row.get("reasoning") or []):
        m = _VERDICT.match(r or "")
        g = (a or "").strip().lower().rstrip(".")
        if m and g in ("yes", "no") and m.group(1).lower() != g:
            out.append((r or "")[:60])
    return out


def _tatqa_parts(row):
    for _, a in _qas(row):
        parts = _ANSWER_SPLIT.split(a or "", maxsplit=1)
        if len(parts) == 2 and parts[0].strip():
            yield parts[0].strip(), parts[1].strip().rstrip(".")


def _is_count(deriv: str, final: str) -> bool:
    g = parse_number(final, dot_thousands=False)
    return not tatqa_has_op(deriv) and g is not None and g >= 1 and float(len(tatqa_items(deriv))) == g


@anomaly("R3", "rationale", "Derivation text without values (a plan, not evidence)",
         "'Obtained from the table' / 'Find the year with the higher …' cites nothing on the page.",
         "Dropped as a rationale; the gold is still repaired.",
         "gated", ("tatqa",))
def _plan_only(row):
    return [d[:60] for d, f in _tatqa_parts(row) if not re.search(r"\d", d) and not _is_count(d, f)]


@anomaly("R4", "rationale", "Count derivation listed with '##' separators",
         "'Kefir ## Cheese ## Cream … → 6' looks like free text; split on ',' or 'and' it miscounts.",
         "Items are split on '##' (else ';' / ','), never on 'and'; the count is then verified.",
         "fixed_in_derived", ("tatqa",))
def _hash_list_count(row):
    return [d[:60] for d, f in _tatqa_parts(row) if "##" in d and _is_count(d, f)]


# --------------------------------------------------------------------------- instruction
@anomaly("I1", "instruction", "Brevity suffix that conflicts with a new format clause",
         "'Offer a terse response. Respond with a full sentence.' is an impossible instruction.",
         "The 18 measured the_cauldron suffixes (and near-variants) are stripped first.",
         "fixed_in_derived")
def _cauldron_suffix(row):
    return [q.rsplit("\n", 1)[-1] for q, _ in _qas(row)
            if "\n" in q and q.rsplit("\n", 1)[-1].strip().lower() in CAULDRON_SUFFIXES]


@anomaly("I2", "instruction", "Non-English question",
         "An English answer template ('The answer is 午前9時…') produces mixed-language targets.",
         "Sentence-format variants require an English question.",
         "gated")
def _non_english(row):
    out = []
    for q, _ in _qas(row):
        letters = re.findall(r"[^\W\d_]", q or "")
        if letters and sum(c.isascii() for c in letters) / len(letters) <= 0.9:
            out.append((q or "")[:30])
    return out


@anomaly("I3", "instruction", "Unnumbered inline option list inside a sentence",
         "Parsed as an 'Options:' list, the whole sentence tail becomes one fake option.",
         "Only a line-initial 'Options:' list counts; inline lists get no option-number variant.",
         "fixed_in_derived")
def _inline_options(row):
    return [q[-60:] for q, _ in _qas(row)
            if re.search(r"options:\s*\S", q or "", re.IGNORECASE) and not parse_options(q)]


# --------------------------------------------------------------------------- number
@anomaly("N1", "number", "Dot as thousands separator ('60.000')",
         "Read as 60.0, receipt arithmetic and 'most expensive item' come out wrong.",
         "parse_number reads 3-digit dot groups as thousands for receipts (not for TAT-QA).",
         "fixed_in_derived", ("cord",))
def _dot_thousands(row):
    return re.findall(r"(?<![\d.,])[1-9]\d{0,2}(?:\.\d{3})+(?![\d.,])", row["answers"][0] if row.get("answers") else "")


@anomaly("N2", "number", "Accounting negative in a derivation ('(6,919)')",
         "Evaluated as +6919, '7,572 - (6,919)' gives 653 instead of the gold 14,491.",
         "safe_eval reads a parenthesized bare number as negative.",
         "fixed_in_derived", ("tatqa",))
def _accounting_negative(row):
    out = []
    for _, a in _qas(row):
        d = _ANSWER_SPLIT.split(a or "", maxsplit=1)[0]
        if "\n" in (a or "") and re.search(r"[-+*/]\s*\(\s*\d[\d,.]*\s*\)", d):
            out.append(d[:50])
    return out


@anomaly("N3", "number", "Percent used as a fraction ('65% * 3,433')",
         "Dropping '%' gives 223,145 instead of 2,231.45; the derivation looks wrong when it is right.",
         "Both readings (percentage points, fraction) are tried; the one matching the gold is kept.",
         "fixed_in_derived", ("tatqa",))
def _percent_fraction(row):
    out = []
    for _, a in _qas(row):
        parts = _ANSWER_SPLIT.split(a or "", maxsplit=1)
        if len(parts) == 2 and "%" in parts[0]:
            g = parse_number(parts[1].strip().rstrip("."), dot_thousands=False)
            v, f = safe_eval(parts[0]), safe_eval(parts[0], pct_as_fraction=True)
            if g is not None and f is not None and _close(f, g, parts[1]) and not (v is not None and _close(v, g, parts[1])):
                out.append(parts[0][:50])
    return out


@anomaly("N4", "number", "Decimal with a leading zero and 3 decimals ('0.039')",
         "A thousands-separator rule reads it as 39.",
         "Thousands groups must start with a non-zero digit.",
         "fixed_in_derived")
def _leading_zero_decimal(row):
    return [a for _, a in _qas(row) if re.fullmatch(r"\s*-?0\.\d{3}\.?\s*", a or "")]


def audit_row(row: dict) -> dict[str, list[str]]:
    """anomaly id -> hits in this row (only ids with hits)."""
    out = {}
    for a in REGISTRY:
        if a.sources and row.get("source") not in a.sources:
            continue
        hits = a.detect(row)
        if hits:
            out[a.id] = hits
    return out


__all__ = ["FAMILIES", "REGISTRY", "STATUS", "Anomaly", "audit_row"]

"""Per-benchmark rationale extraction for UDD rows — evidence-bearing, verified, with provenance.

Public benchmarks almost never ship a rationale, yet many of them ship the *material* one is made
of: a derivation string (TAT-QA), a key->value linking (FUNSD), a receipt parse whose amounts must
add up (CORD), OCR words with boxes (OCR-VQA), UI elements (ScreenQA), layout blocks (OmniDocBench,
DocLayNet, PubLayNet), an HTML table (PubTabNet). Each deriver below turns that material into a
concise ``reasoning`` string aligned with a QA — never inventing a value that is not in the row.

Every rationale carries a **tier** (how much to trust it):

* ``source``           — the benchmark shipped it (HallusionBench details, TAT-QA derivations);
* ``source_verified``  — shipped AND re-checked (the TAT-QA expression evaluates to the answer);
* ``derived_exact``    — computed from the row's own ground truth (CORD arithmetic, FUNSD links,
                         table cells, layout counts); exact by construction;
* ``evidence_match``   — the answer was matched to OCR/UI text in the row (OCR-VQA, ScreenQA);
                         the evidence is real, the pairing is a string match.

and passes a **quality gate** (:func:`quality`) before it is attached: non-empty, bounded, not a
restatement of question + answer, citing at least one evidence string that is present in the row,
and — for arithmetic — ending in a value equal to the gold answer.

Rationales either fill ``reasoning[i]`` of an EXISTING QA or come with a NEW derived QA (new
question + answer computed from the same ground truth). :func:`attach` writes both into a copy of
the row and records ``reasoning_provenance[i]`` so derived text is never mistaken for source GT.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

TIERS = ("source", "source_verified", "derived_exact", "evidence_match")


@dataclass
class Rationale:
    reasoning: str
    tier: str
    op: str                                   # lookup|arithmetic|count|compare|key_value|spatial|consistency|transcribe
    evidence: list[str] = field(default_factory=list)   # strings that must exist in the row payload
    qa_index: int | None = None               # attach to an existing QA ...
    question: str | None = None               # ... or create a new one
    answers: list[str] = field(default_factory=list)
    repaired_answer: str | None = None        # corrected gold for an existing QA (TAT-QA)
    box: list[float] | None = None            # normalized evidence box, when one exists
    deriver: str = ""

    @property
    def is_new(self) -> bool:
        return self.qa_index is None


# --------------------------------------------------------------------------- helpers
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "by", "with", "from", "and", "or",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it", "its",
    "as", "which", "what", "who", "whom", "whose", "where", "when", "how", "many", "much",
    "does", "do", "did", "there", "their", "than", "then", "into", "over", "under", "about",
    "between", "page", "document", "image", "please", "answer",
}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", (s or "").lower())


def _content(s: str) -> set[str]:
    return {t for t in _tokens(s) if t not in _STOP}


_THOUSANDS = re.compile(r"^-?[1-9]\d{0,2}([.,]\d{3})+$")       # "0.039" is a decimal, not 39
_COMMA_THOUSANDS = re.compile(r"^-?[1-9]\d{0,2}(,\d{3})+$")


def parse_number(s: Any, dot_thousands: bool = True) -> float | None:
    """'24,000' / '60.000' (thousands) / '$1,298.00' / '(230)' / '-34.12%' / '48.5.' -> float.

    ``dot_thousands`` reads '60.000' as 60000 (Indonesian receipts, CORD); pass False for sources
    that only use commas as thousands separators (TAT-QA), where '1.234' is a decimal."""
    if s is None:
        return None
    t = str(s).strip().rstrip(".").strip()
    neg = t.startswith("(") and t.endswith(")")
    t = re.sub(r"[\s$€£¥₩()%]|Rp|RM", "", t)
    if not t:
        return None
    if (_THOUSANDS if dot_thousands else _COMMA_THOUSANDS).match(t):
        t = re.sub(r"[.,]", "", t)
    elif t.count(",") and t.count("."):
        t = t.replace(",", "")
    elif t.count(",") == 1 and len(t.split(",")[1]) != 3:
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def fmt_number(v: float, like: str | None = None) -> str:
    """Format ``v`` with the decimals of ``like`` (or minimal decimals)."""
    if like is not None:
        m = re.search(r"[.,](\d+)\D*$", str(like).strip().rstrip("."))
        dec = len(m.group(1)) if m and not _THOUSANDS.match(re.sub(r"[^\d.,-]", "", str(like))) else 0
        return f"{v:,.{dec}f}" if abs(v) >= 1000 else f"{v:.{dec}f}"
    return f"{v:g}"


def _box_of(el: dict, W: float = 0, H: float = 0) -> list[float] | None:
    """Normalized box of an element, or None.

    PIXEL boxes are not trusted: UDD builds up to fine-grained-v3 resized images to <=1000 px but
    kept pixel boxes in the ORIGINAL frame (39% of CORD and 60% of ScreenQA boxes fall outside the
    stored image, and the rest are mis-scaled too). ``UnifiedLoader.iter`` now normalizes them
    before resizing; until the corpus is rebuilt, a pixel box would put a rationale's 'top-right'
    in the wrong place, so it is dropped (the element's TEXT is still used as evidence)."""
    b = el.get("bbox")
    if not b or len(b) < 5 or not b[4]:
        return None
    x1, y1, x2, y2 = (float(v) for v in b[:4])
    if min(x1, y1) < -0.02 or max(x2, y2) > 1.02 or x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def grid_pos(b: list[float]) -> str:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "center")
    row = "top" if cy < 1 / 3 else ("bottom" if cy > 2 / 3 else "middle")
    return "center" if (row, col) == ("middle", "center") else f"{row}-{col}"


def relation(a: list[float], b: list[float]) -> str:
    """Dominant relation of box a relative to box b."""
    dx = (a[0] + a[2]) / 2 - (b[0] + b[2]) / 2
    dy = (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2
    if abs(dx) >= abs(dy):
        return "to the right of" if dx > 0 else "to the left of"
    return "below" if dy > 0 else "above"


def _elements(row: dict) -> list[dict]:
    try:
        return json.loads(row.get("elements_json") or "[]")
    except (TypeError, ValueError):
        return []


def _full(row: dict) -> dict:
    try:
        return json.loads(row.get("full_annotation") or "{}") or {}
    except (TypeError, ValueError):
        return {}


def _wh(row: dict) -> tuple[float, float]:
    return float(row.get("image_width") or 0), float(row.get("image_height") or 0)


# --------------------------------------------------------------------------- safe arithmetic
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str, pct_as_fraction: bool = False) -> float | None:
    """Evaluate + - * / ( ) over plain numbers ('$8,452-$10,322' ok). None if anything else.

    ``pct_as_fraction`` reads '65%' as 0.65 ('65% * 3,433'); by default '%' is dropped, which is
    right for percentage-point arithmetic ('12.8% - 59.9%')."""
    e = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)
    if pct_as_fraction:
        e = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", e)
    e = re.sub(r"\((\s*\d[\d.]*\s*)\)", r"(-\1)", e)        # accounting negative: (6919) = -6919
    e = re.sub(r"[$€£¥%]", "", e).replace("×", "*").replace("÷", "/").strip()
    if not e or not re.fullmatch(r"[\d.\s+\-*/()]+", e):
        return None
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    try:
        v = ev(tree)
    except (ValueError, ZeroDivisionError):
        return None
    return v if math.isfinite(v) else None


def _close(a: float, b: float, answer_text: str = "") -> bool:
    """Equal up to the rounding the gold answer was written with."""
    m = re.search(r"\.(\d+)", answer_text or "")
    tol = 0.5 * 10 ** -(len(m.group(1)) if m else 0)
    return abs(a - b) <= max(tol, 1e-6 * max(1.0, abs(b))) * (1 + 1e-9) + 1e-12


# --------------------------------------------------------------------------- derivers
DERIVERS: dict[str, Callable[[dict], list[Rationale]]] = {}


def deriver(*sources: str):
    def deco(fn):
        for s in sources:
            DERIVERS[s] = fn
        return fn
    return deco


_ANSWER_SPLIT = re.compile(r"\n\s*Answer:\s*", re.IGNORECASE)


@deriver("tatqa")
def _tatqa(row: dict) -> list[Rationale]:
    """TAT-QA (via the_cauldron) ships its derivation INSIDE the answer: '74.5-26.0\\nAnswer: 48.5.'.
    Split it out (repairing the gold), and verify the expression evaluates to the answer —
    applying the question's 'Answer scale should be: percent' (x100) when stated."""
    out = []
    for i, (q, a) in enumerate(zip(row["instructions"], row["answers"])):
        parts = _ANSWER_SPLIT.split(a or "", maxsplit=1)
        if len(parts) != 2:
            continue
        deriv, final = parts[0].strip(), parts[1].strip().rstrip(".").strip()
        if not (deriv and final):
            continue
        gold = parse_number(final, dot_thousands=False)
        items = [x.strip() for x in re.split(r",|;|\band\b", deriv) if x.strip()]
        has_op = bool(re.search(r"[\d)%]\s*[-+*/×÷]\s*[\d($]", deriv))
        if not has_op and gold is not None and float(len(items)) == gold and gold >= 1:
            out.append(Rationale(f"The qualifying entries are {', '.join(items)}; counting them gives {final}.",
                                 "source_verified", "count", items, i, repaired_answer=final,
                                 deriver="tatqa_count"))
            continue
        percent = "scale should be: percent" in (q or "").lower()
        cands = [safe_eval(deriv), safe_eval(deriv, pct_as_fraction=True)] if has_op else []
        cands = [c for c in cands if c is not None]
        v = next((c for c in cands if gold is not None and (_close(c, gold, final) or
                                                            (percent and _close(c * 100, gold, final)))),
                 cands[0] if cands else None)
        if v is not None and gold is not None:
            if _close(v, gold, final):
                text = f"Compute {deriv} = {final}."
            elif percent and _close(v * 100, gold, final):
                text = f"Compute {deriv} = {fmt_number(v, '0.0000')}; as a percentage: {final}%."
            else:                                           # derivation does not reproduce the gold:
                out.append(Rationale("derivation_mismatch", "rejected", "repair", [], i,
                                     repaired_answer=final, deriver="tatqa_derivation"))
                continue
            out.append(Rationale(text, "source_verified", "arithmetic", [deriv], i,
                                 repaired_answer=final, deriver="tatqa_derivation"))
            continue
        if not re.search(r"\d", deriv):         # a plan ("Find the year with ..."), not evidence
            out.append(Rationale("derivation_without_values", "rejected", "repair", [], i,
                                 repaired_answer=final, deriver="tatqa_span"))
            continue
        op = "compare" if re.search(r"\bvs\.?\b|>|<", deriv) else "lookup"
        out.append(Rationale(f"The relevant entries are {deriv}." if op == "compare"
                             else f"The relevant entry is {deriv}.", "source", op, [deriv], i,
                             repaired_answer=final, deriver="tatqa_span"))
    return out


@deriver("docmatix")
def _docmatix(row: dict) -> list[Rationale]:
    """A few Docmatix answers restate the question before '\\nAnswer:'; repair the gold only."""
    out = []
    for i, a in enumerate(row["answers"]):
        parts = _ANSWER_SPLIT.split(a or "", maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            out.append(Rationale("", "source", "repair", [], i, repaired_answer=parts[1].strip(),
                                 deriver="docmatix_repair"))
    return out


_VERDICT = re.compile(r"^\s*(yes|no)\b[\s.,;:!-]*", re.IGNORECASE)


@deriver("hallusionbench")
def _hallusion(row: dict) -> list[Rationale]:
    """HallusionBench details. Some open with the verdict of the PAIRED question ("Yes. In 2013,
    Hong Kong was the largest ..." under a question about the United States, gold "no"): the
    verdict contradicts this QA's gold while the statement after it is still the evidence. Drop
    only that leading verdict; the gate then judges what is left."""
    out = []
    answers = row.get("answers") or []
    for i, r in enumerate(row.get("reasoning") or []):
        text = (r or "").strip()
        if not text:
            continue
        gold = (answers[i] if i < len(answers) else "").strip().lower().rstrip(".")
        m = _VERDICT.match(text)
        if m and gold in ("yes", "no") and m.group(1).lower() != gold and text[m.end():].strip():
            rest = text[m.end():].strip()
            out.append(Rationale(rest[0].upper() + rest[1:], "source", "consistency", [rest], i,
                                 deriver="hallusionbench_details_verdict_stripped"))
            continue
        out.append(Rationale(text, "source", "consistency", [text], i, deriver="hallusionbench_details"))
    return out


def _cord_items(menu) -> list[dict]:
    items = menu if isinstance(menu, list) else [menu] if isinstance(menu, dict) else []
    return [m for m in items if isinstance(m, dict)]


@deriver("cord")
def _cord(row: dict) -> list[Rationale]:
    """Receipt arithmetic from the KIE parse: subtotal = Σ line prices, total = subtotal + tax +
    service − discount, change = cash − total, item count. Emitted only when the printed values
    actually satisfy the relation (so the rationale is true of THIS receipt)."""
    try:
        gt = json.loads(row["answers"][0])
    except (ValueError, IndexError, TypeError):
        return []
    items = _cord_items(gt.get("menu"))
    sub, tot = gt.get("sub_total") or {}, gt.get("total") or {}
    if isinstance(sub, list):
        sub = sub[0] if sub and isinstance(sub[0], dict) else {}
    if isinstance(tot, list):
        tot = tot[0] if tot and isinstance(tot[0], dict) else {}
    out: list[Rationale] = []
    priced = [(m.get("nm", "item"), m.get("price")) for m in items if parse_number(m.get("price")) is not None]
    sub_txt = sub.get("subtotal_price")
    if len(priced) >= 2 and parse_number(sub_txt) is not None:
        s = sum(parse_number(p) for _, p in priced)
        if _close(s, parse_number(sub_txt)):
            terms = " + ".join(f"{p} ({n})" for n, p in priced)
            out.append(Rationale(f"Add the line-item prices: {terms} = {sub_txt}, which matches the printed subtotal.",
                                 "derived_exact", "arithmetic", [p for _, p in priced] + [sub_txt],
                                 question="What is the subtotal on this receipt?", answers=[sub_txt],
                                 deriver="cord_subtotal"))
    total_txt = tot.get("total_price")
    base = parse_number(sub_txt) if sub_txt else None
    if base is not None and parse_number(total_txt) is not None:
        adds = [(k, sub.get(k)) for k in ("tax_price", "service_price", "othersvc_price")
                if parse_number(sub.get(k)) is not None]
        disc = parse_number(sub.get("discount_price"))
        disc_txt = str(sub.get("discount_price") or "").strip().lstrip("-").strip()
        val = base + sum(parse_number(v) for _, v in adds) - (abs(disc) if disc else 0.0)
        if (adds or disc) and _close(val, parse_number(total_txt)):
            expr = f"{sub_txt} (subtotal)" + "".join(f" + {v} ({k.split('_')[0]})" for k, v in adds) + \
                   (f" − {disc_txt} (discount)" if disc else "")
            out.append(Rationale(f"{expr} = {total_txt}, the printed total.", "derived_exact", "arithmetic",
                                 [sub_txt, total_txt] + [v for _, v in adds],
                                 question="What is the total amount to pay?", answers=[total_txt],
                                 deriver="cord_total"))
    cash, change = tot.get("cashprice"), tot.get("changeprice")
    if all(parse_number(x) is not None for x in (cash, change, total_txt)) and \
            _close(parse_number(cash) - parse_number(total_txt), parse_number(change)):
        out.append(Rationale(f"Cash paid {cash} minus total {total_txt} = {change}.", "derived_exact",
                             "arithmetic", [cash, total_txt, change],
                             question="How much change was given back?", answers=[change],
                             deriver="cord_change"))
    if len(priced) >= 2:
        name, price = max(priced, key=lambda t: parse_number(t[1]))
        if sum(1 for _, p in priced if parse_number(p) == parse_number(price)) == 1:
            listing = ", ".join(f"{n} {p}" for n, p in priced)
            out.append(Rationale(f"Compare the line prices ({listing}); the highest is {price} for {name}.",
                                 "derived_exact", "compare", [name, price],
                                 question="Which item on the receipt is the most expensive?", answers=[name],
                                 deriver="cord_max_item"))
    return out


def _funsd_entities(els: list[dict], W: float, H: float) -> list[tuple[str, str, list[float] | None]]:
    """Merge word-level BIO elements into entities (label, text, union box)."""
    ents: list[list] = []
    for e in els:
        cls, txt, b = e.get("class") or "O", (e.get("information") or "").strip(), _box_of(e, W, H)
        lab = cls.split("-", 1)[-1]
        if cls.startswith("B-") or (cls.startswith("I-") and (not ents or ents[-1][0] != lab)):
            ents.append([lab, txt, b])
        elif cls.startswith("I-") and ents:
            ents[-1][1] = f"{ents[-1][1]} {txt}".strip()
            if b and ents[-1][2]:
                eb = ents[-1][2]
                ents[-1][2] = [min(eb[0], b[0]), min(eb[1], b[1]), max(eb[2], b[2]), max(eb[3], b[3])]
            elif b:
                ents[-1][2] = b
    return [(lab, txt, b) for lab, txt, b in ents]


@deriver("funsd")
def _funsd(row: dict, max_items: int = 4) -> list[Rationale]:
    """FUNSD's QA gold is the source key->value linking; ground both ends in the word boxes and
    state where the value sits relative to its label."""
    W, H = _wh(row)
    ents = _funsd_entities(_elements(row), W, H)
    by_text: dict[tuple[str, str], list[float] | None] = {}
    for lab, txt, b in ents:
        by_text.setdefault((lab, re.sub(r"\s+", " ", txt)), b)
    out = []
    for a in row["answers"]:
        if not a.startswith("{"):
            continue
        try:
            kv = ast.literal_eval(a)
        except (ValueError, SyntaxError):
            continue
        for key, vals in kv.items():
            vals = vals if isinstance(vals, list) else [vals]
            if len(out) >= max_items or len(vals) != 1 or not str(vals[0]).strip():
                continue
            val = re.sub(r"\s+", " ", str(vals[0]))
            kb, vb = by_text.get(("QUESTION", re.sub(r"\s+", " ", key))), by_text.get(("ANSWER", val))
            if not (kb and vb):
                continue
            label = key.rstrip(": ").strip() or key
            out.append(Rationale(
                f"The label '{key}' is in the {grid_pos(kb)} of the form; the answer linked to it sits "
                f"{relation(vb, kb)} the label and reads '{val}'.",
                "derived_exact", "key_value", [key, val],
                question=f"On this form, what is filled in for '{label}'?", answers=[val], box=vb,
                deriver="funsd_link"))
    return out


_GENRE_Q = re.compile(r"what (type|kind|genre) of book", re.IGNORECASE)
_REL_Q = re.compile(r"is this (?:book )?related to (.+?)\??$", re.IGNORECASE)
_TITLE_Q = re.compile(r"(title|author|who wrote|edition)", re.IGNORECASE)


@deriver("ocrvqa")
def _ocrvqa(row: dict) -> list[Rationale]:
    """Book covers: (a) title/author answers matched to OCR words (position = where they are
    printed); (b) 'Is this book related to X?' answered from the SAME row's genre answer."""
    W, H = _wh(row)
    words = [(e.get("information") or "", _box_of(e, W, H)) for e in _elements(row)]
    words = [(w.lower(), b) for w, b in words if w.strip() and b]
    out = []
    genre = next((a for q, a in zip(row["instructions"], row["answers"]) if _GENRE_Q.search(q)), None)
    for i, (q, a) in enumerate(zip(row["instructions"], row["answers"])):
        m = _REL_Q.search(q.strip())
        if m and genre and a.strip().lower().rstrip(".") in ("yes", "no"):
            topic = m.group(1).strip()
            same = topic.lower() == genre.lower()
            if same == (a.strip().lower().startswith("yes")):
                why = (f"The cover is categorised as '{genre}', exactly the topic asked about."
                       if same else f"The cover is categorised as '{genre}', a different category from '{topic}'.")
                out.append(Rationale(why, "derived_exact", "consistency", [genre], i,
                                     deriver="ocrvqa_genre_consistency"))
            continue
        if not _TITLE_Q.search(q) or not words:
            continue
        ans_toks = [t for t in _tokens(a) if t not in _STOP and len(t) > 2]
        hits = [(w, b) for w, b in words if w in ans_toks]
        if len(ans_toks) >= 1 and len({w for w, _ in hits}) >= max(1, math.ceil(0.6 * len(set(ans_toks)))):
            # a word can recur ("Killer Su Doku" in the title AND in a tagline): anchor on the
            # largest printed hit (titles are big) and take each other word's NEAREST occurrence
            def _area(b):
                return (b[2] - b[0]) * (b[3] - b[1])

            def _dist(a, b):
                return ((a[0] + a[2]) - (b[0] + b[2])) ** 2 + ((a[1] + a[3]) - (b[1] + b[3])) ** 2
            anchor_w, anchor_b = max(hits, key=lambda t: _area(t[1]))
            chosen = {anchor_w: anchor_b}
            for w in {w for w, _ in hits} - {anchor_w}:
                b = min((b for x, b in hits if x == w), key=lambda b: _dist(b, anchor_b))
                if _dist(b, anchor_b) <= (2 * 0.4) ** 2:      # centers within 40% of the page
                    chosen[w] = b                            # (else: only found far away -> unmatched)
            hits = list(chosen.items())
            if len(hits) < max(1, math.ceil(0.6 * len(set(ans_toks)))):
                continue
            boxes = [b for _, b in hits]
            ub = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            seen = list(dict.fromkeys(w for w, _ in sorted(hits, key=lambda t: (round(t[1][1], 2), t[1][0]))))
            n_ans = len(set(ans_toks))
            out.append(Rationale(f"The cover text in the {grid_pos(ub)} area includes the words "
                                 f"{', '.join(repr(w) for w in seen)}, matching {len(seen)} of the "
                                 f"{n_ans} key words of the answer.",
                                 "evidence_match", "lookup", seen, i, box=ub, deriver="ocrvqa_words"))
    return out


@deriver("screenqa")
def _screenqa(row: dict) -> list[Rationale]:
    """ScreenQA answers are sentences built from UI elements: cite the elements whose text is in
    the answer (with their screen position)."""
    W, H = _wh(row)
    els = [(e.get("information") or "", _box_of(e, W, H)) for e in _elements(row)]
    out = []
    for i, (q, a) in enumerate(zip(row["instructions"], row["answers"])):
        at, qt = _content(a), _content(q)
        cited = []
        for txt, b in els:
            ct = _content(txt) - qt
            if txt.strip() and ct and len(ct & at) / len(ct) >= 0.5:
                cited.append((txt.strip(), b))
        if cited:
            parts = "; ".join(f"'{t}'" + (f" ({grid_pos(b)})" if b else "") for t, b in cited[:4])
            boxes = [b for _, b in cited if b]
            ub = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes),
                  max(b[3] for b in boxes)] if boxes else None
            out.append(Rationale(f"The screen shows {parts}; the answer is composed from "
                                 f"{'this element' if len(cited) == 1 else 'these elements'}.",
                                 "evidence_match", "lookup", [t for t, _ in cited], i,
                                 box=ub, deriver="screenqa_elements"))
    return out


_LAYOUT_NAMES = {"Table": "table", "Picture": "picture", "Figure": "figure", "Formula": "formula",
                 "List-item": "list item", "Section-header": "section header", "Title": "title",
                 "table": "table", "figure": "figure", "equation_isolated": "displayed formula",
                 "title": "title", "header": "page header"}


def _layout_counts(row: dict, deriver_name: str, classes: dict[str, str]) -> list[Rationale]:
    W, H = _wh(row)
    els = [(e.get("class"), _box_of(e, W, H)) for e in _elements(row)]
    out = []
    for cls, name in classes.items():
        boxes = [b for c, b in els if c == cls and b]
        if not boxes or len(boxes) > 6:
            continue
        from collections import Counter
        pos = Counter(grid_pos(b) for b in sorted(boxes, key=lambda b: (b[1], b[0])))
        where = ", ".join(f"{k} in the {p}" if k > 1 else f"one in the {p}" for p, k in pos.items())
        n = len(boxes)
        out.append(Rationale(f"Scanning the page for {name} regions finds {n}: {where}.",
                             "derived_exact", "count", [cls],
                             question=f"How many {name}{'s' if not name.endswith('s') else ''} are on this page?",
                             answers=[str(n)], deriver=deriver_name))
        if len(out) >= 2:
            break
    return out


@deriver("doclaynet", "publaynet")
def _layout(row: dict) -> list[Rationale]:
    classes = {k: v for k, v in _LAYOUT_NAMES.items() if k[0].isupper()}
    if row.get("source") == "publaynet":
        # PubLayNet's "Title" also labels figure tags ("Figure 1") and run-in headings, so "how
        # many titles?" would not match what a reader counts; keep the unambiguous classes
        classes.pop("Title", None)
    return _layout_counts(row, f"{row.get('source')}_count", classes)


@deriver("omnidocbench")
def _omnidoc(row: dict) -> list[Rationale]:
    """Page number / title lookups by layout class, plus element counts."""
    W, H = _wh(row)
    els = [(e.get("class"), (e.get("information") or "").strip(), _box_of(e, W, H)) for e in _elements(row)]
    out = []
    pn = [(t, b) for c, t, b in els if c == "page_number" and t and b]
    if len(pn) == 1 and len(pn[0][0]) <= 8:
        t, b = pn[0]
        out.append(Rationale(f"The element marked as the page number, at the {grid_pos(b)} of the page, reads '{t}'.",
                             "derived_exact", "lookup", [t], question="What is the page number?",
                             answers=[t], box=b, deriver="omnidoc_page_number"))
    titles = [(re.sub(r"^#+\s*", "", t).strip(), b) for c, t, b in els if c == "title" and t and b]
    if len(titles) == 1 and titles[0][0] and len(titles[0][0]) <= 120 and "$" not in titles[0][0]:
        t, b = titles[0]                          # markdown heading marks are not on the page
        out.append(Rationale(f"The block laid out as the title, in the {grid_pos(b)} of the page, reads '{t}'.",
                             "derived_exact", "lookup", [t], question="What is the title on this page?",
                             answers=[t], box=b, deriver="omnidoc_title"))
    out += _layout_counts(row, "omnidoc_count", {k: v for k, v in _LAYOUT_NAMES.items() if k[0].islower()})
    return out


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self.head_rows = 0
        self._in_head = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_head = True
        elif tag == "tr":
            self.rows.append([])
            if self._in_head:
                self.head_rows += 1
        elif tag in ("td", "th"):
            self._cell = []
            self._span = (int(a.get("rowspan", 1) or 1), int(a.get("colspan", 1) or 1))

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_head = False
        elif tag in ("td", "th") and self._cell is not None and self.rows:
            self.rows[-1].append((re.sub(r"\s+", " ", "".join(self._cell)).strip(), *self._span))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table(html: str) -> tuple[list[list[str]], int] | None:
    """HTML -> (rectangular grid, n header rows); None when any span makes the grid ambiguous."""
    p = _TableParser()
    try:
        p.feed(html or "")
    except (ValueError, AssertionError):        # malformed markup / span attribute
        return None
    rows = [r for r in p.rows if r]
    if not rows or any(rs > 1 or cs > 1 for r in rows for _, rs, cs in r):
        return None
    width = len(rows[0])
    if width < 2 or any(len(r) != width for r in rows):
        return None
    return [[c for c, _, _ in r] for r in rows], p.head_rows


@deriver("pubtabnet")
def _pubtabnet(row: dict, max_items: int = 2) -> list[Rationale]:
    """Cell lookup by (row label, column header) on span-free tables; the header and row label
    must be unique so the question has exactly one answer."""
    parsed = parse_table(_full(row).get("html") or "")
    if not parsed:
        return []
    grid, n_head = parsed
    n_head = max(1, n_head)
    header = grid[n_head - 1]
    body = grid[n_head:]
    labels = [r[0] for r in body]
    out = []
    if body:
        out.append(Rationale(f"Below the {n_head} header row{'s' if n_head > 1 else ''}, the table has "
                             f"{len(body)} body rows (first: '{labels[0]}', last: '{labels[-1]}').",
                             "derived_exact", "count", [labels[0], labels[-1]],
                             question="How many rows does the table body have (excluding the header)?",
                             answers=[str(len(body))], deriver="pubtabnet_rows"))
    for j in range(1, len(header)):
        col = header[j]
        if len(out) > max_items or not col or header.count(col) != 1:
            continue
        for r in body:
            if r[0] and labels.count(r[0]) == 1 and r[j] and len(r[j]) <= 40:
                out.append(Rationale(f"Find the row '{r[0]}' and the column '{col}'; the cell where they "
                                     f"meet reads '{r[j]}'.", "derived_exact", "lookup", [r[0], col, r[j]],
                                     question=f"In the table, what is the '{col}' value for '{r[0]}'?",
                                     answers=[r[j]], deriver="pubtabnet_cell"))
                break
    return out


# --------------------------------------------------------------------------- quality gate
def _payload_text(row: dict) -> str:
    fa = _full(row)
    parts = list(row.get("instructions") or []) + list(row.get("answers") or [])
    parts += [e.get("information") or "" for e in _elements(row)] + [e.get("class") or "" for e in _elements(row)]
    parts += [fa.get("text") or "", fa.get("html") or ""] + list(row.get("reasoning") or [])
    return "\n".join(parts).lower()


def quality(r: Rationale, row: dict) -> tuple[bool, str]:
    """Gate: (ok, reason). See module docstring."""
    if r.tier == "rejected":
        return False, r.reasoning
    if r.op == "repair":
        return True, "repair"
    text = r.reasoning.strip()
    if not text:
        return False, "empty"
    if len(text) > 600:
        return False, "too_long"
    if r.tier not in TIERS:
        return False, "unknown_tier"
    q = r.question if r.is_new else row["instructions"][r.qa_index]
    ans = r.answers if r.is_new else [r.repaired_answer or row["answers"][r.qa_index]]
    novel = _content(text) - _content(q or "") - set().union(*(_content(a) for a in ans))
    if len(novel) < (1 if r.tier.startswith("source") else 2):
        return False, "tautological"
    a0 = (ans[0] if ans else "").strip().lower().rstrip(".")
    lead = re.match(r"\s*(yes|no)\b", text, re.IGNORECASE)
    if a0 in ("yes", "no") and lead and lead.group(1).lower() != a0:
        return False, "contradicts_answer"
    blob = _payload_text(row)
    if not r.evidence or not any(str(e).strip().lower() in blob for e in r.evidence if str(e).strip()):
        return False, "evidence_not_in_row"
    if r.op == "arithmetic":
        nums = re.findall(r"-?[\d][\d,]*\.?\d*", text)
        gold = parse_number(ans[0]) if ans else None
        last = parse_number(nums[-1]) if nums else None
        if gold is None or last is None or not _close(last, gold, ans[0]):
            return False, "arithmetic_mismatch"
    return True, "ok"


# --------------------------------------------------------------------------- attach
def derive(row: dict) -> list[Rationale]:
    fn = DERIVERS.get(row.get("source", ""))
    return fn(row) if fn else []


def attach(row: dict, *, tiers: tuple[str, ...] = TIERS, max_new: int = 4) -> tuple[dict, dict]:
    """Return (new_row, report). Fills reasoning for existing QAs, repairs answers, appends derived
    QAs (task_detail ``derived:<deriver>``), and adds aligned ``reasoning_provenance`` and
    ``evidence_box`` (normalized ``[x1,y1,x2,y2]`` or None) columns."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in row.items()}
    n = len(out["instructions"])
    out["reasoning"] = list(out.get("reasoning") or [""] * n) + [""] * max(0, n - len(out.get("reasoning") or []))
    out["answer_variants"] = [list(v) for v in (out.get("answer_variants") or [[a] for a in out["answers"]])]
    prov = ["source" if (x or "").strip() else "" for x in out["reasoning"]]
    boxes: list[list[float] | None] = [None] * n
    report = {"kept": [], "rejected": []}
    new = 0
    for r in derive(row):
        ok, why = quality(r, row)
        if not ok or (r.tier not in tiers and r.op != "repair"):
            report["rejected"].append((r.deriver, why if not ok else "tier_filtered"))
            # a SOURCE rationale that fails the gate is moved aside (kept, but not trained on)
            # (also when the deriver had rewritten it, e.g. a stripped verdict)
            existing = (out["reasoning"][r.qa_index] or "").strip() if not r.is_new else ""
            if existing and r.tier.startswith("source") and r.op != "repair":
                out.setdefault("reasoning_rejected", [""] * n)[r.qa_index] = existing
                out["reasoning"][r.qa_index] = ""
                prov[r.qa_index] = f"rejected:{why}"
            # a contaminated gold is repaired even when its rationale is not kept
            if not r.is_new and r.repaired_answer and r.repaired_answer != out["answers"][r.qa_index]:
                out["answers"][r.qa_index] = r.repaired_answer
                out["answer_variants"][r.qa_index] = [r.repaired_answer]
                prov[r.qa_index] = prov[r.qa_index] or "repaired"
            continue
        report["kept"].append((r.deriver, r.tier, r.op))
        if r.is_new:
            if new >= max_new:
                continue
            new += 1
            out["instructions"].append(r.question)
            out["answers"].append(r.answers[0])
            out["answer_variants"].append(list(dict.fromkeys(r.answers)))
            out["reasoning"].append(r.reasoning)
            out.setdefault("task_detail", [""] * n).append(f"derived:{r.deriver}")
            out.setdefault("qa_tasks", ["vqa"] * n).append("reasoning")
            prov.append(f"{r.tier}:{r.deriver}")
            boxes.append(r.box)
            continue
        i = r.qa_index
        if r.repaired_answer is not None and r.repaired_answer != out["answers"][i]:
            out["answers"][i] = r.repaired_answer
            out["answer_variants"][i] = [r.repaired_answer]
            prov[i] = prov[i] or "repaired"
        if r.reasoning:
            out["reasoning"][i] = r.reasoning
            prov[i] = f"{r.tier}:{r.deriver}"
            boxes[i] = boxes[i] or r.box
    if "reasoning_rejected" in out:
        out["reasoning_rejected"] += [""] * (len(out["instructions"]) - len(out["reasoning_rejected"]))
    out["reasoning_provenance"] = prov
    out["evidence_box"] = boxes
    return out, report

"""Model-free **reasoning** generator for synthetic documents.

The generator authors every value, so any count / aggregate / comparison / relation over a document's
structured content is ground truth *by construction* — no external model. This module turns structured
content (a typed table, or a labelled sequence) into a **varied, sampled** set of reasoning questions,
each with the answer and a chain-of-thought rationale (the A2 target).

Design = a small registry of "question kinds". Each kind is a pure function ``(ctx, rng) -> QA | None``
(``None`` when it doesn't apply). A driver samples a few distinct kinds per document, seeded from the
content, so **different documents get different questions** → unbounded reasoning variety, while every
answer stays exact and reproducible. Add a new kind by writing one function and registering it.

A ``QA`` is a plain dict matching ``DocBuilder.qa`` kwargs:
``{"question", "answers"(list), "metric", "answer_type", "rationale"}``.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime

__all__ = ["table_questions", "sequence_questions"]

_NUM_RE = re.compile(r"^[\s$€£+]*-?[\d,]*\.?\d+\s*%?$")
_DATE_FMTS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y")


def _to_num(s) -> float | None:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if not _NUM_RE.match(s):
        return None
    try:
        return float(s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
                     .replace("%", "").replace("+", "").strip())
    except ValueError:
        return None


def _to_date(s):
    s = str(s).strip()
    for f in _DATE_FMTS:
        try:
            return datetime.strptime(s, f)
        except ValueError:
            continue
    return None


def _fmt(x: float) -> str:
    return f"{x:.2f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


def _numans(x: float) -> list[str]:
    """Acceptable string forms of a number (plain, 2dp, $, thousands) -> tolerant matching."""
    out = {_fmt(x), f"{x:.2f}", f"{x:,.2f}", f"${x:,.2f}", f"{x:g}"}
    return list(out)


# --------------------------------------------------------------------------- typed table context
@dataclass
class _Col:
    name: str
    kind: str            # 'num' | 'date' | 'text'
    raw: list            # original cell strings
    num: list            # parsed floats (num cols)
    date: list           # parsed datetimes (date cols)


@dataclass
class _Table:
    label: str
    header: list
    rows: list
    cols: list           # list[_Col]


def _build_table(label, header, rows) -> _Table:
    cols = []
    n = len(rows)
    for j, name in enumerate(header):
        raw = [str(r[j]) for r in rows]
        nums = [_to_num(v) for v in raw]
        dates = [_to_date(v) for v in raw]
        if n and sum(v is not None for v in nums) >= 0.8 * n:
            kind = "num"
        elif n and sum(v is not None for v in dates) >= 0.8 * n:
            kind = "date"
        else:
            kind = "text"
        cols.append(_Col(name, kind, raw, nums, dates))
    return _Table(label, header, rows, cols)


def _num_cols(t): return [c for c in t.cols if c.kind == "num"]
def _date_cols(t): return [c for c in t.cols if c.kind == "date"]
def _other_col(t, c): return next((o for o in t.cols if o is not c), None)


# --------------------------------------------------------------------------- table question kinds
def _q_count_rows(t, rng):
    return {"question": f"How many rows are in {t.label}?", "answers": [str(len(t.rows))],
            "metric": "exact", "answer_type": "H-count",
            "rationale": f"{t.label.capitalize()} has {len(t.rows)} data rows."}


def _q_sum(t, rng):
    cols = _num_cols(t)
    if not cols:
        return None
    c = rng.choice(cols)
    s = sum(c.num)
    return {"question": f"What is the total of the {c.name} column in {t.label}?",
            "answers": _numans(s), "metric": "relaxed_acc", "answer_type": "H1-aggregate",
            "rationale": f"Sum the {c.name} column: {' + '.join(_fmt(v) for v in c.num)} = {_fmt(s)}."}


def _q_mean(t, rng):
    cols = _num_cols(t)
    if not cols:
        return None
    c = rng.choice(cols)
    m = sum(c.num) / len(c.num)
    return {"question": f"What is the average (mean) {c.name} in {t.label}?",
            "answers": _numans(round(m, 2)), "metric": "relaxed_acc", "answer_type": "H1-aggregate",
            "rationale": f"Mean {c.name} = {_fmt(sum(c.num))} / {len(c.num)} = {m:.2f}."}


def _q_extreme(t, rng):
    cols = _num_cols(t)
    if not cols:
        return None
    c = rng.choice(cols)
    hi = rng.random() < 0.5
    val = max(c.num) if hi else min(c.num)
    return {"question": f"What is the {'largest' if hi else 'smallest'} value in the {c.name} column?",
            "answers": _numans(val), "metric": "relaxed_acc", "answer_type": "H1-aggregate",
            "rationale": f"The {'max' if hi else 'min'} of {c.name} is {_fmt(val)}."}


def _q_argmax_lookup(t, rng):
    """Relational: in the row with the max/min of a numeric col, read another column."""
    cols = _num_cols(t)
    if not cols or len(t.cols) < 2:
        return None
    c = rng.choice(cols)
    other = _other_col(t, c)
    if other is None:
        return None
    hi = rng.random() < 0.5
    i = max(range(len(c.num)), key=lambda k: c.num[k]) if hi else min(range(len(c.num)), key=lambda k: c.num[k])
    ans = other.raw[i]
    return {"question": f"In the row with the {'highest' if hi else 'lowest'} {c.name}, "
                        f"what is the {other.name}?",
            "answers": [ans], "metric": "ned", "answer_type": "H-comprehension",
            "rationale": f"Row {i+1} has the {'highest' if hi else 'lowest'} {c.name} "
                         f"({_fmt(c.num[i])}); its {other.name} is '{ans}'."}


def _q_threshold_count(t, rng):
    cols = _num_cols(t)
    if not cols or len(t.rows) < 3:
        return None
    c = rng.choice(cols)
    srt = sorted(c.num)
    thr = srt[len(srt) // 2]                 # ~median -> non-trivial split
    cnt = sum(1 for v in c.num if v > thr)
    return {"question": f"How many rows have {c.name} greater than {_fmt(thr)}?",
            "answers": [str(cnt)], "metric": "exact", "answer_type": "H-count",
            "rationale": f"{cnt} of the {c.name} values exceed {_fmt(thr)}."}


def _q_ordinal(t, rng):
    cols = _num_cols(t)
    if not cols or len(t.rows) < 3:
        return None
    c = rng.choice(cols)
    k = 2
    val = sorted(c.num, reverse=True)[k - 1]
    return {"question": f"What is the {k}nd-largest value in the {c.name} column?",
            "answers": _numans(val), "metric": "relaxed_acc", "answer_type": "H1-aggregate",
            "rationale": f"Sorted {c.name} descending, the {k}nd is {_fmt(val)}."}


def _q_compare_rows(t, rng):
    cols = _num_cols(t)
    if not cols or len(t.rows) < 2:
        return None
    c = rng.choice(cols)
    i, j = rng.sample(range(len(t.rows)), 2)
    bigger = i if c.num[i] > c.num[j] else j
    return {"question": f"Which has a larger {c.name}, row {i+1} or row {j+1}?",
            "answers": [f"row {bigger+1}", str(bigger + 1)], "metric": "anls",
            "answer_type": "H-comprehension",
            "rationale": f"Row {i+1} {c.name}={_fmt(c.num[i])} vs row {j+1} {c.name}={_fmt(c.num[j])} "
                         f"-> row {bigger+1}."}


def _q_date_extreme(t, rng):
    dcols = _date_cols(t)
    if not dcols or len(t.cols) < 2:
        return None
    c = rng.choice(dcols)
    other = _other_col(t, c)
    if other is None:
        return None
    early = rng.random() < 0.5
    idx = range(len(c.date))
    i = (min if early else max)(idx, key=lambda k: c.date[k])
    return {"question": f"Which {other.name} has the {'earliest' if early else 'latest'} {c.name}?",
            "answers": [other.raw[i]], "metric": "ned", "answer_type": "H-comprehension",
            "rationale": f"The {'earliest' if early else 'latest'} {c.name} ({c.raw[i]}) is row {i+1}, "
                         f"whose {other.name} is '{other.raw[i]}'."}


_TABLE_KINDS = [_q_count_rows, _q_sum, _q_mean, _q_extreme, _q_argmax_lookup,
                _q_threshold_count, _q_ordinal, _q_compare_rows, _q_date_extreme]


def table_questions(header, rows, *, label="the table", n=3, seed=None) -> list[dict]:
    """Sample up to ``n`` distinct reasoning questions over a typed (header, rows) table. Seeded from
    the content so different tables get different questions (variety) yet reproducibly."""
    if not rows or not header:
        return []
    t = _build_table(label, header, rows)
    rng = random.Random(seed if seed is not None else f"{label}:{header}:{rows}")
    out, used = [], set()
    for kind in rng.sample(_TABLE_KINDS, len(_TABLE_KINDS)):
        if len(out) >= n:
            break
        qa = kind(t, rng)
        if qa and qa["question"] not in used:
            out.append(qa)
            used.add(qa["question"])
    return out


# --------------------------------------------------------------------------- sequence question kinds
def sequence_questions(items, *, attr, label="items", value_names=None, n=3, seed=None) -> list[dict]:
    """Reasoning over a labelled sequence: ``items`` is a list of (attr_value, ...) and ``attr`` names
    the grouping attribute (e.g. side 'left'/'right', sender 'in'/'out'). Emits count-by-attribute,
    which-group-has-more, and total. ``value_names`` maps raw attr values to friendly words."""
    if not items:
        return []
    vals = [it[0] if isinstance(it, (tuple, list)) else it for it in items]
    rng = random.Random(seed if seed is not None else f"{label}:{attr}:{vals}")
    names = value_names or {}
    from collections import Counter
    counts = Counter(vals)
    distinct = list(counts)
    out = [{"question": f"How many {label} are there in total?", "answers": [str(len(vals))],
            "metric": "exact", "answer_type": "H-count",
            "rationale": f"There are {len(vals)} {label}."}]
    for v in distinct:
        nm = names.get(v, str(v))
        out.append({"question": f"How many {label} are {nm}?", "answers": [str(counts[v])],
                    "metric": "exact", "answer_type": "H-count",
                    "rationale": f"{counts[v]} of the {len(vals)} {label} are {nm}."})
    if len(distinct) >= 2:
        top = max(counts, key=counts.get)
        lo = min(counts, key=counts.get)
        if counts[top] == counts[lo]:
            ans = ["equal", "the same", "neither", "tie"]
            why = "both groups are equal"
        else:
            ans = [names.get(top, str(top))]
            why = f"{names.get(top, top)} has {counts[top]} vs {counts[lo]}"
        out.append({"question": f"Which group has more {label}, {' or '.join(names.get(v, str(v)) for v in distinct)}?",
                    "answers": ans, "metric": "anls", "answer_type": "H-comprehension",
                    "rationale": f"{why}."})
    rng.shuffle(out)
    return out[:n]

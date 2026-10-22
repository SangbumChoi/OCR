"""Config-driven instruction / answer diversification for UDD QAs.

The public corpus is narrow in two ways a model can overfit to: (1) instruction wording and format
clauses are fixed per source (every SROIE row says "Transcribe all the text ..."; the_cauldron
sources carry one of a handful of brevity suffixes), and (2) each answer appears in one surface
form (and sometimes a broken one: "Mean." with a trailing period in 100% of DVQA/PlotQA golds).

This module diversifies both — under one rule: **the instruction and the answer vary together**.
A *response format* (``short``, ``number``, ``yes_no``, ``sentence``, ``json``, ``mcq_index``,
``reasoned``, ``grounded``) binds an instruction clause to an answer renderer, and it applies only
when its *condition* holds for the QA (numeric gold, yes/no gold, options present, a gate-passed
rationale, an evidence box). So a variant never asks for a number and expects a sentence, and a
``reasoned`` target only exists where a verified rationale exists (``rationale.py``).

What changes, what does not:

* the **question semantics never change** — source questions keep their wording; only the
  conflicting source format clause is stripped and a new one is attached (prefix or suffix);
  derived questions (``derived:<deriver>``) are drawn from paraphrase pools in the config;
* the **gold value never changes** — only its rendering (``2,305.00`` -> ``2305``, ``yes`` ->
  ``Yes``, option text -> option number), and canonical answer variants are added for scoring.

Everything is deterministic in (``seed``, ``sample_id``, QA index), so a config + corpus fully
determines the output — one config = one dataset variant, as for the synthetic generator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rationale import parse_number

# --------------------------------------------------------------------------- config
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "instruction_diversity.yaml"


@dataclass
class FormatSpec:
    name: str
    when: list[str]
    clauses: list[str]
    render: str
    weight: float = 1.0
    metric: str = "source"                   # "source" = keep the QA's metric


@dataclass
class DiversityConfig:
    seed: int = 13
    variants_per_qa: int = 2
    keep_original: bool = True
    strip_source_format_clauses: bool = True
    clause_position: dict[str, float] = field(default_factory=lambda: {"suffix": 0.8, "prefix": 0.2})
    clause_separator: list[str] = field(default_factory=lambda: [" ", "\n"])
    canonicalize: dict[str, bool] = field(default_factory=lambda: {
        "strip_trailing_period": True, "numeric_variants": True, "case_variants": True})
    formats: list[FormatSpec] = field(default_factory=list)
    derived_question_templates: dict[str, list[str]] = field(default_factory=dict)
    sources: dict[str, dict] = field(default_factory=dict)    # per-source overrides

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> DiversityConfig:
        import yaml
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        fmts = [FormatSpec(name=k, **v) for k, v in (raw.pop("formats", {}) or {}).items()]
        return cls(formats=fmts, **raw)

    def for_source(self, source: str) -> DiversityConfig:
        """Apply ``sources.<name>`` overrides (variants_per_qa, disabled formats, weights)."""
        ov = self.sources.get(source) or {}
        if not ov:
            return self
        disabled = set(ov.get("disable_formats", []))
        weights = ov.get("format_weights", {})
        fmts = [FormatSpec(f.name, f.when, f.clauses, f.render, weights.get(f.name, f.weight), f.metric)
                for f in self.formats if f.name not in disabled]
        return DiversityConfig(self.seed, ov.get("variants_per_qa", self.variants_per_qa),
                               ov.get("keep_original", self.keep_original),
                               self.strip_source_format_clauses, self.clause_position,
                               self.clause_separator, self.canonicalize, fmts,
                               self.derived_question_templates, {})


# --------------------------------------------------------------------------- answers
_YES_NO = {"yes", "no"}


def strip_period(a: str) -> str:
    """'Mean.' -> 'Mean'; keeps abbreviations/decimals that END a longer sentence intact."""
    a = (a or "").strip()
    if a.endswith(".") and not a.endswith("..") and len(a.split()) <= 4:
        core = a[:-1]
        if not re.search(r"\b[A-Z]\.$|\b[a-z]{1,3}\.$", a) or parse_number(core) is not None:
            return core.strip()
    return a


def answer_variants(answer: str, cfg: DiversityConfig | None = None) -> list[str]:
    """Canonical answer first, then accepted surface variants (same value)."""
    c = (cfg.canonicalize if cfg else None) or {"strip_trailing_period": True,
                                                  "numeric_variants": True, "case_variants": True}
    a = (answer or "").strip()
    canon = strip_period(a) if c.get("strip_trailing_period") else a
    out = [canon, a]
    v = parse_number(canon)
    if c.get("numeric_variants") and v is not None and re.search(r"\d", canon) and len(canon) <= 24:
        out += [f"{v:g}", f"{v:,.2f}", f"{v:.2f}"] if not float(v).is_integer() else \
               [f"{int(v)}", f"{int(v):,}", f"{v:.2f}"]
        if canon.endswith("%"):
            out.append(f"{v:g}")
    if c.get("case_variants") and canon.lower() in _YES_NO:
        out += [canon.lower(), canon.capitalize()]
    return list(dict.fromkeys(x for x in out if x))


def is_numeric(a: str) -> bool:
    a = strip_period(a)
    return bool(re.fullmatch(r"[\s$€£¥₩+\-(]*\d[\d,]*(\.\d+)?\)?\s*%?", a or ""))


# --------------------------------------------------------------------------- questions
# Source-specific format clauses that would CONFLICT with a new format clause. Semantic clauses
# (e.g. TAT-QA's "Answer scale should be: percent.") are deliberately NOT matched.
_FORMAT_CLAUSE = re.compile(
    r"(?:\s*\|\s*|\n|\s+)(?:"
    r"(?:please\s+)?(?:offer|give|provide|keep|make|answer|reply|respond|your answer)[^.\n|]{0,60}"
    r"(?:short|brief|concise|quick|succinct|word|phrase|to the point)[^.\n|]{0,40}\.?"
    r"|quick response, please\.?"
    r"|answer concisely[^.\n]*\.?"
    r"|answer with (?:the|a) (?:single )?(?:word|phrase|text|value)[^.\n]*\.?"
    r")\s*$", re.IGNORECASE)
_OPTIONS = re.compile(r"Options:\s*(.+)$", re.IGNORECASE | re.DOTALL)


# The 18 brevity suffixes the_cauldron appends (measured over all UDD rows: DVQA, PlotQA, ST-VQA,
# TAT-QA, ...) plus 3 generic ones. Matched exactly as the last line; the regex above catches
# near-variants.
CAULDRON_SUFFIXES = {s.lower() for s in (
    "Give a very brief answer.", "Ensure brevity in your answer.", "Your response must be concise.",
    "Your answer should be very brief.", "Offer a terse response.", "Provide a succinct answer.",
    "Keep it short and to the point.", "Your answer should be compact.", "Offer a very short reply.",
    "Provide a short and direct response.", "Make the answer very short.", "Keep it brief.",
    "Short answer required.", "Write a very short answer.", "Quick response, please.",
    "Concise answer only.", "Answer briefly.", "Be succinct.",
    "Answer concisely.", "Answer with a single word or phrase.", "Reply with the answer only.")}


def strip_format_clause(q: str) -> str:
    """Remove trailing brevity/format clauses that a new format clause would contradict."""
    prev = None
    q = (q or "").rstrip()
    while prev != q:
        prev = q
        head, sep, last = q.rpartition("\n")
        if sep and last.strip().lower() in CAULDRON_SUFFIXES:
            q = head.rstrip()
            continue
        q = _FORMAT_CLAUSE.sub("", q).rstrip()
    return q


def parse_options(q: str) -> list[str]:
    """'Options: 1) Ovums; 2) Eggs' / 'Options: c | D | b' -> option texts."""
    m = _OPTIONS.search(q or "")
    if not m:
        return []
    body = m.group(1).strip()
    if re.search(r"\b\d+\)\s*", body):
        return [x.strip(" ;") for x in re.split(r"\s*\d+\)\s*", body) if x.strip(" ;")]
    return [x.strip() for x in body.split("|") if x.strip()]


# --------------------------------------------------------------------------- conditions
_TRANSCRIBE = re.compile(r"\b(transcribe|transcription|convert .* to (latex|html|markdown)|latex|"
                         r"what is written|read (all )?the text|parse the (page|document))\b", re.IGNORECASE)
_ABSTAIN = re.compile(r"^\s*(<no answer>|unanswerable|not (stated|given|available|present)|n/?a|none)\s*\.?\s*$", re.IGNORECASE)


def conditions(question: str, answer: str, reasoning: str, box: Any, task: str = "") -> set[str]:
    a = strip_period(answer)
    out = {"always"}
    if task in ("recognition", "table", "localization") or _TRANSCRIBE.search(question or ""):
        out.add("transcription")
    if _ABSTAIN.match(a or ""):
        out.add("abstain")
    letters = re.findall(r"[^\W\d_]", question or "")
    if letters and sum(c.isascii() for c in letters) / len(letters) > 0.9:
        out.add("lang_en")
    if is_numeric(a):
        out.add("numeric")
    if a.lower() in _YES_NO:
        out.add("yes_no")
    if len(a.split()) <= 6 and "\n" not in a:
        out.add("short_answer")
    if len(a.split()) > 12:
        out.add("long_answer")
    opts = parse_options(question)
    if opts and any(o.strip().lower() == a.lower() for o in opts):
        out.add("has_options")
    if (reasoning or "").strip():
        out.add("has_reasoning")
    if box:
        out.add("has_box")
    if a.startswith(("{", "<")):
        out.add("structured")
    return out


def applicable(fmt: FormatSpec, conds: set[str]) -> bool:
    if conds & {"transcription", "abstain"} and fmt.render != "reasoned":
        return False                    # transcripts / abstain golds keep their own target form
    if "structured" in conds and fmt.render not in ("answer", "reasoned"):
        return False                                   # never re-render JSON / HTML targets
    return all(c in conds for c in fmt.when)


# --------------------------------------------------------------------------- rendering
def render(fmt: FormatSpec, question: str, answer: str, reasoning: str, box) -> list[str] | None:
    a = strip_period(answer)
    if fmt.render == "answer":
        return answer_variants(answer)
    if fmt.render == "number":
        v = parse_number(a)
        return None if v is None else list(dict.fromkeys([f"{v:g}", f"{v:,.2f}".rstrip("0").rstrip("."),
                                                           strip_period(answer)]))
    if fmt.render == "yes_no":
        return [a.lower(), a.capitalize()]
    if fmt.render == "sentence":
        return [f"The answer is {a}.", f"The answer is {a}"]
    if fmt.render == "json":
        return [json.dumps({"answer": a}, ensure_ascii=False)]
    if fmt.render == "mcq_index":
        opts = parse_options(question)
        idx = [i for i, o in enumerate(opts, 1) if o.strip().lower() == a.lower()]
        return [str(idx[0])] if len(idx) == 1 else None
    if fmt.render == "reasoned":
        return [f"Reasoning: {reasoning.strip()}\nAnswer: {a}"]
    if fmt.render == "grounded":
        b = [round(v * 1000) for v in box[:4]]
        return [f"{a} | {b[0]},{b[1]},{b[2]},{b[3]}"]
    raise ValueError(f"unknown render {fmt.render!r}")


# --------------------------------------------------------------------------- sampling
def _u(*parts) -> float:
    """Deterministic uniform [0,1) from the parts."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def _pick(items: list, *key) -> Any:
    return items[int(_u(*key) * len(items)) % len(items)]


def _weighted(fmts: list[FormatSpec], k: int, *key) -> list[FormatSpec]:
    """k distinct formats, weighted, without replacement (Efraimidis-Spirakis keys)."""
    scored = sorted(fmts, key=lambda f: -(_u(*key, f.name) ** (1.0 / max(f.weight, 1e-6))))
    return scored[:k]


def attach_clause(q: str, clause: str, cfg: DiversityConfig, *key) -> str:
    sep = _pick(cfg.clause_separator, *key, "sep")
    prefix_p = cfg.clause_position.get("prefix", 0.0) / max(sum(cfg.clause_position.values()), 1e-9)
    if _u(*key, "pos") < prefix_p and not _OPTIONS.search(q):
        return f"{clause}{sep}{q}"
    return f"{q}{sep}{clause}"


@dataclass
class Variant:
    sample_id: str
    instruction: str
    answers: list[str]
    format: str
    metric: str
    final_answers: list[str]
    meta: dict = field(default_factory=dict)


def diversify_qa(sample_id: str, source: str, i: int, question: str, answer: str,
                 variants: list[str], reasoning: str, metric: str, task_detail: str,
                 cfg: DiversityConfig, box=None, task: str = "") -> list[Variant]:
    cfg = cfg.for_source(source)
    finals = list(dict.fromkeys(answer_variants(answer, cfg) + [v for v in variants if v]))
    out: list[Variant] = []
    if not finals:                     # no text target (e.g. box-only layout rows): nothing to vary
        return out
    base_q = question
    derived = (task_detail or "").startswith("derived:")
    if derived:
        pool = cfg.derived_question_templates.get(task_detail.split(":", 1)[1])
        if pool:
            base_q = _pick(pool + [question], cfg.seed, sample_id, i, "q")
    if cfg.keep_original:
        out.append(Variant(f"{sample_id}_q{i}_v0", question, finals, "original", metric, finals,
                           {"source": source, "qa_index": i}))
    stripped = strip_format_clause(base_q) if cfg.strip_source_format_clauses else base_q
    conds = conditions(question, answer, reasoning, box, task)
    fmts = [f for f in cfg.formats if applicable(f, conds)]
    # walk formats in weighted order; prefer ones whose TARGET differs from targets already made
    # (two variants that only differ in clause wording teach less than a new answer form)
    seen_targets = {finals[0]} if cfg.keep_original else set()
    ranked = _weighted(fmts, len(fmts), cfg.seed, sample_id, i)
    chosen, dup = [], []
    for f in ranked:
        golds = render(f, question, answer, reasoning, box)
        if not golds:
            continue
        (dup if golds[0] in seen_targets else chosen).append((f, golds))
        seen_targets.add(golds[0])
    # guarantee one new target form, then follow the weights (a different clause for the same
    # target is still instruction diversity)
    head = chosen[:1]
    picked = (head + [x for x in sorted(chosen[1:] + dup, key=lambda t: ranked.index(t[0]))])[:cfg.variants_per_qa]
    for j, (f, golds) in enumerate(picked):
        clause = _pick(f.clauses, cfg.seed, sample_id, i, f.name)
        instr = attach_clause(stripped, clause, cfg, cfg.seed, sample_id, i, f.name)
        out.append(Variant(f"{sample_id}_q{i}_v{j + 1}", instr, golds, f.name,
                           metric if f.metric == "source" else f.metric, finals,
                           {"source": source, "qa_index": i, "clause": clause,
                            "conditions": sorted(conds)}))
    return out


def diversify_row(row: dict, cfg: DiversityConfig) -> list[Variant]:
    """All variants for one UDD row (after :func:`rationale.attach`)."""
    out = []
    n = len(row["instructions"])
    reas = list(row.get("reasoning") or []) + [""] * n
    vars_ = list(row.get("answer_variants") or []) + [[]] * n
    tds = list(row.get("task_detail") or []) + [""] * n
    boxes = list(row.get("evidence_box") or []) + [None] * n
    tasks = list(row.get("qa_tasks") or []) + [row.get("task") or ""] * n
    for i in range(n):
        out += diversify_qa(row["sample_id"], row["source"], i, row["instructions"][i],
                            row["answers"][i], vars_[i], reas[i] or "", row.get("metric") or "anls",
                            tds[i], cfg, box=boxes[i], task=tasks[i])
    return out


# --------------------------------------------------------------------------- measurement
def _template(instr: str) -> str:
    """Instruction skeleton for diversity counting: digits/quoted spans masked, lower-cased."""
    s = re.sub(r"'[^']*'|\"[^\"]*\"", "<Q>", instr.lower())
    return re.sub(r"\d+", "<N>", s)


def _entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values() if v) if n else 0.0


def diversity_stats(pairs: list[tuple[str, str, list[str]]]) -> dict:
    """pairs = (instruction, format, answers). Measures what a model sees as varying."""
    if not pairs:
        return {}
    fmt = Counter(f for _, f, _ in pairs)
    first3 = Counter(" ".join(i.lower().split()[:3]) for i, _, _ in pairs)
    clause = Counter(i.strip().splitlines()[-1].split("|")[-1].strip().lower()[-60:] for i, _, _ in pairs)
    return {
        "n": len(pairs),
        "distinct_instruction_templates": len({_template(i) for i, _, _ in pairs}),
        "distinct_first3": len(first3),
        "first3_entropy_bits": round(_entropy(first3), 3),
        "distinct_tail_clauses": len(clause),
        "tail_clause_entropy_bits": round(_entropy(clause), 3),
        "format_distribution": dict(fmt.most_common()),
        "format_entropy_bits": round(_entropy(fmt), 3),
        "mean_answer_variants": round(sum(len(a) for _, _, a in pairs) / len(pairs), 3),
        # short golds (<= 4 words) whose primary form still ends in a stray period ("Mean.")
        # (sentence-format targets end in a period by design and are not counted)
        "short_answer_with_trailing_period": sum(
            1 for _, f, a in pairs if a and f != "sentence" and len(a[0].split()) <= 4
            and strip_period(a[0]) != a[0].strip()),
    }

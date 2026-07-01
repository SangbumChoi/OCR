"""A single, powerful **unified loader** for every document/OCR benchmark.

`trainset.py` already flattens each benchmark into the training `Sample` (question/answers). This
module goes further: it normalises every dataset into **one task-typed record** — :class:`UnifiedSample`
— that *preserves the structured payload* (KIE fields, localization boxes, table HTML, full text), so
downstream code can **freely filter, merge, and re-task** across heterogeneous sources:

    loader = UnifiedLoader()
    rows   = loader.load("cord", limit=50)                 # one benchmark
    allrows= loader.load_all(limit_per=30)                 # every streamable benchmark
    kie    = [r for r in rows if r.task == Task.KIE]        # filter by task
    boxes  = [f for r in allrows for f in r.fields if f.bbox]   # merge localized fields

Every record carries a **task** (recognition / kie / vqa / localization / table / reasoning) and the
fields that task needs. `UnifiedSample.to_sample()` collapses it back to the flat training `Sample`,
so the unified layer is a *superset* of `trainset.py`, not a replacement.

Design: per-benchmark **typed extractors** in `_UNIFIED` (keyed by catalog `key`); unregistered
benchmarks fall back to the `trainset.extract_qa` adapter wrapped as VQA/recognition records. Adding a
benchmark = one function. All extraction is pure (no network/image I/O) so it is unit-testable offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from ..benchmarks.trainset import _as_answer_list, _first, _s, extract_qa, norm_metric


# --------------------------------------------------------------------------- task taxonomy
class Task:
    """The document-understanding task a record represents (what you'd filter / merge on)."""
    RECOGNITION = "recognition"     # read all text (full page / line / formula)
    KIE = "kie"                     # key-value field extraction (forms, receipts)
    VQA = "vqa"                     # answer a question about the image
    LOCALIZATION = "localization"   # spotting / grounding — boxes for elements
    TABLE = "table"                 # table structure + cell content
    REASONING = "reasoning"         # multi-step / relational / chart-math
    CLASSIFICATION = "classification"

    ALL = (RECOGNITION, KIE, VQA, LOCALIZATION, TABLE, REASONING, CLASSIFICATION)


# default task per benchmark key (catalog) — extractors may emit a more specific one per record
TASK_BY_BENCHMARK = {
    "iam": Task.RECOGNITION, "recognition_fullpage": Task.RECOGNITION, "iiit5k": Task.RECOGNITION,
    "sroie": Task.KIE, "cord": Task.KIE, "funsd": Task.KIE, "xfund": Task.KIE,
    "wildreceipt": Task.KIE, "docile": Task.KIE,
    "docvqa": Task.VQA, "infovqa": Task.VQA, "textvqa": Task.VQA, "stvqa": Task.VQA,
    "ocrvqa": Task.VQA, "ai2d": Task.VQA, "visualmrc": Task.VQA, "ocrbench": Task.VQA,
    "ocrbench_v2": Task.VQA, "pope": Task.VQA, "hallusionbench": Task.VQA,
    "chartqa": Task.REASONING, "mathvista": Task.REASONING, "plotqa": Task.REASONING,
    "dvqa": Task.REASONING, "charxiv": Task.REASONING,
    "pubtabnet": Task.TABLE, "pubtables1m": Task.LOCALIZATION, "fintabnet": Task.TABLE,
    "scitsr": Task.TABLE,
    "im2latex": Task.RECOGNITION, "latexocr": Task.RECOGNITION, "crohme": Task.RECOGNITION,
    "doclaynet": Task.LOCALIZATION, "omnidocbench": Task.RECOGNITION,
}


# --------------------------------------------------------------------------- structured payload
@dataclass
class Box:
    """Axis-aligned box. ``normalized`` = coords in [0,1] (else pixel)."""
    x1: float
    y1: float
    x2: float
    y2: float
    normalized: bool = False

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Field:
    """A key-value pair (KIE / merge), optionally localized."""
    key: str
    value: str
    bbox: Box | None = None


@dataclass
class Region:
    """A localized element (spotting / detection): a box plus optional text/label."""
    label: str
    bbox: Box | None = None
    text: str = ""


@dataclass
class QA:
    """One question + its gold answer(s). Lets a single image carry MANY QAs (OCR-VQA-style)."""
    question: str
    answers: list[str] = field(default_factory=list)


@dataclass
class UnifiedSample:
    """One benchmark item in the unified, task-typed format."""
    sample_id: str
    source: str                       # catalog benchmark key
    task: str                         # Task.*
    instruction: str = ""             # the prompt / question (empty for pure recognition w/ canned prompt)
    answers: list[str] = field(default_factory=list)
    qas: list[QA] = field(default_factory=list)              # MANY QAs on one image (see merge_by_image)
    fields: list[Field] = field(default_factory=list)        # KIE
    regions: list[Region] = field(default_factory=list)      # localization
    full_text: str | None = None      # recognition / parsing target
    table_html: str | None = None     # table
    language: str | None = None
    metric: str = "anls"
    image_path: str | None = None
    # --- provenance / origin (kept as columns so a single merged dataset stays traceable) ---
    hf_id: str | None = None
    split: str | None = None          # the source dataset's split (e.g. test / validation / train)
    hf_config: str | None = None      # the source dataset's HF config, if any
    meta: dict = field(default_factory=dict)

    # -------- conversions
    _DEFAULT_PROMPT = {
        Task.RECOGNITION: "Transcribe all the text in the image. Answer with the text only.",
        Task.TABLE: "Convert the table in the image to HTML.",
        Task.KIE: "Extract the key fields from the document as JSON.",
        Task.LOCALIZATION: "List the text elements in the image with their bounding boxes.",
    }

    def prompt(self) -> str:
        return self.instruction or self._DEFAULT_PROMPT.get(self.task, "Answer the question.")

    def to_sample(self):
        """Collapse to the flat training :class:`~docvlm_eval.schema.Sample` (or None if no target)."""
        from ..schema import Sample
        ans = list(self.answers)
        if not ans:                                   # derive a target from the structured payload
            if self.full_text:
                ans = [self.full_text]
            elif self.table_html:
                ans = [self.table_html]
            elif self.fields:
                ans = [json.dumps({f.key: f.value for f in self.fields}, ensure_ascii=False)]
        if not (self.image_path and ans and _s(ans[0])):
            return None
        return Sample(
            sample_id=self.sample_id, image_path=self.image_path, question=self.prompt(),
            answers=[_s(a) for a in ans if _s(a)], answer_type=self.task, metric=self.metric,
            meta={"source": self.source, "hf_id": self.hf_id, "task": self.task,
                  "n_fields": len(self.fields), "n_regions": len(self.regions), **self.meta},
        )

    def to_samples(self) -> list:
        """Expand to flat training :class:`~docvlm_eval.schema.Sample`(s).

        A record with a ``qas`` list (many questions on one image — see :func:`merge_by_image`) yields
        **one Sample per QA**, all sharing the same ``image_path`` (so the image is cached/loaded once).
        A record without ``qas`` yields at most one Sample (delegates to :meth:`to_sample`)."""
        from ..schema import Sample
        if not self.qas:
            s = self.to_sample()
            return [s] if s is not None else []
        out = []
        for i, qa in enumerate(self.qas):
            ans = [_s(a) for a in qa.answers if _s(a)]
            if not (self.image_path and qa.question and ans):
                continue
            out.append(Sample(
                sample_id=f"{self.sample_id}_q{i}", image_path=self.image_path,
                question=qa.question, answers=ans, answer_type=self.task, metric=self.metric,
                meta={"source": self.source, "hf_id": self.hf_id, "task": self.task,
                      "n_qas": len(self.qas), **self.meta}))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten boxes to lists for JSON friendliness
        for fl in d["fields"]:
            if fl.get("bbox"):
                fl["bbox"] = [fl["bbox"]["x1"], fl["bbox"]["y1"], fl["bbox"]["x2"], fl["bbox"]["y2"],
                              fl["bbox"]["normalized"]]
        for rg in d["regions"]:
            if rg.get("bbox"):
                rg["bbox"] = [rg["bbox"]["x1"], rg["bbox"]["y1"], rg["bbox"]["x2"], rg["bbox"]["y2"],
                              rg["bbox"]["normalized"]]
        return d


# --------------------------------------------------------------------------- box/field extractors
def _norm_box4(x1, y1, x2, y2, normalized) -> Box | None:
    try:
        return Box(float(x1), float(y1), float(x2), float(y2), bool(normalized))
    except (TypeError, ValueError):
        return None


def _regions_from_ocrvqa(ex: dict) -> list[Region]:
    """OCR-VQA `ocr_info`: per-word normalized boxes {top_left_x, top_left_y, width, height}."""
    out: list[Region] = []
    for o in ex.get("ocr_info") or []:
        if not isinstance(o, dict):
            continue
        bb = o.get("bounding_box") or {}
        x, y = bb.get("top_left_x"), bb.get("top_left_y")
        w, h = bb.get("width"), bb.get("height")
        if None not in (x, y, w, h):
            box = _norm_box4(x, y, x + w, y + h, True)
            out.append(Region(label="word", bbox=box, text=_s(o.get("word"))))
    return out


def _quad_to_box(quad: dict) -> Box | None:
    """CORD `quad` {x1..x4,y1..y4} -> enclosing axis-aligned pixel box."""
    xs = [quad.get(f"x{i}") for i in (1, 2, 3, 4)]
    ys = [quad.get(f"y{i}") for i in (1, 2, 3, 4)]
    xs = [v for v in xs if isinstance(v, (int, float))]
    ys = [v for v in ys if isinstance(v, (int, float))]
    if len(xs) >= 2 and len(ys) >= 2:
        return _norm_box4(min(xs), min(ys), max(xs), max(ys), False)
    return None


def _flatten(d: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested gt_parse dict into ('a.b', 'value') pairs (lists expanded)."""
    out: list[tuple[str, str]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        for v in d:
            out += _flatten(v, prefix)
    else:
        s = _s(d)
        if s:
            out.append((prefix, s))
    return out


# --------------------------------------------------------------------------- typed extractors
# Registry of per-benchmark extractors. **Extension point:** to support a new dataset, write a
# function ``(ex: dict, entry: dict) -> list[UnifiedSample]`` and decorate it with
# ``@register("<catalog_key>")`` — nothing else changes. Unregistered benchmarks fall back to the
# flat trainset adapter (wrapped as VQA/recognition by TASK_BY_BENCHMARK).
_UNIFIED: dict[str, Any] = {}


def register(*keys: str):
    """Decorator: register an extractor for one or more catalog keys (the easy extension hook)."""
    def deco(fn):
        for k in keys:
            _UNIFIED[k] = fn
        return fn
    return deco


@register("cord")
def _u_cord(ex, e) -> list[UnifiedSample]:
    gt = ex.get("ground_truth")
    parse = None
    if isinstance(gt, str):
        try:
            j = json.loads(gt); parse = j.get("gt_parse"); valid = j.get("valid_line") or []
        except Exception:
            parse, valid = None, []
    else:
        parse = (gt or {}).get("gt_parse", gt) if isinstance(gt, dict) else None
        valid = (gt or {}).get("valid_line", []) if isinstance(gt, dict) else []
    fields = [Field(key=k, value=v) for k, v in _flatten(parse or {})]
    # attach boxes from valid_line words (localization within KIE)
    for line in (valid or []):
        if not isinstance(line, dict):
            continue
        cat = _s(line.get("category"))
        for w in line.get("words") or []:
            if isinstance(w, dict):
                fields.append(Field(key=cat or "word", value=_s(w.get("text")),
                                    bbox=_quad_to_box(w.get("quad") or {})))
    answers = [json.dumps({f.key: f.value for f in fields if f.bbox is None}, ensure_ascii=False)] \
        if fields else []
    return [UnifiedSample(sample_id="", source=e["key"], task=Task.KIE,
                          instruction="Extract the receipt's key fields as JSON.",
                          answers=answers, fields=fields, metric="anls")]


@register("funsd")
def _u_funsd(ex, e) -> list[UnifiedSample]:
    words = ex.get("words") or ex.get("tokens") or []
    bboxes = ex.get("bboxes") or []
    tags = ex.get("ner_tags") or []
    _TAG = {0: "O", 1: "B-HEADER", 2: "I-HEADER", 3: "B-QUESTION", 4: "I-QUESTION",
            5: "B-ANSWER", 6: "I-ANSWER"}
    fields: list[Field] = []
    for i, w in enumerate(words):
        b = bboxes[i] if i < len(bboxes) and isinstance(bboxes[i], (list, tuple)) and len(bboxes[i]) >= 4 else None
        lbl = _TAG.get(tags[i], "O") if i < len(tags) else "word"
        # FUNSD/LayoutLMv3 boxes are normalized to 0–1000 → divide to true [0,1]
        nb = None
        if b:
            try:
                nb = _norm_box4(b[0] / 1000, b[1] / 1000, b[2] / 1000, b[3] / 1000, True)
            except (TypeError, ValueError, IndexError):
                nb = None
        fields.append(Field(key=lbl, value=_s(w), bbox=nb))
    full = " ".join(_s(w) for w in words)
    return [UnifiedSample(sample_id="", source=e["key"], task=Task.KIE,
                          instruction="Transcribe all the text in the form.",
                          answers=[full] if full else [], fields=fields, full_text=full or None,
                          metric="ned")]


@register("pubtabnet", "fintabnet")
def _u_table(ex, e) -> list[UnifiedSample]:
    html = _s(_first(ex, ("html_table", "html", "table", "text", "label")))
    return [UnifiedSample(sample_id="", source=e["key"], task=Task.TABLE,
                          instruction="Convert the table in the image to HTML.",
                          answers=[html] if html else [], table_html=html or None, metric="anls")]


@register("ocrvqa")
def _u_ocrvqa(ex, e) -> list[UnifiedSample]:
    regions = _regions_from_ocrvqa(ex)
    out: list[UnifiedSample] = []
    qs = ex.get("questions") or ex.get("question"); ans = ex.get("answers") or ex.get("answer")
    qs = qs if isinstance(qs, list) else [qs]; ans = ans if isinstance(ans, list) else [ans]
    for q, a in zip(qs, ans):
        if _s(q) and _as_answer_list(a):
            out.append(UnifiedSample(sample_id="", source=e["key"], task=Task.VQA,
                                     instruction=_s(q), answers=_as_answer_list(a),
                                     regions=regions, metric="exact"))
    return out


_DOCLAYNET = {1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer",
              6: "Page-header", 7: "Picture", 8: "Section-header", 9: "Table", 10: "Text", 11: "Title"}


@register("doclaynet")
def _u_doclaynet(ex, e) -> list[UnifiedSample]:
    """DocLayNet layout detection: COCO [x,y,w,h] boxes per element → localization regions
    (normalized to [0,1] by the page's coco_width/height so downscaling keeps them valid)."""
    md = ex.get("metadata") or {}
    W, H = md.get("coco_width") or 0, md.get("coco_height") or 0
    boxes, cats = ex.get("bboxes") or [], ex.get("category_id") or []
    regions = []
    for i, b in enumerate(boxes):
        if not (isinstance(b, (list, tuple)) and len(b) >= 4):
            continue
        x, y, w, h = b[:4]
        cid = cats[i] if i < len(cats) else None
        box = Box(x / W, y / H, (x + w) / W, (y + h) / H, True) if (W and H) \
            else Box(x, y, x + w, y + h, False)
        regions.append(Region(label=_DOCLAYNET.get(cid, str(cid)), bbox=box))
    return [UnifiedSample(sample_id="", source=e["key"], task=Task.LOCALIZATION,
                          instruction="Localize the document layout elements as bounding boxes.",
                          regions=regions, metric="grounding")]


def _via_trainset(ex, e, task: str) -> list[UnifiedSample]:
    """Fallback: reuse the flat trainset adapter, wrap each QA as a typed UnifiedSample."""
    out = []
    for qa in extract_qa(e["key"], ex, e):
        out.append(UnifiedSample(
            sample_id="", source=e["key"], task=task, instruction=qa["question"],
            answers=qa["answers"], metric=qa.get("metric", "anls"),
            meta={"answer_type": qa.get("answer_type")}))
    return out


def extract_unified(key: str, ex: dict, entry: dict | None = None) -> list[UnifiedSample]:
    """Map one raw benchmark example to typed :class:`UnifiedSample`(s). Returns [] if nothing usable."""
    entry = entry or {"key": key}
    try:
        if key in _UNIFIED:
            recs = _UNIFIED[key](ex, entry)
        else:
            recs = _via_trainset(ex, entry, TASK_BY_BENCHMARK.get(key, Task.VQA))
    except Exception:
        return []
    # carry catalog metric/hf_id defaults where the extractor didn't set them; derive language from
    # the record's own text (script heuristic + source prior) so the column isn't left empty
    from .enrich import detect_language

    for r in recs:
        r.hf_id = entry.get("hf_id")
        if not r.metric:
            r.metric = norm_metric(entry.get("metric"))
        if not r.language:
            text = " ".join(filter(None, [r.full_text, *(rg.text for rg in r.regions),
                                          *(f.value for f in r.fields), *r.answers]))
            r.language = (detect_language(text, r.source)
                          or detect_language(r.instruction, r.source))
    return [r for r in recs if r.answers or r.fields or r.regions or r.full_text or r.table_html]


# --------------------------------------------------------------------------- the loader
class UnifiedLoader:
    """Stream any catalog benchmark (or all) and yield :class:`UnifiedSample`s.

    Pure-Python control flow; the only heavy deps (``datasets``, ``PIL``) are imported lazily inside
    the streaming methods so importing this module (and the extractors) stays cheap and offline-safe.
    """

    def __init__(self, catalog: list[dict] | None = None):
        from ..benchmarks.catalog import load_catalog
        self.catalog = catalog or load_catalog()
        self.by_key = {e["key"]: e for e in self.catalog}

    def streamable_keys(self) -> list[str]:
        return [e["key"] for e in self.catalog if e.get("hf_id")]

    def iter(self, key: str, *, limit: int = 50, max_scan: int = 3000, max_px: int = 1000,
             quality: int = 85, cache_dir: str | None = None,
             global_index: dict | None = None) -> Iterator[UnifiedSample]:
        """Yield up to ``limit`` distinct-image UnifiedSamples for one benchmark.

        ``global_index`` is a persistent cross-run/cross-source dedup cache (md5 → owner key, see
        ``scripts/build_udd.py``): an image already owned by a *different* source is skipped (COCO
        images recur across scene-text sets), while hashes owned by *this* key don't block a rebuild.
        New hashes are recorded in the dict in place — the caller persists it."""
        import hashlib
        from pathlib import Path

        from datasets import load_dataset

        from ..benchmarks.catalog import find_image

        e = self.by_key.get(key)
        if not e or not e.get("hf_id"):
            return
        if key in _SPECIAL_LOADERS:      # datasets that need bespoke loading (json+images, etc.)
            yield from _SPECIAL_LOADERS[key](e, limit=limit, max_px=max_px, cache_dir=cache_dir)
            return
        try:
            ds = load_dataset(e["hf_id"], e.get("config"), split=e["split"], streaming=True)
        except Exception as exc:
            print(f"[unified][fail] {key}: {type(exc).__name__}: {str(exc)[:120]}")
            return

        cache = Path(cache_dir) / key if cache_dir else None
        seen: set[str] = set()
        n_img = 0
        for scanned, ex in enumerate(ds):
            if n_img >= limit or scanned >= max_scan:
                break
            ex = dict(ex)
            img = find_image(ex)
            if img is None:
                continue
            recs = extract_unified(key, ex, e)
            if not recs:
                continue
            small = img.convert("RGB")
            if max(small.size) > max_px:
                s = max_px / max(small.size)
                small = small.resize((max(1, round(small.width * s)), max(1, round(small.height * s))))
            h = hashlib.md5(small.tobytes()).hexdigest()
            owner = (global_index or {}).get(h)
            if h in seen or (owner is not None and owner != key):
                continue
            seen.add(h)
            if global_index is not None:
                global_index[h] = key
            img_path = None
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                img_path = str(cache / f"{n_img:04d}.jpg")
                small.save(img_path, quality=quality)
            for i, r in enumerate(recs):
                r.sample_id = f"{key}_{n_img:04d}_{i}"
                r.image_path = img_path
                r.split = e.get("split")           # origin: keep source split + config as columns
                r.hf_config = e.get("config")
                yield r
            n_img += 1

    def load(self, key: str, **kw) -> list[UnifiedSample]:
        return list(self.iter(key, **kw))

    def load_all(self, *, limit_per: int = 50, only: list[str] | None = None,
                 skip: list[str] | None = None, **kw) -> dict[str, list[UnifiedSample]]:
        keys = [k for k in self.streamable_keys()
                if (only is None or k in only) and (skip is None or k not in skip)]
        out: dict[str, list[UnifiedSample]] = {}
        for k in keys:
            rows = self.load(k, limit=limit_per, **kw)
            if rows:
                out[k] = rows
                print(f"[unified][ok] {k}: {len(rows)} records ({rows[0].task})")
        return out


# --------------------------------------------------------------------------- convenience
def to_training_samples(rows: list[UnifiedSample]) -> list:
    """Collapse unified rows to flat training Samples (dropping those with no usable target).

    Records that carry a ``qas`` list (from :func:`merge_by_image`) expand to one Sample per QA."""
    return [s for r in rows for s in r.to_samples()]


def merge_by_image(rows: list[UnifiedSample], *,
                   qa_tasks: tuple[str, ...] = (Task.VQA, Task.REASONING)
                   ) -> list[UnifiedSample]:
    """Merge records that share the same image into ONE record carrying a **list of Q/A**.

    Many benchmarks (OCR-VQA, DocVQA, …) repeat the *same image* with *different questions*; streamed
    naively that is one :class:`UnifiedSample` per question. This groups them by image so each image
    appears once, with every question collected into ``qas: list[QA]`` (deduped, order-preserving).
    Non-QA tasks (kie/table/recognition/localization) are grouped too — their structured payload
    (fields/regions/full_text/table_html) is unioned — but they keep an empty ``qas`` unless they also
    carry a question.

    Grouping key = ``image_path`` when present, else ``(source, sample_id-without-QA-suffix)`` so it
    still merges pre-cache (in-memory) rows. Downstream, :meth:`UnifiedSample.to_samples` /
    :func:`to_training_samples` re-expand ``qas`` into one training Sample per question — so merging is
    a *lossless regrouping*: fewer rows, identical training set, and one image decode per group.
    """
    groups: dict[Any, list[UnifiedSample]] = {}
    order: list[Any] = []
    for r in rows:
        key = r.image_path or (r.source, r.sample_id.rsplit("_", 1)[0])
        if key not in groups:
            groups[key] = []; order.append(key)
        groups[key].append(r)

    merged: list[UnifiedSample] = []
    for key in order:
        grp = groups[key]
        base = grp[0]
        qas: list[QA] = []
        seen_q: set[str] = set()
        fields: list[Field] = []
        regions: list[Region] = []
        full_text = None
        table_html = None
        for r in grp:
            # collect this record's question(s): its own qas, then its instruction/answers pair
            pairs = list(r.qas)
            if r.instruction and r.answers:
                pairs.append(QA(question=r.instruction, answers=list(r.answers)))
            for qa in pairs:
                sig = qa.question.strip()
                if r.task in qa_tasks and sig and sig not in seen_q:
                    seen_q.add(sig); qas.append(qa)
            fields += r.fields
            regions += r.regions
            full_text = full_text or r.full_text
            table_html = table_html or r.table_html
        merged.append(UnifiedSample(
            sample_id=base.sample_id, source=base.source, task=base.task,
            instruction="" if qas else base.instruction,
            answers=[] if qas else list(base.answers),
            qas=qas, fields=fields, regions=regions, full_text=full_text, table_html=table_html,
            language=base.language, metric=base.metric, image_path=base.image_path,
            hf_id=base.hf_id, split=base.split, hf_config=base.hf_config, meta=dict(base.meta)))
    return merged


# --------------------------------------------------------------------------- special loaders
# Some datasets can't be streamed record-by-record with paired image+GT (annotations in a separate
# file, images in a separate archive, ...). Register a bespoke loader here; UnifiedLoader.iter
# delegates to it. Signature: ``(entry, *, limit, max_px, cache_dir) -> Iterator[UnifiedSample]``.
_SPECIAL_LOADERS: dict = {}


def load_omnidocbench(entry, *, limit=50, max_px=1000, cache_dir=None):
    """OmniDocBench: annotations live in one ``OmniDocBench.json`` and images under ``images/`` — join
    them here. Emits recognition records (full-page text in reading order) + localization ``regions``
    (per-block boxes, normalized to [0,1] by page size, so downscaling the image keeps them valid)."""
    import json as _json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo = entry["hf_id"]
    try:
        data = _json.load(open(hf_hub_download(repo, "OmniDocBench.json", repo_type="dataset")))
    except Exception as exc:
        print(f"[unified][fail] omnidocbench: {type(exc).__name__}: {str(exc)[:120]}"); return
    cache = Path(cache_dir) / "omnidocbench" if cache_dir else None
    n = 0
    for ent in data:
        if n >= limit:
            break
        pi = ent.get("page_info", {}) or {}
        ip, W, H = pi.get("image_path"), pi.get("width"), pi.get("height")
        if not ip or not W or not H:
            continue
        try:
            img = Image.open(hf_hub_download(repo, f"images/{ip}", repo_type="dataset")).convert("RGB")
        except Exception:
            continue
        dets = sorted((d for d in ent.get("layout_dets", []) if not d.get("ignore")),
                      key=lambda d: d.get("order") if d.get("order") is not None else 1_000_000)
        regions, texts = [], []
        for d in dets:
            poly, txt = d.get("poly"), _s(d.get("text"))
            box = None
            if poly and len(poly) >= 8:
                xs, ys = poly[0::2], poly[1::2]
                box = Box(min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H, True)
            regions.append(Region(label=_s(d.get("category_type")) or "block", bbox=box, text=txt))
            if txt:
                texts.append(txt)
        full = "\n".join(texts)
        small = img
        if max(small.size) > max_px:
            s = max_px / max(small.size)
            small = small.resize((max(1, round(small.width * s)), max(1, round(small.height * s))))
        img_path = None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            img_path = str(cache / f"{n:04d}.jpg")
            small.save(img_path, quality=85)
        yield UnifiedSample(
            sample_id=f"omnidocbench_{n:04d}_0", source="omnidocbench", task=Task.RECOGNITION,
            instruction="Transcribe the full page in reading order.",
            answers=[full] if full else [], full_text=full or None, regions=regions,
            image_path=img_path, hf_id=repo, split=entry.get("split"), hf_config=entry.get("config"),
            metric="ned")
        n += 1


_SPECIAL_LOADERS["omnidocbench"] = load_omnidocbench

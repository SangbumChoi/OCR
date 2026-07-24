"""Typed data-transfer objects for synthetic document generation.

This is the **single source of truth** that ties three things together:

  1. *what is drawn* on the page (fields, tables, reading order),
  2. *the ground truth* for every capability axis (boxes, rationales, languages), and
  3. *the ablation factors* the ablation study wants to vary (see
     ``docs/report/ablation_plan.md`` and ``configs/synth_data.yaml``).

The design rule: every value the ablations need to switch on/off is **stored in the GT** and
**controlled by a single :class:`GenConfig`**, so one config file fully determines a dataset
variant. The data-side ablation factors map onto the DTO as:

  * **A1 spotting**      -> :attr:`Field.bbox` and :attr:`QAItem.answer_bbox` (the *where* target);
                            emitted iff ``GenConfig.emit_spotting``.
  * **A2 reasoning**     -> :attr:`QAItem.rationale` (the CoT target);
                            emitted iff ``GenConfig.emit_rationale``.
  * **A3 spot+reason**   -> any combination of the two flags above (no new field; it is composition).
  * **A4 multilingual**  -> :attr:`Field.language` / :attr:`Field.script` and :attr:`DocSample.languages`;
                            driven by ``GenConfig.languages`` / ``language_weights``.
  * **A7 preprocessing** -> :class:`RenderSpec` (``dpi``, ``target_long_side``, ``keep_aspect``,
                            ``tiling``) plus :attr:`Field.font_px` / :attr:`Field.is_small` so a
                            *small-text* slice exists; driven by the render knobs in ``GenConfig``.
  * **Document marks**   -> grounded handwriting/stamp/seal QAs plus
                            :attr:`RenderSpec.overlays`; driven by ``GenConfig.overlay_*``.

(A5 LoRA-placement and A6 HPO are *training-side* — they consume this GT but need no generator knob.)

``DocSample.to_dict()`` is a **superset** of the legacy flat ``gt.json`` schema, so
:mod:`docvlm_eval.synth.to_samples` and existing readers keep working unchanged while the richer
structured view is available alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .hard_layout import HARD_LAYOUT_FAMILIES
from .overlays import OVERLAY_TYPES

# --- language -> writing system, used to label scripts for the A4 transfer matrix ----------
_SCRIPT_BY_LANG = {
    "en": "latin", "es": "latin", "fr": "latin", "de": "latin", "pt": "latin", "it": "latin",
    "ko": "hangul", "ja": "japanese", "zh": "han", "ar": "arabic", "he": "hebrew",
    "ru": "cyrillic", "hi": "devanagari", "th": "thai",
}


def script_for(lang: str) -> str:
    """Best-effort writing-system label for a language code (default latin)."""
    return _SCRIPT_BY_LANG.get((lang or "en").split("-")[0].lower(), "latin")


@dataclass
class BBox:
    """Pixel bounding box ``[x1, y1, x2, y2]`` in the rendered image's coordinate frame."""

    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    def normalized(self, w: int, h: int) -> list[float]:
        return [self.x1 / w, self.y1 / h, self.x2 / w, self.y2 / h]

    @classmethod
    def from_list(cls, b: list[int] | None) -> "BBox | None":
        return cls(int(b[0]), int(b[1]), int(b[2]), int(b[3])) if b and len(b) >= 4 else None


@dataclass
class Field:
    """One labelled value on the page, with everything the ablations slice on."""

    key: str
    value: str
    role: str = "value"            # title|kie-value|transcript|table-cell|ui-nav|caption|mrz|micr|footer
    bbox: BBox | None = None       # A1: the spotting target (None when off or not resolvable)
    language: str = "en"           # A4
    script: str = "latin"          # A4 (derived from language)
    font_px: float | None = None   # A7: rendered font size, when known
    is_small: bool = False         # A7: small-text slice flag (font_px <= GenConfig.small_text_px)
    reading_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = self.bbox.to_list() if self.bbox else None
        return d


@dataclass
class QAItem:
    """An answerable (question, answer) pair carrying the optional A1/A2 supervision targets."""

    question: str
    answers: list[str]
    answer_type: str = "kie"       # capability code (T1/T2/H1/H2/H3/L1/...) or a readable tag
    metric: str = "anls"
    rationale: str | None = None   # A2: chain-of-thought target (None when emit_rationale is off)
    answer_bbox: BBox | None = None  # A1: box of the answer span, when localisable
    evidence_keys: list[str] = field(default_factory=list)
    evidence_bboxes: list[BBox] = field(default_factory=list)
    languages: list[str] = field(default_factory=lambda: ["en"])
    key: str | None = None
    graph_query_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["answer_bbox"] = self.answer_bbox.to_list() if self.answer_bbox else None
        d["evidence_bboxes"] = [box.to_list() for box in self.evidence_bboxes]
        return d


@dataclass
class RenderSpec:
    """How the image was produced — the A7 (preprocessing/resize) knobs are recorded here."""

    source: str = "digital-native-pdf"   # digital-native-pdf | raster-pil | photo | screenshot
    dpi: int | None = 150
    size_px: list[int] = field(default_factory=lambda: [0, 0])
    page_size: str = "A5"
    page_count: int = 1
    target_long_side: int | None = None  # A7: longest side the image was resized to (None = native)
    keep_aspect: bool = True             # A7
    tiling: dict[str, Any] | None = None  # A7: {"n_max": int, ...} when dynamic tiling was simulated
    fonts: list[str] = field(default_factory=list)
    box_resolver: str = "pdf_text"
    color_probe_fallback_count: int = 0
    layout_family: str | None = None
    layout_fingerprint: str | None = None
    overlay_seed: int | None = None
    overlay_fingerprint: str | None = None
    overlays: list[dict[str, Any]] = field(default_factory=list)
    geometry: dict[str, Any] | None = None

    @property
    def aspect_ratio(self) -> float:
        w, h = (self.size_px + [0, 0])[:2]
        return round(w / h, 4) if h else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class Degradation:
    """Photometric degradation applied to the *copy* (geometry preserved → boxes stay valid)."""

    preset: str                          # scan|photo|fax|historical|screenshot
    severity: float = 1.0                # multiplier knob for distribution matching
    seed: int | None = None
    attempts: int = 1
    geometry_preserved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationSupport:
    """Which ablation factors this sample can actually exercise (computed from its content)."""

    spotting: bool = False        # A1: has at least one box
    rationale: bool = False       # A2: has at least one rationale
    multilingual: bool = False    # A4: more than one language present
    small_text: bool = False      # A7: has a small-text field
    table: bool = False
    abstain: bool = False         # H5: has a redacted/absent target
    reading_order: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ======================================================================================
# GenConfig — the one object a config.yaml binds to; its knobs flip the ablation factors.
# ======================================================================================
@dataclass
class GenConfig:
    """Generator knobs. Every variable an ablation controls lives here so each experiment is a
    single config file (``configs/synth_data.yaml`` + an optional named ``ablation`` override)."""

    name: str = "base"
    ablation: str | None = None
    seed: int = 7
    count: int = 1

    # --- A7 render / preprocessing ---
    dpi: int = 150
    target_long_side: int | None = None     # resize longest side to this (None = native render size)
    keep_aspect: bool = True
    tiling_n_max: int | None = None          # simulate dynamic tiling metadata (n_max tiles)
    small_text_px: float = 9.0               # fields at/below this px are the small-text slice

    # --- A1 / A2 supervision toggles (what GT to EMIT) ---
    emit_spotting: bool = True               # A1: include bbox targets
    emit_rationale: bool = True              # A2: include CoT rationales
    # model-free UNDERSTANDING layer: where/how-many/totals derived from the render (derive.py).
    # The OCR GT is free; this is the non-OCR understanding GT that needs no external model.
    emit_understanding: bool = True
    emit_counterfactual_pairs: bool = True
    color_probe_fallback: bool = True
    validate_evidence_pixels: bool = True
    evidence_min_contrast: float = 8.0
    evidence_min_foreground_fraction: float = 0.002
    evidence_min_foreground_pixels: int = 4
    validate_degraded_evidence: bool = True
    degraded_min_structure_correlation: float = 0.25
    degrade_max_attempts: int = 3
    perspective_prob: float = 0.35
    perspective_max_inset_fraction: float = 0.08
    perspective_min_area_ratio: float = 0.70
    overlay_prob: float = 0.35
    overlay_types: list[str] = field(default_factory=lambda: list(OVERLAY_TYPES))
    overlay_max_count: int = 2

    # --- visual diversity (per-doc paper colour / accent / font / margin jitter; geometry-safe) ---
    jitter: bool = False

    # --- textual source (future-optional seam; see docs/report/synth_generation_survey.md §4). ---
    # "offline" = Faker/locale + curated pools, NO network/LLM at generation time (the only mode wired
    # now). "corpus" (local real-text dump) and "llm" (explicit opt-in gateway) are reserved for future
    # use and raise NotImplementedError until a backend is enabled, so the default build stays LLM-free.
    text_source: str = "offline"

    # --- A4 multilingual ---
    languages: list[str] = field(default_factory=lambda: ["en"])
    language_weights: dict[str, float] | None = None

    # --- realism / distribution matching ---
    degrade_prob: float = 1.0                # fraction of docs that get a degraded copy
    degrade_presets: dict[str, list[str]] | None = None  # per doc-type allowed presets (overrides default)
    degrade_severity: float = 1.0
    doc_type_weights: dict[str, float] | None = None     # sampling weights when choosing doc types

    # --- hard-document curriculum and split provenance ---
    difficulty_level: int = 4
    hard_layout_families: list[str] = field(
        default_factory=lambda: list(HARD_LAYOUT_FAMILIES)
    )
    split_name: str = "synthetic"
    split_seed: int = 7
    split_group_by: str = "content"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def __post_init__(self) -> None:
        if not 1 <= self.difficulty_level <= 5:
            raise ValueError("difficulty_level must be within [1, 5]")
        if self.split_name not in {"synthetic", "train", "validation", "heldout"}:
            raise ValueError("split_name must be synthetic, train, validation, or heldout")
        if self.split_group_by not in {"content", "template", "layout", "document"}:
            raise ValueError(
                "split_group_by must be content, template, layout, or document"
            )
        if not self.hard_layout_families:
            raise ValueError("hard_layout_families cannot be empty")
        unknown_layouts = sorted(
            set(self.hard_layout_families) - set(HARD_LAYOUT_FAMILIES)
        )
        if unknown_layouts:
            raise ValueError(f"unknown hard layout families: {unknown_layouts}")
        if not isinstance(self.color_probe_fallback, bool):
            raise ValueError("color_probe_fallback must be boolean")
        if not isinstance(self.validate_evidence_pixels, bool):
            raise ValueError("validate_evidence_pixels must be boolean")
        if self.evidence_min_contrast <= 0:
            raise ValueError("evidence_min_contrast must be positive")
        if not 0 <= self.evidence_min_foreground_fraction <= 1:
            raise ValueError("evidence_min_foreground_fraction must be within [0, 1]")
        if self.evidence_min_foreground_pixels < 1:
            raise ValueError("evidence_min_foreground_pixels must be positive")
        if not isinstance(self.validate_degraded_evidence, bool):
            raise ValueError("validate_degraded_evidence must be boolean")
        if not 0 <= self.degraded_min_structure_correlation <= 1:
            raise ValueError("degraded_min_structure_correlation must be within [0, 1]")
        if self.degrade_max_attempts < 1:
            raise ValueError("degrade_max_attempts must be positive")
        if not 0 <= self.perspective_prob <= 1:
            raise ValueError("perspective_prob must be within [0, 1]")
        if not 0 < self.perspective_max_inset_fraction < 0.5:
            raise ValueError("perspective_max_inset_fraction must be within (0, 0.5)")
        if not 0 < self.perspective_min_area_ratio <= 1:
            raise ValueError("perspective_min_area_ratio must be within (0, 1]")
        if not 0 <= self.overlay_prob <= 1:
            raise ValueError("overlay_prob must be within [0, 1]")
        if not self.overlay_types:
            raise ValueError("overlay_types cannot be empty")
        unknown_overlays = sorted(set(self.overlay_types) - set(OVERLAY_TYPES))
        if unknown_overlays:
            raise ValueError(f"unknown document overlay types: {unknown_overlays}")
        if self.overlay_max_count < 1:
            raise ValueError("overlay_max_count must be positive")

    @classmethod
    def from_yaml(cls, path: str, ablation: str | None = None) -> "GenConfig":
        """Load the base config, then deep-merge ``ablation_overrides[<ablation>]`` if requested."""
        import yaml
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        base = dict(doc.get("base", doc))
        overrides = (doc.get("ablation_overrides") or {})
        if ablation:
            if ablation not in overrides:
                raise KeyError(f"ablation '{ablation}' not in {sorted(overrides)} of {path}")
            base.update(overrides[ablation] or {})
            base["ablation"] = ablation
            base.setdefault("name", ablation)
        return cls.from_dict(base)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocSample:
    """The full ground-truth record for one synthetic document.

    Serialised to ``gt.json`` via :meth:`to_dict`, which is a backward-compatible **superset** of
    the legacy flat schema (``type``/``fields``/``spotting``/``qa``/``table_html``/...).
    """

    doc_id: str
    doc_type: str
    stressors: list[str]
    anchor_metric: str
    fields: list[Field] = field(default_factory=list)
    qa: list[QAItem] = field(default_factory=list)
    table_html: str | None = None
    selection: dict[str, list[str]] = field(default_factory=dict)
    redacted: dict[str, str] = field(default_factory=dict)
    reading_order: Any = field(default_factory=list)
    probes: list[dict] = field(default_factory=list)
    render: RenderSpec = field(default_factory=RenderSpec)
    degradation: Degradation | None = None
    languages: list[str] = field(default_factory=lambda: ["en"])
    domain: str | None = None             # finance|identity|medical|web|legal|literary|...
    acquisition: str | None = None        # pdf-native|scan|photo|screenshot (modality split)
    source: str = "SYNTHETIC (docvlm_eval.synth) — renders the task; not official data"
    gen_config: dict[str, Any] | None = None
    semantic_graph: dict[str, Any] | None = None
    difficulty: dict[str, Any] | None = None
    split: str = "synthetic"

    # -------------------------------------------------------------------- support flags
    def support(self) -> AblationSupport:
        langs = {f.language for f in self.fields} | set(self.languages)
        return AblationSupport(
            spotting=any(f.bbox for f in self.fields) or any(q.answer_bbox for q in self.qa),
            rationale=any(q.rationale for q in self.qa),
            multilingual=len(langs) > 1,
            small_text=any(f.is_small for f in self.fields),
            table=self.table_html is not None,
            abstain=bool(self.redacted) or any(p.get("kind") == "abstain" for p in self.probes),
            reading_order=bool(self.reading_order),
        )

    # -------------------------------------------------------------------- builder adapter
    @classmethod
    def from_builder_gt(
        cls,
        gt: dict[str, Any],
        *,
        builder: Any = None,
        gen_config: "GenConfig | None" = None,
        degradation: "Degradation | None" = None,
        domain: str | None = None,
        acquisition: str | None = None,
    ) -> "DocSample":
        """Upgrade a legacy :class:`~docvlm_eval.synth.patterns.DocBuilder` ``gt`` dict into a
        structured :class:`DocSample`. Box/lang/role metadata is pulled from ``builder`` when given
        (``field_lang`` / ``field_role`` / ``field_font_px`` maps it records), else inferred."""
        spotting = gt.get("spotting", {}) or {}
        flds_meta_lang = getattr(builder, "field_lang", {}) if builder else {}
        flds_meta_role = getattr(builder, "field_role", {}) if builder else {}
        flds_meta_px = getattr(builder, "field_font_px", {}) if builder else {}
        small_px = gen_config.small_text_px if gen_config else 9.0

        fields: list[Field] = []
        for key, value in (gt.get("fields") or {}).items():
            if key.startswith("_"):
                continue
            lang = flds_meta_lang.get(key, "en")
            px = flds_meta_px.get(key)
            fields.append(Field(
                key=key, value=str(value), role=flds_meta_role.get(key, "value"),
                bbox=BBox.from_list(spotting.get(key)),
                language=lang, script=script_for(lang),
                font_px=px, is_small=(px is not None and px <= small_px),
            ))
        # boxes that have no matching field (e.g. table cells, ui buttons) still carry A1 signal
        for key, box in spotting.items():
            if key not in {f.key for f in fields}:
                fields.append(
                    Field(
                        key=key,
                        value="",
                        role="region",
                        bbox=BBox.from_list(box),
                        language=getattr(builder, "language", "en"),
                        script=script_for(getattr(builder, "language", "en")),
                    )
                )

        qa: list[QAItem] = []
        for q in gt.get("qa", []):
            # answer box: an explicit derived box (q["box"]) or the spotting box for its key
            abox = q.get("box") if q.get("box") else spotting.get(q.get("key"))
            qa.append(QAItem(
                question=q["question"], answers=list(q["answers"]),
                answer_type=q.get("answer_type", "kie"), metric=q.get("metric", "anls"),
                rationale=q.get("rationale"),
                answer_bbox=BBox.from_list(abox),
                evidence_keys=list(q.get("evidence_keys") or []),
                evidence_bboxes=[
                    box for key in (q.get("evidence_keys") or [])
                    if (box := BBox.from_list(spotting.get(key))) is not None
                ],
                languages=q.get(
                    "languages",
                    [getattr(builder, "language", "en")],
                ),
                key=q.get("key"),
                graph_query_id=q.get("graph_query_id"),
            ))

        rj = gt.get("render", {}) or {}
        render = RenderSpec(
            source=gt.get("acquisition") or acquisition or "digital-native-pdf",
            dpi=rj.get("dpi"), size_px=list(rj.get("size_px") or [0, 0]),
            page_size=getattr(builder, "page", rj.get("page_size", "A5")) if builder else rj.get("page_size", "A5"),
            page_count=rj.get("page_count", 1),
            target_long_side=(gen_config.target_long_side if gen_config else None),
            keep_aspect=(gen_config.keep_aspect if gen_config else True),
            tiling=({"n_max": gen_config.tiling_n_max} if gen_config and gen_config.tiling_n_max else None),
            box_resolver=rj.get("box_resolver", "pdf_text"),
            color_probe_fallback_count=int(
                rj.get("color_probe_fallback_count", 0)
            ),
            layout_family=rj.get("layout_family"),
            layout_fingerprint=rj.get("layout_fingerprint"),
            overlay_seed=rj.get("overlay_seed"),
            overlay_fingerprint=rj.get("overlay_fingerprint"),
            overlays=list(rj.get("overlays") or []),
            geometry=rj.get("geometry"),
        )
        langs = sorted({f.language for f in fields} | {gt.get("fields", {}).get("language", "en")}
                       if isinstance(gt.get("fields", {}).get("language"), str) else
                       {f.language for f in fields} or {"en"})

        return cls(
            doc_id=gt.get("doc_id", gt.get("type", "doc")),
            doc_type=gt.get("type", "document"), stressors=list(gt.get("stressors", [])),
            anchor_metric=gt.get("anchor_metric", "anls"),
            fields=fields, qa=qa, table_html=gt.get("table_html"),
            selection=gt.get("selection", {}) or {}, redacted=gt.get("redacted", {}) or {},
            reading_order=gt.get("reading_order", []), probes=gt.get("probes", []),
            render=render, degradation=degradation, languages=langs or ["en"],
            domain=domain, acquisition=gt.get("acquisition") or acquisition,
            gen_config=(gen_config.to_dict() if gen_config else None),
            semantic_graph=gt.get("semantic_graph"),
            difficulty=gt.get("difficulty"),
            split=(gen_config.split_name if gen_config else gt.get("split", "synthetic")),
        )

    # -------------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Structured view + a flat back-compat mirror the legacy readers consume."""
        d: dict[str, Any] = {
            "doc_id": self.doc_id,
            "type": self.doc_type,                 # legacy key
            "doc_type": self.doc_type,
            "domain": self.domain,
            "acquisition": self.acquisition,
            "stressors": list(self.stressors),
            "anchor_metric": self.anchor_metric,
            "languages": list(self.languages),
            "source": self.source,
            "split": self.split,
            # --- structured ---
            "fields_detailed": [f.to_dict() for f in self.fields],
            "qa_detailed": [q.to_dict() for q in self.qa],
            "render": self.render.to_dict(),
            "ablation_support": self.support().to_dict(),
        }
        if self.degradation:
            d["degradation"] = self.degradation.to_dict()
            d["degraded_preset"] = self.degradation.preset  # legacy key
        if self.gen_config is not None:
            d["gen_config"] = self.gen_config
        if self.semantic_graph is not None:
            d["semantic_graph"] = self.semantic_graph
        if self.difficulty is not None:
            d["difficulty"] = self.difficulty

        # --- legacy flat mirror (so to_samples.py and existing tests keep working) ---
        d["fields"] = {f.key: f.value for f in self.fields if f.value != ""}
        spotting = {f.key: f.bbox.to_list() for f in self.fields if f.bbox}
        if spotting:
            d["spotting"] = spotting
        if self.qa:
            d["qa"] = [{"key": q.key, "question": q.question, "answers": q.answers,
                        "metric": q.metric, "answer_type": q.answer_type,
                        **({"box": q.answer_bbox.to_list()} if q.answer_bbox else {}),
                        **({"evidence_keys": q.evidence_keys} if q.evidence_keys else {}),
                        **({"evidence_bboxes": [b.to_list() for b in q.evidence_bboxes]}
                           if q.evidence_bboxes else {}),
                        **({"rationale": q.rationale} if q.rationale else {}),
                        **({"graph_query_id": q.graph_query_id}
                           if q.graph_query_id else {})} for q in self.qa]
        if self.table_html:
            d["table_html"] = self.table_html
        if self.selection:
            d["selection"] = self.selection
        if self.redacted:
            d["redacted"] = self.redacted
        if self.reading_order:
            d["reading_order"] = self.reading_order
        if self.probes:
            d["probes"] = self.probes
        return d

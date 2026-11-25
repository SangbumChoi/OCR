"""UDD — the **Universal Document Dataset** HuggingFace layer.

Turns :class:`~docvlm_eval.unified.core.UnifiedSample` records into a HuggingFace ``datasets.Dataset``
with **one uniform schema for every task** (recognition / kie / vqa / localization / table /
reasoning), then shards + uploads it to the Hub. The structured payload (KIE fields, localization
regions with boxes) is JSON-encoded into string columns so a single schema covers all datasets while
losing nothing — decode `fields_json` / `regions_json` to recover the typed payload.

Workflow (see ``scripts/build_udd.py``):
  1. per-dataset **converter**: UnifiedLoader.load(key) → list[UnifiedSample]
  2. ``to_hf_dataset(rows)`` → Dataset (image cast to the HF Image feature)
  3. ``safety_check(rows)`` → build + save + reload + verify nothing was lost (run BEFORE upload)
  4. ``push(ds, repo, config_name=key, ...)`` → sharded upload (one config per benchmark + an "all")
"""

from __future__ import annotations

import json
import re
from typing import Any

from .core import (UnifiedSample, encode_full_annotation, decode_full_annotation,
                   task_detail_for_source, visual_subtype_for_source,
                   visual_type_for_source)


def udd_features():
    """The single, uniform UDD schema (every task fits it; structured payload is JSON-encoded).

    QA columns are aligned one-to-one; alternatives live in ``answer_variants``. Full-page text and
    table markup share the versioned ``full_annotation`` envelope.
    """
    from datasets import Features, Image, Sequence, Value
    return Features({
        "image": Image(),
        "sample_id": Value("string"),
        "source": Value("string"),          # benchmark key
        "task": Value("string"),            # recognition/kie/vqa/localization/table/reasoning
        "qa_tasks": Sequence(Value("string")),  # top-level task aligned per question
        "visual_type": Value("string"),     # primary visual content family
        "visual_subtype": Value("string"),  # source- or annotation-supported subtype
        "task_detail": Sequence(Value("string")),  # fine-grained operation aligned per QA
        "instructions": Sequence(Value("string")),           # N questions on this image
        "answers": Sequence(Value("string")),                # one primary answer per instruction
        "answer_variants": Sequence(Sequence(Value("string"))),
        "reasoning": Sequence(Value("string")),              # optional rationale aligned with each QA
        "fields_json": Value("string"),     # json [{key,value,bbox:[x1,y1,x2,y2,normalized]|null}]
        "regions_json": Value("string"),    # json [{label,text,bbox:[...]|null}]
        "full_annotation": Value("string"),
        "language": Value("string"),
        "metric": Value("string"),
        "page_count": Value("int32"),
        "document_count": Value("int32"),
        # provenance / origin
        "hf_id": Value("string"),
        "split": Value("string"),
        "hf_config": Value("string"),
    })


def _positive_composition_count(value: Any, name: str) -> int:
    count = 1 if value is None else int(value)
    if count < 1:
        raise ValueError(f"{name} must be positive")
    return count


def _row_to_record(r: UnifiedSample) -> dict[str, Any]:
    d = r.to_dict()                         # bbox already flattened to [x1,y1,x2,y2,normalized]
    if r.qas:                               # grouped record -> the lists carry every QA
        instructions = [qa.question for qa in r.qas]
        variants = [[str(a) for a in qa.answers] for qa in r.qas]
        reasoning = [qa.reasoning or "" for qa in r.qas]
        task_details = [qa.task_detail or task_detail_for_source(r.source, qa.task or r.task)
                        for qa in r.qas]
        qa_tasks = [qa.task or r.task for qa in r.qas]
    else:
        instructions = [r.prompt()]
        variants = [[str(a) for a in r.answers]]
        reasoning = [r.reasoning or ""]
        task_details = [r.task_detail or task_detail_for_source(r.source, r.task)]
        qa_tasks = [r.task]
    return {
        "image": r.image_path,              # path -> cast to Image() reads the bytes
        "sample_id": r.sample_id,
        "source": r.source,
        "task": r.task,
        "qa_tasks": qa_tasks,
        "visual_type": r.visual_type or visual_type_for_source(
            r.source, str(r.meta.get("doc_type") or "")),
        "visual_subtype": r.visual_subtype or visual_subtype_for_source(
            r.source, str(r.meta.get("doc_type") or "")),
        "task_detail": task_details,
        "instructions": instructions,
        "answers": [values[0] if values else "" for values in variants],
        "answer_variants": variants,
        "reasoning": reasoning,
        "fields_json": json.dumps(d["fields"], ensure_ascii=False),
        "regions_json": json.dumps(d["regions"], ensure_ascii=False),
        "full_annotation": encode_full_annotation(r.full_text, r.table_html),
        "language": r.language or "",
        "metric": r.metric or "anls",
        "page_count": _positive_composition_count(
            r.meta.get("page_count"),
            "page_count",
        ),
        "document_count": _positive_composition_count(
            r.meta.get("document_count"),
            "document_count",
        ),
        "hf_id": r.hf_id or "",
        "split": r.split or "",
        "hf_config": r.hf_config or "",
    }


def to_hf_dataset(rows: list[UnifiedSample]):
    """Build a ``datasets.Dataset`` (UDD schema) from unified rows that have a cached ``image_path``."""
    from datasets import Dataset
    recs = [_row_to_record(r) for r in rows if r.image_path]
    if not recs:
        raise ValueError("no rows with a cached image_path — load with cache_dir set")
    # cast the FULL uniform schema (not just image) so sources whose optional columns are all-empty
    # — e.g. localization has no `answers` — still get List(string), keeping every source's features
    # identical for a clean cross-dataset concat.
    ds = Dataset.from_list(recs)
    return ds.cast(udd_features())


def safety_check(
    rows: list[UnifiedSample],
    workdir: str,
    *,
    enrich: bool = False,
) -> dict:
    """Build → save_to_disk → reload → verify nothing was lost. Returns a report; raises on mismatch.

    Confirms: row count preserved, the image decodes, and the JSON-encoded structured payload
    round-trips (field/region counts survive). ``enrich=True`` also persists deterministic UDD
    metadata such as image dimensions before the round trip. Run this BEFORE any upload."""
    from pathlib import Path

    from datasets import load_from_disk

    ds = to_hf_dataset(rows)
    if enrich:
        from .enrich import enrich_dataset

        ds = enrich_dataset(ds)
    src = [r for r in rows if r.image_path]
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    ds2 = load_from_disk(str(out))

    assert len(ds2) == len(src), f"row count changed: {len(src)} -> {len(ds2)}"
    img0 = ds2[0]["image"]
    assert getattr(img0, "size", None), "image did not decode after round-trip"
    # structured payload survived
    exp_fields = sum(len(r.fields) for r in src)
    exp_reg = sum(len(r.regions) for r in src)
    if "elements_json" in ds2.column_names:
        elements = [
            item
            for payload in ds2["elements_json"]
            for item in json.loads(payload or "[]")
        ]
        got_fields = sum(item.get("kind") == "field" for item in elements)
        got_reg = sum(item.get("kind") == "region" for item in elements)
    else:
        got_fields = sum(len(json.loads(x or "[]")) for x in ds2["fields_json"])
        got_reg = sum(len(json.loads(x or "[]")) for x in ds2["regions_json"])
    assert exp_fields == got_fields, f"fields lost: {exp_fields} -> {got_fields}"
    assert exp_reg == got_reg, f"regions lost: {exp_reg} -> {got_reg}"
    validate_payload_shapes(ds2)
    return {"rows": len(ds2), "fields": got_fields, "regions": got_reg,
            "image_ok": True, "columns": ds2.column_names}


def validate_payload_shapes(ds) -> None:
    """Payload SHAPE conformance: one DTO for every source, so an adapter emitting a different
    shape fails HERE (safety_check), not in a downstream consumer.

    Published elements use {bbox, information, class, kind} with optional stable `id` and `ref`;
    legacy key/value payloads remain readable for older snapshots. Build-time intermediates
    (fields_json/regions_json in per-source dirs) are
    validated in their pre-merge shapes. The QA pairing invariant
    ``instructions``, QA tasks, primary answers, answer variants, reasoning, and task details share QA indices."""
    cols = ds.column_names
    specs = ([("elements_json", {"key", "value", "bbox", "kind"})] if "elements_json" in cols
             else [("regions_json", {"label", "text", "bbox"}),
                   ("fields_json", {"key", "value", "bbox"})])
    for col, req in specs:
        for x in ds[col]:
            for el in json.loads(x or "[]"):
                expected = req
                if (col == "elements_json" and isinstance(el, dict)
                        and {"bbox", "information", "class", "kind"} <= set(el)):
                    expected = {"bbox", "information", "class", "kind"}
                optional = {"id", "ref"} if col == "elements_json" and expected == {
                    "bbox", "information", "class", "kind"
                } else set()
                assert isinstance(el, dict) and expected <= set(el) and set(el) <= expected | optional, \
                    f"{col} element off-DTO: {el!r} (want keys {sorted(expected)})"
                bb = el["bbox"]
                assert bb is None or (isinstance(bb, list) and len(bb) == 5), \
                    f"{col} bbox off-DTO: {bb!r} (want [x1,y1,x2,y2,normalized] or null)"
                for k in expected - {"bbox", "kind"}:
                    assert isinstance(el[k], str), \
                        f"{col} '{k}' must be a string (required), got {el[k]!r}"
                if "kind" in req:
                    allowed_kinds = ("field", "region")
                    if col == "elements_json" and expected == {"bbox", "information", "class", "kind"}:
                        allowed_kinds = ("field", "region", "figure", "text", "page_title")
                    assert el["kind"] in allowed_kinds, \
                        f"{col} kind off-DTO: {el['kind']!r} (want one of {allowed_kinds})"
    if "full_annotation" in cols:
        for row_index, value in enumerate(ds["full_annotation"]):
            if not value:
                continue
            payload = json.loads(value)
            assert isinstance(payload, dict) and payload.get("version") == 1, \
                f"full_annotation row {row_index} must be a version-1 JSON object"
            assert isinstance(payload.get("text"), str) and isinstance(payload.get("html"), str), \
                f"full_annotation row {row_index} must contain string text and html members"
    if "instructions" in cols:
        for row_index, (qs, ans) in enumerate(zip(ds["instructions"], ds["answers"])):
            assert len(qs) == len(ans) >= 1, \
                f"QA pairing broken: {len(qs)} instructions vs {len(ans)} answers (need ==, >=1)"
            legacy_nested = any(isinstance(a, (list, tuple)) for a in ans)
            assert legacy_nested or all(isinstance(a, str) for a in ans), \
                "answers must contain strings"
            if "answer_variants" in cols:
                variants = ds["answer_variants"][row_index]
                assert len(variants) == len(qs), "answer variants must align with instructions"
                for primary, alternatives in zip(ans, variants):
                    assert isinstance(alternatives, list) and all(
                        isinstance(a, str) for a in alternatives
                    ), "each answer_variants entry must be a list of strings"
                    assert not primary or primary in alternatives, \
                        "primary answer must be retained among answer variants"
            if "qa_tasks" in cols:
                assert len(ds["qa_tasks"][row_index]) == len(qs), \
                    f"QA task pairing broken at row {row_index}"
            if "reasoning" in cols:
                assert len(ds["reasoning"][row_index]) == len(qs), \
                    f"QA reasoning pairing broken at row {row_index}"
            if "task_detail" in cols:
                assert len(ds["task_detail"][row_index]) == len(qs), \
                    f"QA task-detail pairing broken at row {row_index}"
            if "qa_tasks" in cols:
                assert len(ds["qa_tasks"][row_index]) == len(qs), \
                    f"QA task pairing broken at row {row_index}"


def upgrade_udd_dataset(dataset):
    """Migrate legacy/v2 UDD rows to the compact v3 public schema without losing annotations."""
    from .core import VisualType
    from datasets import Sequence, Value

    def upgrade_row(row):
        instructions = list(row.get("instructions") or [])
        raw_answers = list(row.get("answers") or [])
        raw_variants = row.get("answer_variants")
        if raw_variants is None:
            raw_variants = raw_answers
        variants = []
        answers = []
        for item in raw_variants:
            values = [str(v) for v in item if v is not None] if isinstance(item, (list, tuple)) else ([str(item)] if item else [])
            variants.append(values)
        if len(variants) != len(instructions) and len(raw_answers) == len(instructions):
            variants = [[str(v) for v in (a if isinstance(a, (list, tuple)) else [a]) if v is not None]
                        for a in raw_answers]
        answers = [str(a[0]) if a else "" for a in variants]
        if not raw_variants and raw_answers and all(isinstance(a, str) for a in raw_answers):
            answers = [str(a) for a in raw_answers]
            variants = [[a] if a else [] for a in answers]
        if len(instructions) != len(answers):
            raise ValueError("cannot migrate UDD row with mismatched instruction/answer lists")
        reasoning = row.get("reasoning")
        if reasoning is None:
            reasoning = [""] * len(instructions)
        else:
            reasoning = [str(value or "") for value in reasoning]
            if len(reasoning) != len(instructions):
                raise ValueError("cannot migrate UDD row with misaligned reasoning list")
        task_details = row.get("task_detail")
        if task_details is None:
            task_details = [task_detail_for_source(str(row.get("source") or ""),
                                                   str(row.get("task") or ""))
                            for _ in instructions]
        else:
            fallback_detail = task_detail_for_source(str(row.get("source") or ""),
                                                     str(row.get("task") or ""))
            task_details = [str(value or fallback_detail) for value in task_details]
            if len(task_details) != len(instructions):
                raise ValueError("cannot migrate UDD row with misaligned task-detail list")
        qa_tasks = row.get("qa_tasks")
        if qa_tasks is None:
            qa_tasks = [str(row.get("task") or "other") for _ in instructions]
        else:
            qa_tasks = [str(value or row.get("task") or "other") for value in qa_tasks]
            if len(qa_tasks) != len(instructions):
                raise ValueError("cannot migrate UDD row with misaligned task list")
        source = str(row.get("source") or "")
        if source == "hallusionbench":
            kept = []
            for index, question in enumerate(instructions):
                suffix = " Explain your answer."
                if (question.endswith(suffix) and index > 0
                        and instructions[index - 1] == question[:-len(suffix)]):
                    explanation = answers[index]
                    explanation = re.sub(
                        r"\s*So the answer is (?:yes|no)\.?\s*$", "", explanation,
                        flags=re.IGNORECASE,
                    ).strip()
                    if not reasoning[index - 1] and explanation.lower() not in {"", "yes", "no"}:
                        reasoning[index - 1] = explanation
                    continue
                kept.append(index)
            if len(kept) != len(instructions):
                instructions = [instructions[i] for i in kept]
                answers = [answers[i] for i in kept]
                variants = [variants[i] for i in kept]
                reasoning = [reasoning[i] for i in kept]
                task_details = [task_details[i] for i in kept]
                qa_tasks = [qa_tasks[i] for i in kept]
        visual_type = str(row.get("visual_type") or "")
        if visual_type not in VisualType.ALL:
            visual_type = visual_type_for_source(source)
        visual_subtype = str(row.get("visual_subtype") or "")
        if not visual_subtype:
            visual_subtype = visual_subtype_for_source(source)
        old_text, old_html = decode_full_annotation(row.get("full_annotation"))
        full_text = row.get("full_text") or old_text
        table_html = row.get("table_html") or old_html
        return {
            "instructions": instructions,
            "answers": answers,
            "answer_variants": variants,
            "full_annotation": encode_full_annotation(full_text, table_html),
            "visual_type": visual_type,
            "visual_subtype": visual_subtype,
            "reasoning": reasoning,
            "task_detail": task_details,
            "qa_tasks": qa_tasks,
        }

    def _migrate(rows, split_name=""):
        remove = [name for name in ("full_text", "table_html", "n_fields", "n_regions")
                  if name in rows.column_names]
        features = rows.features.copy()
        for name in remove:
            features.pop(name, None)
        features["answers"] = Sequence(Value("string"))
        features["answer_variants"] = Sequence(Sequence(Value("string")))
        features["full_annotation"] = Value("string")
        features["reasoning"] = Sequence(Value("string"))
        features["task_detail"] = Sequence(Value("string"))
        features["qa_tasks"] = Sequence(Value("string"))
        features["visual_type"] = Value("string")
        features["visual_subtype"] = Value("string")
        mapped = rows.map(upgrade_row, desc=f"upgrade UDD v3: {split_name}",
                          load_from_cache_file=False, remove_columns=remove, features=features)
        validate_payload_shapes(mapped)
        return mapped

    if isinstance(dataset, dict):
        from datasets import DatasetDict

        upgraded = DatasetDict({
            split: _migrate(rows, split)
            for split, rows in dataset.items()
        })
        return upgraded
    return _migrate(dataset)


def dedupe_by_phash(ds):
    """Same phash (+ same stored dims) = the same image: store it ONCE, gather every row's
    question/answer pairs into the survivor's native ``instructions``/``answers`` lists (identical
    questions deduped, order preserved, index pairing kept). Non-QA payload (fields/regions/
    full_text/table_html) rides with the survivor — for a truly identical image the duplicates
    carry the same payload. The survivor keeps its own ``fold``; a duplicate that sat in the other
    fold is REMOVED, which closes the leak of identical pixels appearing on both sides of the
    split. Rows without a phash pass through untouched."""
    from collections import defaultdict

    phashes = ds["phash"]
    widths = ds["image_width"]
    heights = ds["image_height"]
    groups: dict[tuple, list[int]] = defaultdict(list)
    order: list[tuple] = []
    for i in range(len(ds)):
        key = (phashes[i], widths[i], heights[i]) if phashes[i] else ("", i, i)
        if key not in groups:
            order.append(key)
        groups[key].append(i)

    keep: list[int] = []
    merged: dict[int, tuple[list, list, list, list, list]] = {}   # survivor index -> paired QA fields
    n_dropped = 0
    instrs = ds["instructions"]
    answers = ds["answers"]
    legacy_nested = bool(answers and answers[0] and isinstance(answers[0][0], (list, tuple)))
    answer_variants = ds["answer_variants"] if "answer_variants" in ds.column_names else answers
    reasoning = ds["reasoning"] if "reasoning" in ds.column_names else [
        [""] * len(q) for q in instrs
    ]
    task_details = ds["task_detail"] if "task_detail" in ds.column_names else [
        [task_detail_for_source(ds["source"][i], ds["task"][i]) for _ in q]
        for i, q in enumerate(instrs)
    ]
    qa_tasks = ds["qa_tasks"] if "qa_tasks" in ds.column_names else [
        [ds["task"][i] for _ in q] for i, q in enumerate(instrs)
    ]
    for key in order:
        idxs = groups[key]
        keep.append(idxs[0])
        if len(idxs) == 1:
            continue
        qs: list[str] = []
        ans: list[str] = []
        variants: list[list[str]] = []
        why: list[str] = []
        detail: list[str] = []
        tasks: list[str] = []
        seen_q: set[str] = set()
        for i in idxs:
            for q, a, golds, rationale, job, qa_task in zip(instrs[i] or [], answers[i] or [],
                                                    answer_variants[i] or [], reasoning[i] or [],
                                                    task_details[i] or [], qa_tasks[i] or []):
                if q.strip() and q.strip() in seen_q:
                    continue
                seen_q.add(q.strip())
                qs.append(q)
                ans.append(list(a) if legacy_nested else str(a or ""))
                variants.append(list(golds) if isinstance(golds, (list, tuple)) else [str(golds)])
                why.append(str(rationale or ""))
                detail.append(str(job or "other"))
                tasks.append(str(qa_task or "other"))
        merged[idxs[0]] = (qs, ans, variants, why, detail, tasks)
        n_dropped += len(idxs) - 1

    out = ds.select(keep)
    if merged:
        pos_of = {orig: pos for pos, orig in enumerate(keep)}
        upd = {pos_of[i]: qa for i, qa in merged.items()}
        def update_row(r, idx):
            qa = upd.get(idx)
            if qa is None:
                result = {"instructions": r["instructions"], "answers": r["answers"],
                          "reasoning": r.get("reasoning", [""] * len(r["instructions"])),
                          "task_detail": r.get("task_detail", [
                              task_detail_for_source(r["source"], r["task"])
                              for _ in r["instructions"]])}
                if "answer_variants" in r:
                    result["answer_variants"] = r["answer_variants"]
                result["qa_tasks"] = r.get("qa_tasks", [r["task"] for _ in r["instructions"]])
                return result
            result = {"instructions": qa[0], "answers": qa[1],
                      "reasoning": qa[3], "task_detail": qa[4], "qa_tasks": qa[5]}
            if "answer_variants" in r:
                result["answer_variants"] = qa[2]
            return result
        out = out.map(update_row,
                      with_indices=True, desc="dedupe: gather QAs onto surviving image",
                      load_from_cache_file=False)
    print(f"[dedupe] {len(ds)} rows -> {len(out)} ({n_dropped} duplicate-image rows folded into "
          f"{len(merged)} survivors' instructions/answers)")
    return out


def unified_from_hf_row(row: dict, image_path: str | None = None):
    """Reconstruct a :class:`~docvlm_eval.unified.core.UnifiedSample` from a UDD/HF row.

    Inverse of :func:`_row_to_record`: decodes ``fields_json`` / ``regions_json`` back into typed
    ``Field`` / ``Region`` (with boxes) and re-attaches provenance. ``image_path`` overrides the row's
    (embedded) image when the caller has already written the decoded image to disk — needed because HF
    rows store image *bytes*, but training / visualization want a file path."""
    from .core import Box, Field, Region, UnifiedSample

    def _box(b):
        return Box(b[0], b[1], b[2], b[3], bool(b[4])) if b else None

    if row.get("elements_json"):        # published schema: ONE element datatype, kind-discriminated
        els = json.loads(row["elements_json"])
        fields = [Field(e.get("class", e.get("key", "")),
                        e.get("information", e.get("value", "")), _box(e.get("bbox")))
                  for e in els if e.get("kind") == "field"]
        regions = [Region(e.get("class", e.get("key", "")), _box(e.get("bbox")),
                          e.get("information", e.get("value", "")))
                   for e in els if e.get("kind") == "region"]
    else:                               # build-time intermediates (per-source dirs)
        fields = [Field(f.get("key", ""), f.get("value", ""), _box(f.get("bbox")))
                  for f in json.loads(row.get("fields_json") or "[]")]
        regions = [Region(r.get("label", ""), _box(r.get("bbox")), r.get("text", ""))
                   for r in json.loads(row.get("regions_json") or "[]")]
    # instructions[i] pairs with answers[i]: >1 QA -> the grouped state (qas populated, flat pair
    # empty, per the flat-XOR-grouped invariant); exactly one QA -> the flat state
    instrs = list(row.get("instructions") or [])
    raw_answers = list(row.get("answers") or [])
    raw_variants = row.get("answer_variants")
    ans = []
    for index, answer in enumerate(raw_answers):
        if raw_variants is not None and index < len(raw_variants):
            item = raw_variants[index]
            ans.append(list(item) if isinstance(item, (list, tuple)) else ([str(item)] if item else []))
        else:
            ans.append(list(answer) if isinstance(answer, (list, tuple)) else ([str(answer)] if answer else []))
    full_text, table_html = decode_full_annotation(row.get("full_annotation"))
    full_text = row.get("full_text") or full_text
    table_html = row.get("table_html") or table_html
    reasoning = list(row.get("reasoning") or [""] * len(instrs))
    task_details = list(row.get("task_detail") or [
        task_detail_for_source(row.get("source", ""), row.get("task", ""))
        for _ in instrs
    ])
    qa_tasks = list(row.get("qa_tasks") or [row.get("task", "") for _ in instrs])
    common = dict(
        sample_id=row.get("sample_id", ""), source=row.get("source", ""), task=row.get("task", ""),
        fields=fields, regions=regions, full_text=full_text,
        table_html=table_html,
        reasoning=reasoning[0] if len(instrs) == 1 and reasoning else "",
        visual_type=row.get("visual_type") or "",
        visual_subtype=row.get("visual_subtype") or "",
        task_detail=task_details[0] if len(instrs) == 1 and task_details else "",
        language=row.get("language") or None,
        metric=row.get("metric") or "anls", image_path=image_path,
        hf_id=row.get("hf_id") or None, split=row.get("split") or None,
        hf_config=row.get("hf_config") or None)
    if len(instrs) > 1:
        from .core import QA
        return UnifiedSample(instruction="", answers=[],
                             qas=[QA(q, a, reasoning[i] if i < len(reasoning) else "",
                                     task_details[i] if i < len(task_details) else "other",
                                     qa_tasks[i] if i < len(qa_tasks) else row.get("task", ""))
                                  for i, (q, a) in enumerate(zip(instrs, ans))], **common)
    return UnifiedSample(instruction=instrs[0] if instrs else "",
                         answers=ans[0] if ans else [], **common)


def push(ds, repo: str, *, config_name: str | None = None, token: str | None = None,
         private: bool = True, max_shard_size: str = "500MB") -> str:
    """Push a UDD Dataset to the Hub as ``config_name`` (sharded). Returns the repo URL."""
    ds.push_to_hub(repo, config_name=config_name or "default", token=token,
                   private=private, max_shard_size=max_shard_size)
    return f"https://huggingface.co/datasets/{repo}"

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

from .core import (UnifiedSample, task_detail_for_source, visual_subtype_for_source,
                   visual_type_for_source)


def udd_features():
    """The single, uniform UDD schema (every task fits it; structured payload is JSON-encoded).

    The QA pairing is NATIVE list columns, not a JSON side-channel: ``instructions[i]`` is answered
    by ``answers[i]`` (a list of interchangeable gold variants). One image = one row = N QAs, with
    the two list levels structurally distinct — outer index pairs with the question, inner list is
    surface variants of ONE answer. ``len(instructions) == len(answers) >= 1`` always
    (enforced by :func:`validate_payload_shapes`)."""
    from datasets import Features, Image, Sequence, Value
    return Features({
        "image": Image(),
        "sample_id": Value("string"),
        "source": Value("string"),          # benchmark key
        "task": Value("string"),            # recognition/kie/vqa/localization/table/reasoning
        "visual_type": Value("string"),     # primary visual content family
        "visual_subtype": Value("string"),  # source- or annotation-supported subtype
        "task_detail": Sequence(Value("string")),  # fine-grained operation aligned per QA
        "instructions": Sequence(Value("string")),           # N questions on this image
        "answers": Sequence(Sequence(Value("string"))),      # answers[i] = golds for instructions[i]
        "reasoning": Sequence(Value("string")),              # optional rationale aligned with each QA
        "fields_json": Value("string"),     # json [{key,value,bbox:[x1,y1,x2,y2,normalized]|null}]
        "regions_json": Value("string"),    # json [{label,text,bbox:[...]|null}]
        "full_text": Value("string"),
        "table_html": Value("string"),
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
        answers = [[str(a) for a in qa.answers] for qa in r.qas]
        reasoning = [qa.reasoning or "" for qa in r.qas]
        task_details = [qa.task_detail or task_detail_for_source(r.source, r.task) for qa in r.qas]
    else:
        instructions = [r.prompt()]
        answers = [[str(a) for a in r.answers]]
        reasoning = [r.reasoning or ""]
        task_details = [r.task_detail or task_detail_for_source(r.source, r.task)]
    return {
        "image": r.image_path,              # path -> cast to Image() reads the bytes
        "sample_id": r.sample_id,
        "source": r.source,
        "task": r.task,
        "visual_type": r.visual_type or visual_type_for_source(
            r.source, str(r.meta.get("doc_type") or "")),
        "visual_subtype": r.visual_subtype or visual_subtype_for_source(
            r.source, str(r.meta.get("doc_type") or "")),
        "task_detail": task_details,
        "instructions": instructions,
        "answers": answers,
        "reasoning": reasoning,
        "fields_json": json.dumps(d["fields"], ensure_ascii=False),
        "regions_json": json.dumps(d["regions"], ensure_ascii=False),
        "full_text": r.full_text or "",
        "table_html": r.table_html or "",
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

    Published schema: every localized element in ``elements_json`` is exactly
    {key, value, bbox: [x1,y1,x2,y2,normalized]|null, kind: "field"|"region"} with key/value
    required strings. Build-time intermediates (fields_json/regions_json in per-source dirs) are
    validated in their pre-merge shapes. The QA pairing invariant
    ``len(instructions) == len(answers) >= 1`` holds in both."""
    cols = ds.column_names
    specs = ([("elements_json", {"key", "value", "bbox", "kind"})] if "elements_json" in cols
             else [("regions_json", {"label", "text", "bbox"}),
                   ("fields_json", {"key", "value", "bbox"})])
    for col, req in specs:
        for x in ds[col]:
            for el in json.loads(x or "[]"):
                assert isinstance(el, dict) and set(el) == req, \
                    f"{col} element off-DTO: {el!r} (want keys {sorted(req)})"
                bb = el["bbox"]
                assert bb is None or (isinstance(bb, list) and len(bb) == 5), \
                    f"{col} bbox off-DTO: {bb!r} (want [x1,y1,x2,y2,normalized] or null)"
                for k in req - {"bbox", "kind"}:
                    assert isinstance(el[k], str), \
                        f"{col} '{k}' must be a string (required), got {el[k]!r}"
                if "kind" in req:
                    assert el["kind"] in ("field", "region"), \
                        f"{col} kind off-DTO: {el['kind']!r} (want 'field'|'region')"
    if "instructions" in cols:
        for row_index, (qs, ans) in enumerate(zip(ds["instructions"], ds["answers"])):
            assert len(qs) == len(ans) >= 1, \
                f"QA pairing broken: {len(qs)} instructions vs {len(ans)} answer lists (need ==, >=1)"
            if "reasoning" in cols:
                assert len(ds["reasoning"][row_index]) == len(qs), \
                    f"QA reasoning pairing broken at row {row_index}"
            if "task_detail" in cols:
                assert len(ds["task_detail"][row_index]) == len(qs), \
                    f"QA task-detail pairing broken at row {row_index}"


def upgrade_udd_dataset(dataset):
    """Add the fine-grained UDD columns without changing existing answers or payloads.

    Missing rationale values become empty strings: the migration never manufactures explanations.
    Existing non-empty labels and aligned rationale values are retained.
    """
    from .core import VisualType

    def upgrade_row(row):
        instructions = list(row.get("instructions") or [])
        answers = list(row.get("answers") or [])
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
        source = str(row.get("source") or "")
        if source == "hallusionbench":
            kept = []
            for index, question in enumerate(instructions):
                suffix = " Explain your answer."
                if (question.endswith(suffix) and index > 0
                        and instructions[index - 1] == question[:-len(suffix)]):
                    explanation = answers[index][0] if answers[index] else ""
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
                reasoning = [reasoning[i] for i in kept]
                task_details = [task_details[i] for i in kept]
        visual_type = str(row.get("visual_type") or "")
        if visual_type not in VisualType.ALL:
            visual_type = visual_type_for_source(source)
        visual_subtype = str(row.get("visual_subtype") or "")
        if not visual_subtype:
            visual_subtype = visual_subtype_for_source(source)
        return {
            "instructions": instructions,
            "answers": answers,
            "visual_type": visual_type,
            "visual_subtype": visual_subtype,
            "reasoning": reasoning,
            "task_detail": task_details,
        }

    if isinstance(dataset, dict):
        from datasets import DatasetDict

        upgraded = DatasetDict({
            split: rows.map(upgrade_row, desc=f"upgrade UDD schema: {split}",
                            load_from_cache_file=False)
            for split, rows in dataset.items()
        })
        for rows in upgraded.values():
            validate_payload_shapes(rows)
        return upgraded
    upgraded = dataset.map(upgrade_row, desc="upgrade UDD schema", load_from_cache_file=False)
    validate_payload_shapes(upgraded)
    return upgraded


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
    merged: dict[int, tuple[list, list, list, list]] = {}   # survivor index -> paired QA fields
    n_dropped = 0
    instrs = ds["instructions"]
    answers = ds["answers"]
    reasoning = ds["reasoning"] if "reasoning" in ds.column_names else [
        [""] * len(q) for q in instrs
    ]
    task_details = ds["task_detail"] if "task_detail" in ds.column_names else [
        [task_detail_for_source(ds["source"][i], ds["task"][i]) for _ in q]
        for i, q in enumerate(instrs)
    ]
    for key in order:
        idxs = groups[key]
        keep.append(idxs[0])
        if len(idxs) == 1:
            continue
        qs: list[str] = []
        ans: list[list[str]] = []
        why: list[str] = []
        detail: list[str] = []
        seen_q: set[str] = set()
        for i in idxs:
            for q, a, rationale, job in zip(instrs[i] or [], answers[i] or [],
                                             reasoning[i] or [], task_details[i] or []):
                if q.strip() and q.strip() in seen_q:
                    continue
                seen_q.add(q.strip())
                qs.append(q)
                ans.append(list(a))
                why.append(str(rationale or ""))
                detail.append(str(job or "other"))
        merged[idxs[0]] = (qs, ans, why, detail)
        n_dropped += len(idxs) - 1

    out = ds.select(keep)
    if merged:
        pos_of = {orig: pos for pos, orig in enumerate(keep)}
        upd = {pos_of[i]: qa for i, qa in merged.items()}
        out = out.map(lambda r, idx: ({"instructions": upd[idx][0], "answers": upd[idx][1],
                                       "reasoning": upd[idx][2], "task_detail": upd[idx][3]}
                                      if idx in upd else
                                      {"instructions": r["instructions"], "answers": r["answers"],
                                       "reasoning": r.get("reasoning", [""] * len(r["instructions"])),
                                       "task_detail": r.get("task_detail", [
                                           task_detail_for_source(r["source"], r["task"])
                                           for _ in r["instructions"]])}),
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
        fields = [Field(e.get("key", ""), e.get("value", ""), _box(e.get("bbox")))
                  for e in els if e.get("kind") == "field"]
        regions = [Region(e.get("key", ""), _box(e.get("bbox")), e.get("value", ""))
                   for e in els if e.get("kind") == "region"]
    else:                               # build-time intermediates (per-source dirs)
        fields = [Field(f.get("key", ""), f.get("value", ""), _box(f.get("bbox")))
                  for f in json.loads(row.get("fields_json") or "[]")]
        regions = [Region(r.get("label", ""), _box(r.get("bbox")), r.get("text", ""))
                   for r in json.loads(row.get("regions_json") or "[]")]
    # instructions[i] pairs with answers[i]: >1 QA -> the grouped state (qas populated, flat pair
    # empty, per the flat-XOR-grouped invariant); exactly one QA -> the flat state
    instrs = list(row.get("instructions") or [])
    ans = [list(a) for a in (row.get("answers") or [])]
    reasoning = list(row.get("reasoning") or [""] * len(instrs))
    task_details = list(row.get("task_detail") or [
        task_detail_for_source(row.get("source", ""), row.get("task", ""))
        for _ in instrs
    ])
    common = dict(
        sample_id=row.get("sample_id", ""), source=row.get("source", ""), task=row.get("task", ""),
        fields=fields, regions=regions, full_text=row.get("full_text") or None,
        table_html=row.get("table_html") or None,
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
                                     task_details[i] if i < len(task_details) else "other")
                                  for i, (q, a) in enumerate(zip(instrs, ans))], **common)
    return UnifiedSample(instruction=instrs[0] if instrs else "",
                         answers=ans[0] if ans else [], **common)


def push(ds, repo: str, *, config_name: str | None = None, token: str | None = None,
         private: bool = True, max_shard_size: str = "500MB") -> str:
    """Push a UDD Dataset to the Hub as ``config_name`` (sharded). Returns the repo URL."""
    ds.push_to_hub(repo, config_name=config_name or "default", token=token,
                   private=private, max_shard_size=max_shard_size)
    return f"https://huggingface.co/datasets/{repo}"

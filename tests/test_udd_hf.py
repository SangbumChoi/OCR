"""Offline test for the UDD HuggingFace layer (docvlm_eval.unified.hf).

Builds a couple of UnifiedSamples over a real temp image, runs to_hf_dataset + safety_check, and
asserts the uniform schema + that the JSON-encoded structured payload (fields/boxes) round-trips."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvlm_eval.unified import Box, Field, Region, Task, UnifiedSample
from docvlm_eval.unified.hf import safety_check, to_hf_dataset, udd_features


def _img(tmp: Path) -> str:
    from PIL import Image
    p = tmp / "img.png"
    Image.new("RGB", (40, 30), (200, 200, 200)).save(p)
    return str(p)


def test_udd_schema_columns():
    cols = set(udd_features().keys())
    assert {"image", "task", "visual_type", "visual_subtype", "task_detail", "reasoning",
            "fields_json", "regions_json", "answers", "source"} <= cols


def test_to_hf_dataset_uniform_across_tasks(tmp_path):
    img = _img(tmp_path)
    rows = [
        UnifiedSample(sample_id="cord_0_0", source="cord", task=Task.KIE, image_path=img,
                      fields=[Field("menu.nm", "Coffee", Box(10, 20, 60, 40, False))],
                      answers=['{"menu.nm": "Coffee"}']),
        UnifiedSample(sample_id="docvqa_0_0", source="docvqa", task=Task.VQA, image_path=img,
                      instruction="Total?", answers=["$5"], reasoning="Read the total field.",
                      meta={"page_count": 3, "document_count": 1}),
        UnifiedSample(sample_id="ocrvqa_0_0", source="ocrvqa", task=Task.VQA, image_path=img,
                      instruction="Author?", answers=["X"],
                      regions=[Region("word", Box(0.1, 0.2, 0.4, 0.25, True), "X")]),
    ]
    ds = to_hf_dataset(rows)
    # one uniform schema covers all three tasks
    assert set(ds.column_names) == set(udd_features().keys())
    assert len(ds) == 3
    by = {r["sample_id"]: r for r in ds}
    f = json.loads(by["cord_0_0"]["fields_json"])
    assert f[0]["key"] == "menu.nm" and f[0]["bbox"][:4] == [10.0, 20.0, 60.0, 40.0]
    rg = json.loads(by["ocrvqa_0_0"]["regions_json"])
    assert rg[0]["bbox"][4] is True                      # normalized flag preserved
    assert by["cord_0_0"]["page_count"] == 1
    assert by["cord_0_0"]["document_count"] == 1
    assert by["docvqa_0_0"]["page_count"] == 3
    assert by["cord_0_0"]["visual_type"] == "document"
    assert by["cord_0_0"]["visual_subtype"] == "receipt"
    assert by["cord_0_0"]["task_detail"] == ["receipt_field_extraction"]
    assert by["docvqa_0_0"]["reasoning"] == ["Read the total field."]


def test_safety_check_roundtrip(tmp_path):
    from datasets import load_from_disk

    img = _img(tmp_path)
    rows = [UnifiedSample(sample_id="cord_0_0", source="cord", task=Task.KIE, image_path=img,
                          fields=[Field("k", "v", Box(1, 2, 3, 4, False)),
                                  Field("k2", "v2", None)],
                          answers=['{"k": "v"}'])]
    output = tmp_path / "ds"
    rep = safety_check(rows, str(output), enrich=True)
    saved = load_from_disk(str(output))
    assert rep["rows"] == 1 and rep["fields"] == 2 and rep["image_ok"] is True
    assert saved[0]["image_width"] == 40
    assert saved[0]["image_height"] == 30


def test_to_hf_dataset_requires_image():
    import pytest
    with pytest.raises(ValueError):
        to_hf_dataset([UnifiedSample(sample_id="x", source="y", task=Task.VQA, answers=["a"])])


def test_table_html_and_full_text_remain_distinct_targets(tmp_path):
    from docvlm_eval.unified import Task

    img = _img(tmp_path)
    table = UnifiedSample(
        sample_id="table-1", source="pubtabnet", task=Task.TABLE, image_path=img,
        full_text="Header A Header B Total 42", table_html="<table><tr><td>42</td></tr></table>",
    )
    sample = table.to_sample()

    assert sample.answers == ["<table><tr><td>42</td></tr></table>"]
    assert table.full_text == "Header A Header B Total 42"


def test_schema_upgrade_preserves_payload_and_adds_aligned_empty_rationale():
    from datasets import Dataset, DatasetDict
    from docvlm_eval.unified import upgrade_udd_dataset

    row = {
        "sample_id": "table-1", "source": "pubtabnet", "task": "table",
        "instructions": ["Convert this table to HTML."],
        "answers": [["<table><tr><td>42</td></tr></table>"]],
        "full_text": "Header A Header B Total 42",
        "table_html": "<table><tr><td>42</td></tr></table>",
        "elements_json": "[]",
    }
    source = Dataset.from_list([row])
    migrated = upgrade_udd_dataset(DatasetDict({"train": source, "validation": source}))
    migrated = migrated["train"][0]

    assert migrated["visual_type"] == "table"
    assert migrated["visual_subtype"] == "scientific_table"
    assert migrated["task_detail"] == ["table_structure_reconstruction"]
    assert migrated["reasoning"] == [""]
    assert migrated["answers"] == row["answers"]
    assert migrated["full_text"] == row["full_text"]
    assert migrated["table_html"] == row["table_html"]


def test_schema_upgrade_pairs_legacy_hallusion_explanations_as_rationale():
    from datasets import Dataset
    from docvlm_eval.unified import upgrade_udd_dataset

    row = {
        "sample_id": "hallusion-1", "source": "hallusionbench", "task": "reasoning",
        "instructions": ["Is the chart increasing?", "Is the chart increasing? Explain your answer."],
        "answers": [["no"], ["The line peaks in May. So the answer is no."]],
        "elements_json": "[]",
    }
    migrated = upgrade_udd_dataset(Dataset.from_list([row]))[0]

    assert migrated["instructions"] == ["Is the chart increasing?"]
    assert migrated["answers"] == [["no"]]
    assert migrated["reasoning"] == ["The line peaks in May."]
    assert len(migrated["task_detail"]) == len(migrated["instructions"])


def _mini_udd(tmp_path, rows_spec):
    """Build a tiny enriched-shaped Dataset (native instructions/answers lists) from specs."""
    from datasets import Dataset
    recs = []
    for i, (ph, q, a) in enumerate(rows_spec):
        recs.append({"sample_id": f"src_{i:04d}_0", "source": "src", "task": "vqa",
                     "instructions": [q], "answers": [[a]], "fields_json": "[]",
                     "regions_json": "[]", "full_text": "", "table_html": "", "language": "en",
                     "metric": "exact", "hf_id": "", "split": "test", "hf_config": "",
                     "n_fields": 0, "n_regions": 0, "image_width": 40, "image_height": 30,
                     "phash": ph, "license": "unspecified", "fold": "train"})
    return Dataset.from_list(recs)


def test_dedupe_by_phash_gathers_qas(tmp_path):
    from docvlm_eval.unified import dedupe_by_phash
    ds = _mini_udd(tmp_path, [
        ("aaaa", "Who wrote this?", "Smith"),
        ("aaaa", "What is the title?", "Physics"),      # same image -> folded into row 0
        ("aaaa", "Who wrote this?", "Smith"),           # identical question -> deduped
        ("bbbb", "What genre?", "Science"),             # different image -> untouched
    ])
    out = dedupe_by_phash(ds)
    assert len(out) == 2                                # one row per distinct image
    assert out[0]["instructions"] == ["Who wrote this?", "What is the title?"]
    assert out[0]["answers"] == [["Smith"], ["Physics"]]   # index pairing preserved
    assert out[0]["task_detail"] == ["visual_question_answering"] * 2
    assert out[1]["instructions"] == ["What genre?"]


def test_unified_from_hf_row_expands_native_lists(tmp_path):
    from docvlm_eval.unified import to_training_samples, unified_from_hf_row
    row = {"sample_id": "src_0000_0", "source": "src", "task": "vqa",
           "instructions": ["Who wrote this?", "What is the title?"],
           "answers": [["Smith"], ["Physics", "PHYSICS BOOK"]],
           "reasoning": ["Read the author line.", "Read the title line."],
           "fields_json": "[]", "regions_json": "[]", "metric": "exact"}
    r = unified_from_hf_row(row, image_path=_img(tmp_path))
    assert len(r.qas) == 2 and not r.instruction        # grouped state (flat-XOR-grouped holds)
    assert r.qas[1].answers == ["Physics", "PHYSICS BOOK"]   # inner list = variants of ONE answer
    assert r.qas[0].reasoning == "Read the author line."
    assert r.qas[1].reasoning == "Read the title line."
    samples = to_training_samples([r])
    assert len(samples) == 2                            # both QAs train, one image decode
    assert {s.question for s in samples} == {"Who wrote this?", "What is the title?"}
    # single-QA row stays flat
    r2 = unified_from_hf_row({**row, "instructions": ["Who wrote this?"], "answers": [["Smith"]]},
                             image_path=_img(tmp_path))
    assert r2.instruction == "Who wrote this?" and not r2.qas


def test_elements_json_single_datatype_roundtrip(tmp_path):
    # published schema: fields and regions share ONE {key,value,bbox,kind} element type
    from docvlm_eval.unified import unified_from_hf_row
    row = {"sample_id": "x_0_0", "source": "cord", "task": "kie",
           "instructions": ["Extract."], "answers": [["{}"]], "metric": "anls",
           "elements_json": json.dumps([
               {"key": "total", "value": "60,000", "bbox": [0.1, 0.8, 0.4, 0.9, True],
                "kind": "field"},
               {"key": "Table", "value": "", "bbox": [0.0, 0.1, 1.0, 0.5, True],
                "kind": "region"}])}
    r = unified_from_hf_row(row, image_path=_img(tmp_path))
    assert len(r.fields) == 1 and r.fields[0].key == "total" and r.fields[0].bbox.normalized
    assert len(r.regions) == 1 and r.regions[0].label == "Table"


def test_validate_payload_shapes_rejects_off_dto(tmp_path):
    import pytest
    from docvlm_eval.unified.hf import validate_payload_shapes
    img = _img(tmp_path)
    ds = to_hf_dataset([UnifiedSample(sample_id="x_0_0", source="x", task=Task.VQA, image_path=img,
                                      instruction="Q?", answers=["a"])])
    validate_payload_shapes(ds)                          # conforming corpus passes
    bad = ds.map(lambda r: {"regions_json": json.dumps([{"label": "w", "boxes": [[1, 2], [3, 4]]}])},
                 load_from_cache_file=False)             # list-of-lists box + wrong key = off-DTO
    with pytest.raises(AssertionError, match="off-DTO"):
        validate_payload_shapes(bad)

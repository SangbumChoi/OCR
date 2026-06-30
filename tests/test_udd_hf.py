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
    assert {"image", "task", "fields_json", "regions_json", "answers", "source"} <= cols


def test_to_hf_dataset_uniform_across_tasks(tmp_path):
    img = _img(tmp_path)
    rows = [
        UnifiedSample(sample_id="cord_0_0", source="cord", task=Task.KIE, image_path=img,
                      fields=[Field("menu.nm", "Coffee", Box(10, 20, 60, 40, False))],
                      answers=['{"menu.nm": "Coffee"}']),
        UnifiedSample(sample_id="docvqa_0_0", source="docvqa", task=Task.VQA, image_path=img,
                      instruction="Total?", answers=["$5"]),
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


def test_safety_check_roundtrip(tmp_path):
    img = _img(tmp_path)
    rows = [UnifiedSample(sample_id="cord_0_0", source="cord", task=Task.KIE, image_path=img,
                          fields=[Field("k", "v", Box(1, 2, 3, 4, False)),
                                  Field("k2", "v2", None)],
                          answers=['{"k": "v"}'])]
    rep = safety_check(rows, str(tmp_path / "ds"))
    assert rep["rows"] == 1 and rep["fields"] == 2 and rep["image_ok"] is True


def test_to_hf_dataset_requires_image():
    import pytest
    with pytest.raises(ValueError):
        to_hf_dataset([UnifiedSample(sample_id="x", source="y", task=Task.VQA, answers=["a"])])

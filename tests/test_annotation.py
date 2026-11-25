"""Layered document annotation (DAR): schema, validation, rotation, export, ablation arms,
structured metrics and error attribution — all offline, on the worked example."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from docvlm_eval.annotation import (
    ARMS,
    FAMILIES,
    Cell,
    DocAnnotation,
    Table,
    attribute,
    build_arm,
    diagnostic_samples,
    eval_samples,
    export_samples,
    failure_profile,
    materialize_rotations,
    rotate_annotation,
    rotate_box,
    to_unified,
    to_upright,
    validate,
    value_per_hour,
)
from docvlm_eval.annotation.validate import ERROR, WARNING
from docvlm_eval.metrics.structured import extract_final_answer, final_answer, kie_f1
from docvlm_eval.metrics.text import score_sample

EX = Path(__file__).resolve().parents[1] / "data" / "annotations" / "example" / "invoice_0001.json"


@pytest.fixture()
def ann() -> DocAnnotation:
    return DocAnnotation.from_json(EX.read_text(encoding="utf-8"))


def _levels(issues, level):
    return [i for i in issues if i.level == level]


# --------------------------------------------------------------------------- schema + validation
def test_example_is_clean_and_roundtrips(ann):
    assert validate(ann) == []
    again = DocAnnotation.from_dict(json.loads(ann.to_json()))
    assert again.to_dict() == ann.to_dict()
    assert all(ann.has_layer(x) for x in ("page", "layout", "ocr", "table", "kie", "understanding"))


def test_full_text_follows_reading_order(ann):
    text = ann.full_text().splitlines()
    assert text[0] == "INVOICE"
    # totals (order 4) come before the stamp (order 5) even though the stamp is further left
    assert text.index("TOTAL DUE") < text.index("PAID") < text.index(text[-1])


@pytest.mark.parametrize("mutate,level,needle", [
    (lambda a: a.understanding.qa[0].evidence.append("t999"), ERROR, "unknown id"),
    (lambda a: setattr(a.kie[6], "value", "1,289.00"), WARNING, "not found in its evidence"),
    (lambda a: setattr(a.ocr[0], "bbox", [10, 10, 900, 60]), ERROR, "outside"),
    (lambda a: setattr(a.provenance.layers["kie"], "status", "absent"), ERROR, "status is 'absent'"),
    (lambda a: setattr(a.kie[8], "value", "PO-1"), ERROR, "present=False"),
    (lambda a: a.tables[0].cells.append(Cell(row=1, col=1, text="x")), ERROR, "overlaps"),
    (lambda a: setattr(a.layout[1], "order", 0), ERROR, "not unique"),
    (lambda a: setattr(a.page, "orientation", 45), ERROR, "orientation"),
])
def test_validator_catches_annotation_drift(ann, mutate, level, needle):
    mutate(ann)
    hits = [i for i in _levels(validate(ann), level) if needle in i.message]
    assert hits, [str(i) for i in validate(ann)]


def test_table_html_honours_spans():
    t = Table("tb", 2, 3, cells=[Cell(0, 0, "A", col_span=2, header=True), Cell(0, 2, "B", header=True),
                                 Cell(1, 0, "1"), Cell(1, 1, "2"), Cell(1, 2, "3")])
    assert t.to_html() == ('<table><tr><th colspan="2">A</th><th>B</th></tr>'
                           '<tr><td>1</td><td>2</td><td>3</td></tr></table>')


# --------------------------------------------------------------------------- geometry
def test_rotate_box_is_exact_and_invertible():
    assert rotate_box([10, 20, 30, 40], 100, 200, 90) == [160, 10, 180, 30]
    b = [10, 20, 30, 40]
    W, H = 100, 200
    for deg in (90, 180, 270):
        r = rotate_box(b, W, H, deg)
        W2, H2 = (H, W) if deg in (90, 270) else (W, H)
        assert rotate_box(r, W2, H2, (360 - deg) % 360) == b


def test_rotate_annotation_four_times_is_identity(ann):
    r = ann
    for _ in range(4):
        r = rotate_annotation(r, 90)
    assert r.page.orientation == 0 and (r.page.width, r.page.height) == (800, 1000)
    assert [ln.bbox for ln in r.ocr] == [ln.bbox for ln in ann.ocr]
    r90 = rotate_annotation(ann, 90)
    assert r90.page.orientation == 90 and validate(r90) == []
    assert [ln.bbox for ln in to_upright(r90).ocr] == [ln.bbox for ln in ann.ocr]


def test_materialize_rotations_writes_matching_images(ann, tmp_path):
    from PIL import Image
    pairs = materialize_rotations(ann, str(EX.parent / ann.image), str(tmp_path))
    assert [a.page.orientation for a, _ in pairs] == [90, 180, 270]
    for a, p in pairs:
        assert Image.open(p).size == (a.page.width, a.page.height)


# --------------------------------------------------------------------------- export + arms
def test_target_arm_only_emits_end_task(ann):
    s = build_arm([ann], "L0_target", base_dir=EX.parent)
    assert {x.meta["layer"] for x in s} == {"kie", "qa"}
    assert all(Path(x.image_path).is_absolute() for x in s)


def test_additive_arms_add_exactly_their_family(ann):
    base = {x.sample_id for x in build_arm([ann], "L0_target")}
    for fam, spec in FAMILIES.items():
        if spec.get("rotate"):
            continue                      # needs rot_dir; covered separately
        arm = build_arm([ann], f"L1_+{fam}")
        extra = {x.meta["layer"] for x in arm if x.sample_id not in base}
        assert extra <= set(spec.get("layers", ())), (fam, extra)


def test_leave_one_out_arms_differ_by_one_family():
    full = ARMS["L2_all"]
    for fam in FAMILIES:
        loo = ARMS[f"L2_all-{fam}"]
        assert set(full.families) - set(loo.families) == {fam}
    assert ARMS["L2_all-reasoning"].reasoning == "answer"
    assert ARMS["L2_all-knowledge"].reasoning == "chain"
    assert ARMS["L1_+knowledge"].reasoning == "answer"
    assert ARMS["L1_+reasoning+knowledge"].reasoning == "guided_chain"


def test_orientation_arm_adds_rotated_orientation_questions(ann, tmp_path):
    s = build_arm([ann], "L1_+orientation", base_dir=EX.parent, rot_dir=tmp_path)
    orient = sorted(x.answers[0] for x in s if x.meta["layer"] == "orientation")
    assert orient == ["0", "180", "270", "90"]


def test_chain_targets_end_in_the_gold_answer(ann):
    for style in ("chain", "guided_chain"):
        qa = [x for x in export_samples(ann, ("qa",), reasoning=style) if x.meta["layer"] == "qa"]
        q2 = next(x for x in qa if x.sample_id.endswith("qa_q2"))
        assert q2.answers[0].endswith("Answer: 2024-05-03")
        assert ("Context:" in q2.answers[0]) == (style == "guided_chain")
        # a bare answer and the gold chain agree under the final_answer metric
        assert score_sample(q2.metric, "2024-05-03", q2.answers) == 1.0
        assert score_sample(q2.metric, "Answer: 2024-03-04", q2.answers) < 1.0


def test_pseudo_layers_can_be_excluded(ann):
    ann.provenance.layers["ocr"].status = "pseudo"
    assert any(x.meta["layer"] == "ocr" for x in export_samples(ann, ("ocr",)))
    assert not export_samples(ann, ("ocr",), include_pseudo=False)


def test_eval_set_is_heldout_end_task_only(ann):
    assert eval_samples([ann]) == []
    ann.split = "heldout"
    ev = eval_samples([ann], "L1_+reasoning")
    assert {x.meta["layer"] for x in ev} == {"kie", "qa"}
    assert not build_arm([ann], "L2_all")          # held-out images never reach training


def test_value_per_hour_uses_provenance_seconds(ann):
    v = value_per_hour({"L0_target": 0.50, "L1_+ocr": 0.56}, [ann])
    assert v["ocr"]["delta"] == pytest.approx(0.06)
    assert v["ocr"]["delta_per_hour"] == pytest.approx(0.06 / (240 / 3600))


# --------------------------------------------------------------------------- metrics
def test_kie_f1_rewards_values_and_punishes_hallucinated_absent_fields():
    gold = json.dumps({"total": "1298.00", "po": None})
    assert kie_f1('{"total": "1,298.00"}', [gold]) == 1.0
    assert kie_f1('{"total": "1298.00", "po": "PO-1"}', [gold]) < 1.0
    assert kie_f1("not json", [gold]) == 0.0


def test_final_answer_takes_the_last_marker():
    assert extract_final_answer("Answer: 3\nwait\nAnswer: 5") == "5"
    assert extract_final_answer("59.00") == "59.00"
    assert final_answer("1. 1180 x 0.05\nAnswer: 59", ["steps\nAnswer: 59.00"]) == 1.0


# --------------------------------------------------------------------------- diagnosis + bridge
def test_attribution_finds_the_lowest_failed_layer(ann):
    probes = {s.sample_id: s for s in diagnostic_samples(ann)}
    preds = {sid: s.answers[0] for sid, s in probes.items()}          # a perfect model ...
    did = ann.doc_id
    preds[f"{did}:kie_f7"] = "1289.00"                                 # total wrong, misread
    preds[f"{did}:read_t29"] = "1,289.00"
    preds[f"{did}:qa_q2"] = "2024-04-03"                               # reads fine, reasons wrong
    rows = {r["item"]: r for r in attribute(ann, preds)}
    assert rows["f7"]["failed_layer"] == "ocr"
    assert rows["q2"]["failed_layer"] == "reasoning"
    assert rows["f1"]["correct"]
    preds[f"{did}:orient"] = "90"
    assert {r["item"]: r for r in attribute(ann, preds)}["q2"]["failed_layer"] == "orientation"
    prof = failure_profile(attribute(ann, preds))
    assert prof["n_errors"] == 2 and prof["error_share"]["orientation"] == 1.0


def test_to_unified_bridges_into_udd(ann):
    u = to_unified(ann, base_dir=EX.parent)
    assert u.qas and not u.answers                 # grouped form only (flat XOR grouped)
    assert {f.key for f in u.fields} >= {"invoice.total", "invoice.date"}
    assert next(f for f in u.fields if f.key == "invoice.date").value == "2024-04-03"
    assert u.table_html.startswith("<table>") and "INVOICE" in u.full_text
    assert len(u.to_samples()) == len(ann.understanding.qa)


def test_cli_validate_and_export(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "annotate_dar", Path(__file__).resolve().parents[1] / "scripts" / "annotate_dar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["validate", str(EX)]) == 0
    assert mod.main(["export", str(EX), "--arm", "L1_+ocr", "--out", str(tmp_path)]) == 0
    rows = [json.loads(x) for x in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert {r["meta"]["layer"] for r in rows} == {"kie", "qa", "ocr"}
    bad = copy.deepcopy(json.loads(EX.read_text()))
    bad["kie"][0]["value_lines"] = ["nope"]
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    assert mod.main(["validate", str(tmp_path / "bad.json")]) == 1

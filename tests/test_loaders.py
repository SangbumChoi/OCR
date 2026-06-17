"""JSONL loader/saver round-trip."""

from docvlm_eval.benchmarks import load_jsonl, save_jsonl
from docvlm_eval.schema import Sample


def test_roundtrip(tmp_path):
    samples = [
        Sample("a", "img/a.png", "q1?", ["x", "y"], "t1", "anls", {"benchmark": "b"}),
        Sample("b", "img/b.png", "q2?", ["z"], "t2", "exact", {}),
    ]
    p = tmp_path / "x.jsonl"
    save_jsonl(samples, p)
    out = load_jsonl(p)
    assert len(out) == 2
    assert out[0].sample_id == "a"
    assert out[0].answers == ["x", "y"]
    assert out[0].metric == "anls"
    assert out[0].meta["benchmark"] == "b"
    assert out[1].answer_type == "t2"


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "y.jsonl"
    p.write_text(
        '{"sample_id":"a","image_path":"i","question":"q","answers":["1"]}\n\n'
        '{"sample_id":"b","image_path":"i","question":"q","answers":["2"]}\n',
        encoding="utf-8",
    )
    assert len(load_jsonl(p)) == 2


def test_answers_coerced_to_str(tmp_path):
    p = tmp_path / "z.jsonl"
    p.write_text('{"sample_id":"a","image_path":"i","question":"q","answers":[100, 2.5]}\n', encoding="utf-8")
    out = load_jsonl(p)
    assert out[0].answers == ["100", "2.5"]

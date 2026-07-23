import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tokenizers") is None,
    reason="student tokenizer tests require tokenizers",
)


def test_byte_level_student_tokenizer_round_trips_document_scripts(tmp_path):
    from docvlm_eval.student.tokenizer import DocumentTokenizer

    corpus = [
        "Invoice total: $12,345.67",
        "黃河入海流",
        "연구 문서의 합계는 42입니다.",
        r"\int_0^\infty e^{-x^2}\,dx",
        "<table><tr><td>Δ revenue</td></tr></table>",
    ]
    tokenizer = DocumentTokenizer.train(
        corpus,
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )

    for text in corpus:
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert tokenizer.vocab_size <= 300

    tokenizer.save_pretrained(tmp_path)
    loaded = DocumentTokenizer.from_pretrained(tmp_path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.decode(loaded.encode(corpus[2])) == corpus[2]
    assert loaded.encode("answer", add_special_tokens=True)[0] == loaded.bos_token_id
    assert loaded.encode("answer", add_special_tokens=True)[-1] == loaded.eos_token_id


def test_udd_text_iterator_includes_native_qa_and_structured_payload():
    import json

    from docvlm_eval.student.tokenizer import iter_udd_text

    rows = [
        {
            "instructions": ["What is the total?"],
            "answers": [["42", "forty two"]],
            "full_text": "TOTAL 42",
            "table_html": "<table><td>42</td></table>",
            "elements_json": json.dumps(
                [
                    {
                        "key": "total",
                        "value": "42",
                        "bbox": [0, 0, 1, 1, True],
                        "kind": "field",
                    }
                ]
            ),
        }
    ]

    texts = list(iter_udd_text(rows))

    assert "What is the total?" in texts
    assert "forty two" in texts
    assert "TOTAL 42" in texts
    assert "<table><td>42</td></table>" in texts
    assert "total" in texts


def test_collator_rejects_token_ids_outside_the_student_vocabulary():
    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        StudentExample,
    )

    class BadTokenizer:
        pad_token_id = 0
        bos_token_id = 1
        eos_token_id = 2

        @staticmethod
        def encode(text, add_special_tokens=False):
            del text, add_special_tokens
            return [999]

    collator = StudentCollator(
        BadTokenizer(),
        StudentCollatorConfig(max_length=16, vocab_size=256),
    )

    with pytest.raises(ValueError, match="outside student vocabulary"):
        collator(
            [
                StudentExample(
                    sample_id="bad",
                    source="test",
                    task="text",
                    prompt="Q",
                    answer="A",
                )
            ]
        )

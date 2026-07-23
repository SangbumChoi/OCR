from copy import deepcopy

from docvlm_eval.synth.dto import GenConfig
from docvlm_eval.synth.supervision import apply_supervision_toggles


def _ground_truth():
    return {
        "spotting": {"cell": [1, 2, 3, 4]},
        "qa": [
            {
                "question": "Where?",
                "answers": ["1,2,3,4;10,10"],
                "metric": "grounding",
                "box": [1, 2, 3, 4],
                "derived": True,
                "rationale": "Locate the cell.",
            },
            {
                "question": "Total?",
                "answers": ["5"],
                "metric": "relaxed_acc",
                "evidence_keys": ["cell"],
                "derived": True,
                "rationale": "Read and add.",
            },
            {
                "question": "Field?",
                "answers": ["x"],
                "metric": "anls",
                "rationale": "Read it.",
            },
        ],
        "semantic_graph": {
            "queries": [
                {
                    "resolved": {
                        "answer": "5",
                        "evidence_keys": ["cell"],
                        "rationale": "Read and add.",
                    }
                }
            ]
        },
    }


def test_spotting_off_removes_all_box_supervision_paths():
    gt = _ground_truth()
    apply_supervision_toggles(
        gt,
        GenConfig(emit_spotting=False, emit_rationale=True, emit_understanding=True),
    )
    assert "spotting" not in gt
    assert all(query["metric"] != "grounding" for query in gt["qa"])
    assert all("box" not in query and "evidence_keys" not in query for query in gt["qa"])
    assert "evidence_keys" not in gt["semantic_graph"]["queries"][0]["resolved"]


def test_rationale_off_redacts_legacy_and_graph_views():
    gt = _ground_truth()
    apply_supervision_toggles(
        gt,
        GenConfig(emit_spotting=True, emit_rationale=False, emit_understanding=True),
    )
    assert all("rationale" not in query for query in gt["qa"])
    assert gt["semantic_graph"]["supervision_redacted"] is True
    assert "nodes" not in gt["semantic_graph"]
    assert "edges" not in gt["semantic_graph"]
    assert "queries" not in gt["semantic_graph"]


def test_understanding_off_removes_derived_queries_and_graph_program():
    gt = deepcopy(_ground_truth())
    apply_supervision_toggles(
        gt,
        GenConfig(emit_spotting=True, emit_rationale=True, emit_understanding=False),
    )
    assert [query["question"] for query in gt["qa"]] == ["Field?"]
    assert "semantic_graph" not in gt

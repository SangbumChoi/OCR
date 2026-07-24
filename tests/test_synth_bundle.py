"""Cross-document bundles preserve independent source identity and exact boxes."""

from __future__ import annotations

from PIL import Image

from docvlm_eval.synth.bundle import BundleDocument, compose_document_bundle
from docvlm_eval.synth.dto import DocSample
from docvlm_eval.synth.to_samples import case_to_samples


def _document(
    document_id: str,
    *,
    size: tuple[int, int],
    value: str,
) -> BundleDocument:
    image = Image.new("RGB", size, "white")
    ground_truth = {
        "type": f"{document_id} source",
        "fields": {"value": value},
        "spotting": {"value": [2, 3, size[0] - 2, size[1] - 3]},
        "render": {
            "size_px": list(size),
            "page_count": 1,
            "rendered_page_count": 1,
            "page_origins_px": [[0, 0]],
            "page_sizes_px": [list(size)],
        },
    }
    return BundleDocument(document_id, image, ground_truth)


def test_grid_bundle_namespaces_keys_and_records_document_page_provenance():
    first = _document("filing", size=(40, 60), value="A")
    second = _document("market", size=(50, 50), value="B")
    third = _document("memo", size=(30, 70), value="C")

    image, ground_truth = compose_document_bundle(
        [first, second, third],
        mode="grid",
        gap_px=10,
        qa=[
            {
                "question": "Combine A and C.",
                "answers": ["AC"],
                "answer_type": "H-cross-document",
                "evidence_keys": ["filing.value", "memo.value"],
            }
        ],
    )

    render = ground_truth["render"]
    assert image.size == (100, 140)
    assert render["document_count"] == 3
    assert render["document_ids"] == ["filing", "market", "memo"]
    assert render["document_origins_px"] == [[0, 0], [50, 5], [5, 70]]
    assert render["page_document_indices"] == [0, 1, 2]
    assert render["page_document_ids"] == ["filing", "market", "memo"]
    assert ground_truth["fields"] == {
        "filing.value": "A",
        "market.value": "B",
        "memo.value": "C",
    }
    assert ground_truth["spotting"]["market.value"] == [52, 8, 98, 52]


def test_cross_document_evidence_reaches_eval_metadata():
    image, ground_truth = compose_document_bundle(
        [
            _document("left", size=(40, 60), value="A"),
            _document("right", size=(40, 60), value="B"),
        ],
        mode="vertical",
        gap_px=8,
        qa=[
            {
                "question": "Compare the values.",
                "answers": ["different"],
                "answer_type": "H-cross-document",
                "evidence_keys": ["left.value", "right.value"],
            }
        ],
    )
    structured = DocSample.from_builder_gt(ground_truth).to_dict()
    samples = case_to_samples(
        structured,
        "bundle.png",
        "bundle",
        render_variant="clean",
    )
    query = next(
        sample
        for sample in samples
        if sample.answer_type == "H-cross-document"
    )

    assert image.size == (40, 128)
    assert query.meta["document_count"] == 2
    assert query.meta["evidence_documents"] == [0, 1]
    assert query.meta["cross_document_evidence"] is True
    assert query.meta["evidence_pages"] == [0, 1]
    assert query.meta["cross_page_evidence"] is True


def test_bundle_rejects_unknown_cross_document_evidence_key():
    first = _document("filing", size=(40, 60), value="A")

    try:
        compose_document_bundle(
            [first],
            qa=[
                {
                    "question": "Missing?",
                    "answers": ["x"],
                    "evidence_keys": ["memo.missing"],
                }
            ],
        )
    except ValueError as error:
        assert "unknown evidence keys" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unknown evidence key was accepted")


def test_bundle_namespaces_required_spotting_keys():
    source = _document("filing", size=(40, 60), value="A")
    source = BundleDocument(
        source.document_id,
        source.image,
        source.ground_truth,
        required_spotting_keys=("value", "missing"),
    )

    _, ground_truth = compose_document_bundle([source])

    assert ground_truth["render"]["required_spotting_keys"] == [
        "filing.missing",
        "filing.value",
    ]

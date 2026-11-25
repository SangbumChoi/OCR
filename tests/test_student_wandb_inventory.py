from __future__ import annotations

import copy

import pytest

from docvlm_eval.student.wandb_inventory import (
    build_wandb_run_inventory,
    wandb_run_inventory_valid,
)


def _records():
    return [
        {
            "id": "run-b",
            "name": "docvlm-smol-vision-transfer-pilot--candidate",
            "state": "running",
            "created_at": "2026-07-25T10:00:00+00:00",
            "updated_at": "2026-07-25T10:02:00+00:00",
            "summary": {"large_metric_table": list(range(10_000))},
        },
        {
            "id": "run-a",
            "name": "legacy",
            "state": "finished",
            "created_at": "2026-07-24T10:00:00+00:00",
            "updated_at": "2026-07-24T11:00:00+00:00",
        },
    ]


def test_wandb_inventory_is_compact_sorted_and_fingerprinted():
    result = build_wandb_run_inventory(
        _records(),
        entity="sbdc",
        project="docvlm-ablation",
        observed_at="2026-07-25T12:00:00+00:00",
    )

    assert result["run_count"] == 2
    assert result["states"] == {"finished": 1, "running": 1}
    assert [run["id"] for run in result["runs"]] == ["run-b", "run-a"]
    assert "summary" not in result["runs"][0]
    assert wandb_run_inventory_valid(result)


def test_wandb_inventory_detects_tampering():
    result = build_wandb_run_inventory(
        _records(),
        entity="sbdc",
        project="docvlm-ablation",
        observed_at="2026-07-25T12:00:00+00:00",
    )
    tampered = copy.deepcopy(result)
    tampered["runs"][0]["state"] = "finished"

    assert wandb_run_inventory_valid(tampered) is False


def test_wandb_inventory_rejects_duplicate_ids():
    records = _records()
    records[1]["id"] = records[0]["id"]

    with pytest.raises(ValueError, match="duplicate W&B run IDs"):
        build_wandb_run_inventory(
            records,
            entity="sbdc",
            project="docvlm-ablation",
            observed_at="2026-07-25T12:00:00+00:00",
        )


def test_wandb_inventory_rejects_empty_source_context():
    with pytest.raises(ValueError, match="context is empty"):
        build_wandb_run_inventory(
            _records(),
            entity="",
            project="docvlm-ablation",
            observed_at="2026-07-25T12:00:00+00:00",
        )

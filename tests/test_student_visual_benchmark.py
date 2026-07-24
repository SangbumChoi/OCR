from __future__ import annotations

import os
import json

import pytest
import torch

from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.visual_benchmark import (
    VisualBenchmarkConfig,
    run_visual_backend_benchmark,
)


def _run(**overrides):
    values = {
        "sequence_lengths": (3, 5),
        "backends": ("loop", "auto", "flex"),
        "warmup_iterations": 0,
        "measured_iterations": 1,
        "mode": "forward",
        "precision": "float32",
        "device": "cpu",
        "seed": 11,
    }
    values.update(overrides)
    return run_visual_backend_benchmark(
        StudentConfig.tiny(),
        VisualBenchmarkConfig(**values),
    )


def test_cpu_benchmark_records_fallback_and_explicit_flex_error():
    report = _run()

    assert report["scope"] == "student_vision_tower_and_gated_resampler"
    assert report["language_decoder_included"] is False
    assert report["visual_tokens"] == 8
    assert report["batch_size"] == 2
    records = {record["requested_backend"]: record for record in report["results"]}
    assert records["loop"]["status"] == "ok"
    assert records["loop"]["resolved_backend"] == "loop"
    assert records["loop"]["max_abs_delta_vs_loop"] == pytest.approx(0.0)
    assert records["loop"]["median_speedup_vs_loop"] == pytest.approx(1.0)
    assert records["auto"]["status"] == "ok"
    assert records["auto"]["resolved_backend"] == "loop"
    assert records["auto"]["max_abs_delta_vs_loop"] == pytest.approx(0.0)
    assert records["flex"]["status"] == "error"
    assert records["flex"]["error_type"] == "RuntimeError"
    assert report["gates"] == {
        "require_flex": False,
        "flex_resolved": False,
        "passed": True,
    }
    json.dumps(report)


def test_require_flex_gate_preserves_report_on_cpu():
    report = _run(backends=("auto",), require_flex=True)

    assert report["results"][0]["status"] == "ok"
    assert report["results"][0]["resolved_backend"] == "loop"
    assert report["gates"] == {
        "require_flex": True,
        "flex_resolved": False,
        "passed": False,
    }


def test_training_mode_measures_forward_and_backward():
    report = _run(
        backends=("loop",),
        mode="training",
        warmup_iterations=1,
    )

    record = report["results"][0]
    assert record["status"] == "ok"
    assert record["median_ms"] > 0
    assert record["tokens_per_second"] > 0


def test_dense_policies_share_patch_content_and_match_packed_output():
    report = _run(
        sequence_lengths=None,
        patch_grids=((1, 3), (3, 1)),
        backends=("loop", "dense_adaptive", "dense_fixed_square"),
    )

    records = {
        record["requested_backend"]: record for record in report["results"]
    }
    assert report["patch_grids"] == [[1, 3], [3, 1]]
    assert records["dense_adaptive"]["status"] == "ok"
    assert records["dense_fixed_square"]["status"] == "ok"
    assert records["dense_adaptive"]["resolved_backend"] == "dense"
    assert records["dense_adaptive"]["max_abs_delta_vs_loop"] < 1e-5
    assert records["dense_fixed_square"]["max_abs_delta_vs_loop"] < 1e-5
    assert records["loop"]["executed_visual_tokens"] == 6
    assert records["dense_adaptive"]["executed_visual_tokens"] == 18
    assert records["dense_fixed_square"]["executed_visual_tokens"] == 128
    assert records["loop"]["median_speedup_vs_dense_adaptive"] > 0


def test_rejects_visual_sequence_above_position_grid():
    with pytest.raises(ValueError, match="max_position_tokens"):
        _run(sequence_lengths=(65,), backends=("loop",))


def test_rejects_dense_policy_without_patch_grids():
    with pytest.raises(ValueError, match="require patch_grids"):
        _run(backends=("loop", "dense_adaptive"))


def test_rejects_both_sequence_lengths_and_patch_grids():
    with pytest.raises(ValueError, match="only one"):
        _run(patch_grids=((1, 3), (1, 5)))


@pytest.mark.skipif(
    os.environ.get("DOCVLM_RUN_FLEX_INTEGRATION") != "1" or not torch.cuda.is_available(),
    reason="set DOCVLM_RUN_FLEX_INTEGRATION=1 on a supported CUDA host",
)
def test_require_flex_cuda_smoke():
    report = _run(
        device="cuda",
        precision="float16",
        backends=("auto", "flex"),
        require_flex=True,
        warmup_iterations=1,
    )

    assert report["gates"]["passed"] is True

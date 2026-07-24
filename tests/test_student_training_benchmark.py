from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.pretrain import ContrastiveMemoryConfig
from docvlm_eval.student.training_benchmark import (
    TrainingBenchmarkConfig,
    run_training_feasibility_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cpu_training_probe_executes_full_optimizer_step():
    report = run_training_feasibility_benchmark(
        StudentConfig.tiny(),
        TrainingBenchmarkConfig(
            patch_grid=(1, 2),
            text_tokens=8,
            micro_batch_size=1,
            warmup_steps=0,
            measured_steps=1,
            packed_attention_backend="loop",
            precision="float32",
            gradient_checkpointing=True,
            device="cpu",
            seed=13,
        ),
        loss_weights={
            "autoregressive": 1.0,
            "region_text_contrastive": 0.15,
            "box_regression": 0.2,
            "orientation": 0.05,
        },
        learning_rate=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        grad_accum_steps=2,
        max_grad_norm=1.0,
        contrastive=True,
        box_iou_loss="ciou",
        contrastive_memory=ContrastiveMemoryConfig(
            enabled=True,
            size=2,
            min_negatives=1,
        ),
    )

    assert report["schema_version"] == 1
    assert report["scope"] == "full_student_multimodal_training_step"
    assert report["benchmark_config"]["box_iou_loss"] == "ciou"
    assert report["benchmark_config"]["contrastive_memory"]["enabled"] is True
    assert report["status"] == "ok"
    assert report["resolved_visual_attention_backend"] == "loop"
    assert report["gradient_checkpointing"] == {
        "enabled": True,
        "components": ["vision", "connector", "language"],
        "use_reentrant": False,
    }
    assert report["training_flops_per_microbatch"]["algorithmic"] > 0
    assert (
        report["training_flops_per_microbatch"]["checkpoint_recompute"]
        > 0
    )
    assert report["training_flops_per_microbatch"]["executed"] == (
        report["training_flops_per_microbatch"]["algorithmic"]
        + report["training_flops_per_microbatch"][
            "checkpoint_recompute"
        ]
    )
    assert report["all_finite"] is True
    assert report["all_optimizer_steps_succeeded"] is True
    assert report["optimizer_state"]["parameter_states"] > 0
    assert report["optimizer_state"]["tensor_bytes"] > 0
    assert report["optimizer_state"]["max_step"] == 1
    assert report["median_step_ms"] > 0
    measured = report["measured_steps"][0]
    assert measured["contrastive_memory_size"] == 2
    assert measured["contrastive_negative_pairs"] == 2
    assert measured["contrastive_additional_flops"] > 0
    assert report["effective_peak_memory"] is None
    json.dumps(report)


def test_blocking_training_preflight_rejects_cpu_before_model_allocation():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_student_training_step.py"),
            "--require-deployment-gate",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "requires an available CUDA device" in completed.stderr

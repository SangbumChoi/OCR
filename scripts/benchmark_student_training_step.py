#!/usr/bin/env python3
"""Benchmark one production-shaped full-student training micro-step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.gates import (
    evaluate_training_feasibility_gate,
    write_gate_report,
)
from docvlm_eval.student.optim import OptimizerSpec
from docvlm_eval.student.pretrain import ContrastiveMemoryConfig
from docvlm_eval.student.training_benchmark import (
    TrainingBenchmarkConfig,
    run_training_feasibility_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def _patch_grid(value: str) -> tuple[int, int]:
    try:
        dimensions = tuple(
            int(item.strip()) for item in value.lower().split("x")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected HEIGHTxWIDTH") from error
    if len(dimensions) != 2 or any(value <= 0 for value in dimensions):
        raise argparse.ArgumentTypeError(
            "patch grid must be a positive HEIGHTxWIDTH pair"
        )
    return dimensions


def _wandb_log(args: argparse.Namespace, report: dict[str, Any]) -> None:
    if not args.wandb_project:
        return
    try:
        import wandb
    except ImportError as error:
        raise SystemExit("install wandb or omit --wandb-project") from error
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        group=args.wandb_group,
        tags=args.wandb_tags,
        config={
            "architecture_config": str(args.config),
            **report["benchmark_config"],
            "student_config_fingerprint": report[
                "student_config_fingerprint"
            ],
            "environment": report["environment"],
        },
    )
    payload: dict[str, Any] = {
        "training_feasibility/success": float(report["status"] == "ok"),
        "training_feasibility/oom": float(bool(report["oom"])),
        "training_feasibility/gate_pass": float(
            report["deployment_gate"]["status"] == "pass"
        ),
    }
    for key in (
        "parameter_count",
        "median_step_ms",
        "p95_step_ms",
        "steps_per_second",
    ):
        if report.get(key) is not None:
            payload[f"training_feasibility/{key}"] = report[key]
    memory = report.get("effective_peak_memory") or {}
    for key in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_reserved_fraction",
        "free_bytes",
        "total_bytes",
    ):
        if memory.get(key) is not None:
            payload[f"training_feasibility/{key}"] = memory[key]
    optimizer_state = report.get("optimizer_state") or {}
    for key in ("parameter_states", "tensor_bytes", "max_step"):
        if optimizer_state.get(key) is not None:
            payload[f"training_feasibility/optimizer_{key}"] = (
                optimizer_state[key]
            )
    training_flops = report.get("training_flops_per_microbatch") or {}
    for key in ("algorithmic", "checkpoint_recompute", "executed"):
        if training_flops.get(key) is not None:
            payload[
                f"training_feasibility/{key}_flops_per_microbatch"
            ] = training_flops[key]
    measured = report.get("measured_steps") or []
    if measured:
        for key in (
            "contrastive_memory_size",
            "contrastive_negative_pairs",
            "contrastive_additional_flops",
        ):
            if measured[-1].get(key) is not None:
                payload[f"training_feasibility/{key}"] = measured[-1][key]
    run.log(payload)
    run.summary["training_feasibility_report"] = report
    run.summary["training_feasibility_gate_status"] = report[
        "deployment_gate"
    ]["status"]
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--patch-grid", type=_patch_grid, default=(40, 63))
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument(
        "--packed-attention-backend",
        choices=["loop", "auto", "flex"],
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--require-deployment-gate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*")
    args = parser.parse_args()
    if args.require_deployment_gate and (
        not torch.cuda.is_available()
        or (
            args.device != "auto"
            and torch.device(args.device).type != "cuda"
        )
    ):
        raise SystemExit(
            "--require-deployment-gate requires an available CUDA device"
        )

    blueprint = load_blueprint(args.config)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    pretraining = blueprint["training"]["pretraining"]
    pipeline = pretraining["input_pipeline"]
    optimizer = pretraining["optimizer"]
    checkpointing = blueprint["training"]["activation_checkpointing"]
    config = TrainingBenchmarkConfig(
        patch_grid=args.patch_grid,
        text_tokens=(
            args.text_tokens
            if args.text_tokens is not None
            else int(pipeline["max_text_tokens"])
        ),
        micro_batch_size=(
            args.micro_batch_size
            if args.micro_batch_size is not None
            else int(optimizer["micro_batch_size"])
        ),
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        packed_attention_backend=(
            args.packed_attention_backend
            or str(pipeline["packed_attention_backend"])
        ),
        precision=args.precision or str(optimizer["precision"]),
        gradient_checkpointing=bool(checkpointing["enabled"]),
        gradient_checkpointing_components=tuple(
            str(value) for value in checkpointing["components"]
        ),
        gradient_checkpointing_use_reentrant=bool(
            checkpointing["use_reentrant"]
        ),
        device=args.device,
        seed=args.seed if args.seed is not None else int(optimizer["seed"]),
    )
    report = run_training_feasibility_benchmark(
        StudentConfig.from_blueprint(blueprint),
        config,
        optimizer_spec=OptimizerSpec.from_mapping(optimizer),
        loss_weights={
            str(name): float(weight)
            for name, weight in pretraining["losses"].items()
        },
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        grad_accum_steps=int(optimizer["grad_accum_steps"]),
        max_grad_norm=float(optimizer["max_grad_norm"]),
        contrastive=bool(pipeline["contrastive"]),
        box_iou_loss=str(pretraining.get("box_iou_loss", "giou")),
        contrastive_memory=ContrastiveMemoryConfig.from_blueprint(
            blueprint
        ),
    )
    report["deployment_gate"] = evaluate_training_feasibility_gate(
        blueprint,
        report,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        write_gate_report(args.output, report)
        print(f"[training-feasibility] wrote {args.output}")
    _wandb_log(args, report)
    if (
        args.require_deployment_gate
        and report["deployment_gate"]["status"] != "pass"
    ):
        raise SystemExit(
            "training feasibility deployment gate "
            f"{report['deployment_gate']['status']}: "
            f"{report['deployment_gate']['reason']}"
        )
    if report["status"] != "ok":
        raise SystemExit(
            f"training feasibility benchmark failed: "
            f"{report['error_type']}: {report['error']}"
        )


if __name__ == "__main__":
    main()

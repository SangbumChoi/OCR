#!/usr/bin/env python3
"""Benchmark packed and dense policies in the native student's vision path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.gates import evaluate_visual_efficiency_gate
from docvlm_eval.student.visual_benchmark import (
    VisualBenchmarkConfig,
    run_visual_backend_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def _sequence_lengths(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(length <= 0 for length in result):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return result


def _patch_grids(value: str) -> tuple[tuple[int, int], ...]:
    try:
        result = tuple(
            tuple(int(dimension.strip()) for dimension in item.lower().split("x"))
            for item in value.split(",")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated HEIGHTxWIDTH grids"
        ) from error
    if (
        not result
        or any(len(grid) != 2 for grid in result)
        or any(dimension <= 0 for grid in result for dimension in grid)
    ):
        raise argparse.ArgumentTypeError("patch grids must be positive HEIGHTxWIDTH pairs")
    return result


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
            "student_config_fingerprint": report["student_config_fingerprint"],
            "environment": report["environment"],
        },
    )
    payload: dict[str, Any] = {
        "visual_benchmark/visual_tokens": report["visual_tokens"],
        "visual_benchmark/batch_size": report["batch_size"],
        "visual_benchmark/rounds": report["benchmark_config"]["rounds"],
        "visual_benchmark/deployment_gate_pass": float(
            report["deployment_gate"]["status"] == "pass"
        ),
    }
    for record in report["results"]:
        backend = record["requested_backend"]
        payload[f"visual_benchmark/{backend}/success"] = float(record["status"] == "ok")
        if record["status"] != "ok":
            continue
        for metric in (
            "mean_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "tokens_per_second",
            "max_abs_delta_vs_loop",
            "median_speedup_vs_loop",
            "min_speedup_vs_loop",
            "peak_memory_ratio_vs_loop",
            "peak_memory_reduction_fraction_vs_loop",
            "median_speedup_vs_dense_adaptive",
            "min_speedup_vs_dense_adaptive",
            "peak_memory_ratio_vs_dense_adaptive",
            "peak_memory_reduction_fraction_vs_dense_adaptive",
            "rounds",
            "peak_memory_allocated_bytes",
            "peak_memory_reserved_bytes",
            "executed_visual_tokens",
            "valid_visual_token_fraction",
        ):
            value = record.get(metric)
            if value is not None:
                payload[f"visual_benchmark/{backend}/{metric}"] = value
        payload[f"visual_benchmark/{backend}/resolved_flex"] = float(
            record["resolved_backend"] == "flex"
        )
    run.log(payload)
    run.summary["visual_backend_report"] = report
    run.summary["visual_backend_gate_status"] = report[
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
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument(
        "--sequence-lengths",
        type=_sequence_lengths,
        help="Comma-separated unpadded visual-token counts, one per document.",
    )
    shapes.add_argument(
        "--patch-grids",
        type=_patch_grids,
        help="Comma-separated HEIGHTxWIDTH patch grids, one per document.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "loop",
            "auto",
            "flex",
            "dense_adaptive",
            "dense_fixed_square",
        ],
        default=[
            "loop",
            "auto",
            "flex",
            "dense_adaptive",
            "dense_fixed_square",
        ],
    )
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mode", choices=["forward", "training"], default="training")
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--parity-atol", type=float)
    parser.add_argument(
        "--require-flex",
        action="store_true",
        help="Exit nonzero unless every requested auto/flex run resolves to FlexAttention.",
    )
    parser.add_argument(
        "--require-deployment-gate",
        action="store_true",
        help="Exit nonzero unless the complete visual_efficiency gate passes.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*")
    args = parser.parse_args()
    if args.sequence_lengths is None and args.patch_grids is None:
        args.patch_grids = ((40, 63), (63, 40))
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
    config = VisualBenchmarkConfig(
        sequence_lengths=args.sequence_lengths,
        patch_grids=args.patch_grids,
        backends=tuple(args.backends),
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.iterations,
        rounds=args.rounds,
        mode=args.mode,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        require_flex=args.require_flex,
        parity_atol=args.parity_atol,
    )
    try:
        report = run_visual_backend_benchmark(
            StudentConfig.from_blueprint(blueprint),
            config,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    report["deployment_gate"] = evaluate_visual_efficiency_gate(
        blueprint,
        report,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[visual-benchmark] wrote {args.output}")
    _wandb_log(args, report)
    if (
        args.require_deployment_gate
        and report["deployment_gate"]["status"] != "pass"
    ):
        raise SystemExit(
            "visual efficiency deployment gate "
            f"{report['deployment_gate']['status']}: "
            f"{report['deployment_gate']['reason']}"
        )
    if not report["gates"]["passed"]:
        raise SystemExit(
            "FlexAttention gate failed: every requested auto/flex run must resolve to flex"
        )


if __name__ == "__main__":
    main()

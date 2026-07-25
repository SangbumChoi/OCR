#!/usr/bin/env python3
"""Evaluate a native student checkpoint on one or more structured splits."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.benchmarks import load_jsonl
from docvlm_eval.student.data import StudentCollator, StudentCollatorConfig
from docvlm_eval.student.evaluate import (
    StructuredEvalConfig,
    apply_temperature_calibration,
    compare_split_summaries,
    evaluate_structured_student,
    partition_calibration_samples,
    wandb_metrics_for_split,
    write_split_comparison,
)
from docvlm_eval.student.gates import (
    evaluate_deployment_gates,
    load_evaluation_artifacts,
    load_training_feasibility_report,
    load_visual_backend_report,
    write_gate_report,
)
from docvlm_eval.student.model import DocumentVLMStudent
from docvlm_eval.student.model import count_unique_parameters
from docvlm_eval.student.posttrain import StructuredPostTrainingDataset
from docvlm_eval.student.rewards import RewardConfig
from docvlm_eval.student.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parents[1]
_SPLIT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_splits(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--split must be NAME=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        if not _SPLIT_NAME.fullmatch(name):
            raise SystemExit(f"invalid split name {name!r}")
        if name in seen:
            raise SystemExit(f"duplicate split name {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"split file does not exist: {path}")
        seen.add(name)
        parsed.append((name, path))
    return parsed


def _parse_answer_type_token_budgets(
    values: list[str] | None,
) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for value in values or []:
        if "=" not in value:
            raise SystemExit(
                "--answer-type-token-budget must be PATTERN=TOKENS"
            )
        pattern, raw_budget = value.rsplit("=", 1)
        try:
            budget = int(raw_budget)
        except ValueError as error:
            raise SystemExit(
                "--answer-type-token-budget TOKENS must be an integer"
            ) from error
        parsed.append((pattern, budget))
    return tuple(parsed)


def _metadata(checkpoint: Path) -> dict:
    path = checkpoint / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _checkpoint_step(metadata: dict) -> int:
    state = metadata.get("trainer_state")
    if not isinstance(state, dict):
        return 0
    for key in ("global_step", "rollout_step", "optimizer_step"):
        if key in state:
            return int(state[key])
    return 0


def _device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _start_wandb(args, metadata: dict, split_paths: list[tuple[str, Path]]):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("install wandb or omit --wandb-project") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run,
        group=args.wandb_group,
        tags=args.wandb_tags,
        id=args.wandb_id,
        resume="allow" if args.wandb_id else None,
        config={
            "checkpoint": str(args.checkpoint),
            "checkpoint_stage": metadata.get("run_stage"),
            "tokenizer": str(args.tokenizer),
            "splits": {name: str(path) for name, path in split_paths},
            "max_new_tokens": args.max_new_tokens,
            "max_samples": args.max_samples,
            "use_kv_cache": not args.no_kv_cache,
            "repetition_guard": {
                "min_tokens": args.repetition_guard_min_tokens,
                "max_period": args.repetition_guard_max_period,
                "repetitions": args.repetition_guard_repetitions,
            },
            "precision": args.precision,
            "temperature_calibration": {
                "enabled": not args.no_temperature_calibration,
                "source_split": args.calibration_source_split,
                "fraction": args.calibration_fraction,
                "min_samples": args.calibration_min_samples,
                "correct_threshold": args.calibration_correct_threshold,
                "temperature_bounds": [
                    args.calibration_min_temperature,
                    args.calibration_max_temperature,
                ],
                "seed": args.calibration_seed,
            },
        },
    )
    wandb.define_metric("evaluation/checkpoint_step")
    for namespace in (
        "eval/*",
        "eval_by_axis/*",
        "eval_by_source/*",
        "eval_by_language/*",
        "eval_reward/*",
        "gate/*",
    ):
        wandb.define_metric(
            namespace,
            step_metric="evaluation/checkpoint_step",
        )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Repeat for train, heldout, or other JSONL splits.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Native checkpoint student/ directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens-hard-cap", type=int)
    parser.add_argument(
        "--answer-type-token-budget",
        action="append",
        metavar="PATTERN=TOKENS",
        help="Repeat for exact labels or trailing-wildcard task prefixes.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--repetition-guard-min-tokens", type=int, default=24)
    parser.add_argument("--repetition-guard-max-period", type=int, default=16)
    parser.add_argument("--repetition-guard-repetitions", type=int, default=3)
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Recompute the full language prefix for generation ablations.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-temperature-calibration", action="store_true")
    parser.add_argument("--calibration-source-split", default="heldout")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-min-samples", type=int, default=20)
    parser.add_argument(
        "--calibration-correct-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--calibration-min-temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--calibration-max-temperature",
        type=float,
        default=20.0,
    )
    parser.add_argument("--calibration-seed", type=int, default=47)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--wandb-id")
    parser.add_argument(
        "--baseline-evaluation",
        type=Path,
        help="Reference-checkpoint evaluation root for improvement gates.",
    )
    parser.add_argument(
        "--monolingual-control-evaluation",
        type=Path,
        help="Evaluation root containing per-language monolingual controls.",
    )
    parser.add_argument(
        "--visual-backend-benchmark",
        type=Path,
        help="Target-device JSON from benchmark_student_visual_backend.py.",
    )
    parser.add_argument(
        "--training-feasibility-benchmark",
        type=Path,
        help="Target-device JSON from benchmark_student_training_step.py.",
    )
    args = parser.parse_args()

    split_paths = _parse_splits(args.split)
    answer_type_token_budgets = _parse_answer_type_token_budgets(
        args.answer_type_token_budget
    )
    samples_by_split = {
        split_name: load_jsonl(path)
        for split_name, path in split_paths
    }
    calibration_enabled = not args.no_temperature_calibration
    if calibration_enabled:
        if not 0.0 <= args.calibration_correct_threshold <= 1.0:
            raise SystemExit(
                "calibration correct threshold must be within [0, 1]"
            )
        if not (
            0.0
            < args.calibration_min_temperature
            < args.calibration_max_temperature
        ):
            raise SystemExit(
                "calibration temperature bounds must be positive and ordered"
            )
        if args.calibration_seed < 0:
            raise SystemExit("calibration seed must be non-negative")
        source_split = args.calibration_source_split
        if source_split == "calibration":
            raise SystemExit("calibration source split cannot be named calibration")
        if "calibration" in samples_by_split:
            raise SystemExit(
                "temperature calibration reserves the split name 'calibration'"
            )
        if source_split not in samples_by_split:
            raise SystemExit(
                f"calibration source split {source_split!r} was not provided"
            )
        try:
            calibration_samples, evaluation_samples = (
                partition_calibration_samples(
                    samples_by_split[source_split],
                    fraction=args.calibration_fraction,
                    min_samples=args.calibration_min_samples,
                    seed=args.calibration_seed,
                )
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        samples_by_split[source_split] = evaluation_samples
        ordered_samples: dict[str, list] = {}
        for split_name in [name for name, _ in split_paths]:
            if split_name == source_split:
                ordered_samples["calibration"] = calibration_samples
            ordered_samples[split_name] = samples_by_split[split_name]
        samples_by_split = ordered_samples
    blueprint = load_blueprint(args.config)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    tokenizer = DocumentTokenizer.from_pretrained(args.tokenizer)
    metadata = _metadata(args.checkpoint)
    fingerprint = metadata.get("tokenizer_fingerprint")
    if fingerprint is not None and fingerprint != tokenizer.fingerprint:
        raise SystemExit("checkpoint tokenizer fingerprint does not match --tokenizer")
    device = _device(args.device)
    collator = StudentCollator(
        tokenizer,
        replace(
            StudentCollatorConfig.from_blueprint(blueprint),
            rotation_probability=0.0,
            contrastive=False,
        ),
    )
    model = DocumentVLMStudent.from_pretrained(
        args.checkpoint,
        map_location=device,
    )
    reward_config = RewardConfig.from_blueprint(blueprint)
    base_config = StructuredEvalConfig(
        output_dir=str(args.output),
        max_new_tokens=args.max_new_tokens,
        max_new_tokens_hard_cap=(
            args.max_new_tokens
            if args.max_new_tokens_hard_cap is None
            else args.max_new_tokens_hard_cap
        ),
        max_new_tokens_by_answer_type=answer_type_token_budgets,
        max_samples=args.max_samples,
        use_kv_cache=not args.no_kv_cache,
        precision=args.precision,
        device=device,
        seed=args.seed,
        repetition_guard_min_tokens=args.repetition_guard_min_tokens,
        repetition_guard_max_period=args.repetition_guard_max_period,
        repetition_guard_repetitions=args.repetition_guard_repetitions,
    )
    run = _start_wandb(args, metadata, split_paths)
    summaries: dict[str, dict] = {}
    rows_by_split: dict[str, list[dict]] = {}
    wandb_payload: dict[str, float] = {
        "evaluation/checkpoint_step": float(_checkpoint_step(metadata))
    }
    try:
        for split_name, samples in samples_by_split.items():
            dataset = StructuredPostTrainingDataset(
                samples,
                target_mode="evidence_linked",
            )
            result = evaluate_structured_student(
                model,
                dataset,
                collator,
                tokenizer,
                replace(
                    base_config,
                    output_dir=str(args.output / split_name),
                ),
                reward_config,
                split_name=split_name,
            )
            summaries[split_name] = result.summary
            rows_by_split[split_name] = result.per_sample
            print(
                f"[student-eval:{split_name}] score={result.summary['score']:.4f} "
                f"reward={result.summary['reward']:.4f} "
                f"valid={result.summary['valid_structure_fraction']:.4f}",
                flush=True,
            )
        calibration_artifact = None
        if calibration_enabled:
            calibration_artifact = apply_temperature_calibration(
                rows_by_split,
                summaries,
                fit_split="calibration",
                output_dir=args.output,
                correct_threshold=args.calibration_correct_threshold,
                min_samples=args.calibration_min_samples,
                min_temperature=args.calibration_min_temperature,
                max_temperature=args.calibration_max_temperature,
                partition={
                    "source_split": args.calibration_source_split,
                    "fit_split": "calibration",
                    "fraction": args.calibration_fraction,
                    "min_samples": args.calibration_min_samples,
                    "seed": args.calibration_seed,
                    "source_samples": (
                        len(samples_by_split["calibration"])
                        + len(
                            samples_by_split[
                                args.calibration_source_split
                            ]
                        )
                    ),
                    "fit_samples": len(samples_by_split["calibration"]),
                    "evaluation_samples": len(
                        samples_by_split[args.calibration_source_split]
                    ),
                },
            )
        for split_name, summary in summaries.items():
            wandb_payload.update(
                wandb_metrics_for_split(summary, split_name)
            )
        comparison_path = write_split_comparison(args.output, summaries)
        comparison = compare_split_summaries(summaries)
        baseline_comparison = None
        baseline_rows = None
        if args.baseline_evaluation is not None:
            baseline_comparison, baseline_rows = load_evaluation_artifacts(
                args.baseline_evaluation
            )
        monolingual_comparison = None
        if args.monolingual_control_evaluation is not None:
            monolingual_comparison, _ = load_evaluation_artifacts(
                args.monolingual_control_evaluation
            )
        visual_backend_report = None
        if args.visual_backend_benchmark is not None:
            visual_backend_report = load_visual_backend_report(
                args.visual_backend_benchmark
            )
        training_feasibility_report = None
        if args.training_feasibility_benchmark is not None:
            training_feasibility_report = load_training_feasibility_report(
                args.training_feasibility_benchmark
            )
        gate_report = evaluate_deployment_gates(
            blueprint,
            count_unique_parameters(model),
            comparison,
            rows_by_split,
            baseline_comparison=baseline_comparison,
            baseline_rows=baseline_rows,
            monolingual_control_comparison=monolingual_comparison,
            visual_backend_report=visual_backend_report,
            training_feasibility_report=training_feasibility_report,
        )
        gate_path = write_gate_report(args.output / "gates.json", gate_report)
        manifest = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_metadata": metadata,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "evaluation": asdict(base_config),
            "temperature_calibration": calibration_artifact,
            "comparison": str(comparison_path),
            "gates": str(gate_path),
            "baseline_evaluation": (
                str(args.baseline_evaluation)
                if args.baseline_evaluation is not None
                else None
            ),
            "monolingual_control_evaluation": (
                str(args.monolingual_control_evaluation)
                if args.monolingual_control_evaluation is not None
                else None
            ),
            "visual_backend_benchmark": (
                str(args.visual_backend_benchmark)
                if args.visual_backend_benchmark is not None
                else None
            ),
            "training_feasibility_benchmark": (
                str(args.training_feasibility_benchmark)
                if args.training_feasibility_benchmark is not None
                else None
            ),
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if run is not None:
            for gate in gate_report["gates"]:
                if gate["status"] != "insufficient_evidence":
                    wandb_payload[f"gate/{gate['id']}"] = float(
                        gate["status"] == "pass"
                    )
            run.log(wandb_payload)
        print(
            "[student-eval:gates] "
            f"overall={gate_report['overall_status']} "
            f"pass={gate_report['counts']['pass']} "
            f"fail={gate_report['counts']['fail']} "
            f"insufficient={gate_report['counts']['insufficient_evidence']}",
            flush=True,
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()

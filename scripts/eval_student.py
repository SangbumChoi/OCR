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
    evaluate_structured_student,
    wandb_metrics_for_split,
    write_split_comparison,
)
from docvlm_eval.student.model import DocumentVLMStudent
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
        config={
            "checkpoint": str(args.checkpoint),
            "checkpoint_stage": metadata.get("run_stage"),
            "tokenizer": str(args.tokenizer),
            "splits": {name: str(path) for name, path in split_paths},
            "max_new_tokens": args.max_new_tokens,
            "max_samples": args.max_samples,
            "precision": args.precision,
        },
    )
    wandb.define_metric("evaluation/checkpoint_step")
    for namespace in (
        "eval/*",
        "eval_by_axis/*",
        "eval_by_source/*",
        "eval_by_language/*",
        "eval_reward/*",
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
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--precision",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*")
    args = parser.parse_args()

    split_paths = _parse_splits(args.split)
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
        max_samples=args.max_samples,
        precision=args.precision,
        device=device,
        seed=args.seed,
    )
    run = _start_wandb(args, metadata, split_paths)
    summaries: dict[str, dict] = {}
    wandb_payload: dict[str, float] = {
        "evaluation/checkpoint_step": float(_checkpoint_step(metadata))
    }
    try:
        for split_name, path in split_paths:
            dataset = StructuredPostTrainingDataset(
                load_jsonl(path),
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
            wandb_payload.update(
                wandb_metrics_for_split(result.summary, split_name)
            )
            print(
                f"[student-eval:{split_name}] score={result.summary['score']:.4f} "
                f"reward={result.summary['reward']:.4f} "
                f"valid={result.summary['valid_structure_fraction']:.4f}",
                flush=True,
            )
        comparison_path = write_split_comparison(args.output, summaries)
        manifest = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_metadata": metadata,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "evaluation": asdict(base_config),
            "comparison": str(comparison_path),
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if run is not None:
            run.log(wandb_payload)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()

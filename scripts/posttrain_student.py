#!/usr/bin/env python3
"""Run structured SFT, verifier-ranked preferences, or RLVR."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.benchmarks import load_jsonl
from docvlm_eval.student.data import StudentCollator, StudentCollatorConfig
from docvlm_eval.student.model import DocumentVLMStudent
from docvlm_eval.student.posttrain import (
    PreferenceConfig,
    RLVRConfig,
    SFTConfig,
    StructuredPostTrainingDataset,
    train_preference,
    train_grpo,
    train_sft,
)
from docvlm_eval.student.rewards import RewardConfig
from docvlm_eval.student.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_metadata(checkpoint: Path) -> dict:
    path = checkpoint / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_samples(args) -> list:
    if args.samples:
        return load_jsonl(args.samples)
    from docvlm_eval.synth.to_samples import load_realistic_samples

    return load_realistic_samples(
        args.realistic_root,
        variant=args.variant,
        include_probes=not args.no_probes,
    )


def _checkpoint_id(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    for name in ("student_config.json", "metadata.json"):
        path = checkpoint / name
        if path.exists():
            digest.update(path.read_bytes())
    model_path = checkpoint / "model.pt"
    stat = model_path.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return f"native:{digest.hexdigest()}"


def _device(args) -> str:
    if args.device != "auto":
        return args.device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sft", "preference", "rlvr"])
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path)
    source.add_argument("--realistic-root", type=Path)
    parser.add_argument(
        "--replay-samples",
        type=Path,
        help=(
            "Optional supervised replay JSONL for RLVR. "
            "Defaults to the active RLVR samples."
        ),
    )
    parser.add_argument("--variant", choices=["clean", "degraded"], default="clean")
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Native checkpoint student/ directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", help="Checkpoint path or 'latest'.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--replay-every-steps", type=int)
    parser.add_argument("--replay-loss-coefficient", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--target-mode",
        choices=["answer_only", "free_rationale", "evidence_linked"],
    )
    args = parser.parse_args()

    blueprint = load_blueprint(args.config)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    samples = _load_samples(args)
    tokenizer = DocumentTokenizer.from_pretrained(args.tokenizer)
    metadata = _checkpoint_metadata(args.checkpoint)
    fingerprint = metadata.get("tokenizer_fingerprint")
    if fingerprint is not None and fingerprint != tokenizer.fingerprint:
        raise SystemExit("checkpoint tokenizer fingerprint does not match --tokenizer")
    device = _device(args)
    collator_config = replace(
        StudentCollatorConfig.from_blueprint(blueprint),
        rotation_probability=0.0,
        contrastive=False,
    )
    collator = StudentCollator(tokenizer, collator_config)

    replay_requested = (
        args.replay_samples is not None
        or args.replay_every_steps is not None
        or args.replay_loss_coefficient is not None
    )
    if args.stage != "rlvr" and replay_requested:
        raise SystemExit("replay options apply only to RLVR")
    if args.stage == "sft":
        overrides = {
            "device": device,
            "resume_from": args.resume,
            "tokenizer_fingerprint": tokenizer.fingerprint,
        }
        if args.max_steps is not None:
            overrides["max_steps"] = args.max_steps
        if args.num_workers is not None:
            overrides["num_workers"] = args.num_workers
        if args.target_mode is not None:
            overrides["target_mode"] = args.target_mode
        config = SFTConfig.from_blueprint(
            blueprint,
            args.output,
            **overrides,
        )
        dataset = StructuredPostTrainingDataset(samples, config.target_mode)
        student = DocumentVLMStudent.from_pretrained(
            args.checkpoint,
            map_location=device,
        )
        result = train_sft(student, dataset, collator, config)
        print(
            f"Finished SFT step={result.global_step} checkpoint={result.last_checkpoint}"
        )
        return

    if args.target_mode is not None or args.num_workers is not None:
        raise SystemExit("--target-mode and --num-workers apply only to SFT")
    if not str(metadata.get("run_stage", "")).startswith("sft:"):
        raise SystemExit(f"{args.stage.upper()} must start from an SFT checkpoint")
    overrides = {
        "device": device,
        "resume_from": args.resume,
        "tokenizer_fingerprint": tokenizer.fingerprint,
    }
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    reference_id = _checkpoint_id(args.checkpoint)
    dataset = StructuredPostTrainingDataset(samples, target_mode="evidence_linked")
    policy = DocumentVLMStudent.from_pretrained(
        args.checkpoint,
        map_location=device,
    )
    reference = DocumentVLMStudent.from_pretrained(
        args.checkpoint,
        map_location=device,
    )
    reward_config = RewardConfig.from_blueprint(blueprint)
    if args.stage == "preference":
        config = PreferenceConfig.from_blueprint(
            blueprint,
            args.output,
            reference_id=reference_id,
            **overrides,
        )
        result = train_preference(
            policy,
            reference,
            dataset,
            collator,
            tokenizer,
            config,
            reward_config,
        )
        print(
            f"Finished {config.objective.upper()} "
            f"preference_step={result.preference_step} "
            f"optimizer_step={result.optimizer_step} "
            f"accepted_pairs={result.accepted_pairs} "
            f"student_flops={result.student_flops_seen:,} "
            f"executed_student_flops="
            f"{result.executed_student_flops_seen:,} "
            f"checkpoint={result.last_checkpoint}"
        )
        return

    if args.replay_every_steps is not None:
        overrides["supervised_replay_every_steps"] = args.replay_every_steps
    if args.replay_loss_coefficient is not None:
        overrides["supervised_replay_loss_coefficient"] = (
            args.replay_loss_coefficient
        )
    config = RLVRConfig.from_blueprint(
        blueprint,
        args.output,
        reference_id=reference_id,
        **overrides,
    )
    replay_dataset = (
        StructuredPostTrainingDataset(
            load_jsonl(args.replay_samples),
            target_mode="evidence_linked",
        )
        if args.replay_samples is not None
        else None
    )
    result = train_grpo(
        policy,
        reference,
        dataset,
        collator,
        tokenizer,
        config,
        reward_config,
        replay_dataset=replay_dataset,
    )
    print(
        f"Finished RLVR step={result.rollout_step} "
        f"student_flops={result.student_flops_seen:,} "
        f"executed_student_flops="
        f"{result.executed_student_flops_seen:,} "
        f"checkpoint={result.last_checkpoint}"
    )


if __name__ == "__main__":
    main()

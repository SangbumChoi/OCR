#!/usr/bin/env python3
"""Run native UDD pretraining, optionally with a same-tokenizer online teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.data import (
    BalancedGroupBatchSampler,
    StudentCollator,
    StudentCollatorConfig,
    UDDStudentDataset,
)
from docvlm_eval.student.distillation import (
    DistillationConfig,
    DistillationLoss,
    NativeStudentTeacher,
)
from docvlm_eval.student.model import DocumentVLMStudent
from docvlm_eval.student.pretrain import PretrainConfig, train_student
from docvlm_eval.student.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_tokenizer_fingerprint(checkpoint: Path) -> str | None:
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8")).get(
        "tokenizer_fingerprint"
    )


def _load_udd(src: Path, repo: str | None, split: str):
    from datasets import load_dataset, load_from_disk

    return load_dataset(repo, split=split) if repo else load_from_disk(str(src))


def _make_eval_loaders(
    dataset,
    tokenizer,
    collator_config,
    batch_size,
    num_workers,
    group_by,
    *,
    world_size=1,
    rank=0,
):
    from torch.utils.data import DataLoader, Subset

    if dataset is None or len(dataset) == 0:
        return {}
    expanded = UDDStudentDataset(dataset)
    groups = expanded.groups(group_by)
    eval_collator = StudentCollator(
        tokenizer,
        StudentCollatorConfig(
            **{
                **collator_config.__dict__,
                "rotation_probability": 0.0,
                "contrastive": False,
            }
        ),
    )
    out = {}
    for group in sorted(set(groups)):
        indices = [index for index, value in enumerate(groups) if value == group]
        indices = indices[rank::world_size]
        out[group] = DataLoader(
            Subset(expanded, indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_collator,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--src",
        type=Path,
        default=ROOT / "data" / "udd" / "hf" / "_all",
    )
    source.add_argument("--repo")
    parser.add_argument(
        "--eval-src",
        type=Path,
        help=(
            "Explicit validation UDD dataset. Every row is evaluated regardless "
            "of its internal fold label."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "student_pretrain",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-group-by",
        choices=["task", "source", "language", "component"],
        default="task",
    )
    parser.add_argument("--no-grounding", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    blueprint = load_blueprint(args.config)
    _, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    raw = _load_udd(args.src, args.repo, args.split)
    if args.eval_src is not None:
        train_rows = (
            raw.filter(
                lambda row: row["fold"] == "train",
                desc="UDD train fold",
            )
            if "fold" in raw.column_names
            else raw
        )
        heldout_rows = _load_udd(args.eval_src, None, args.split)
    elif "fold" in raw.column_names:
        train_rows = raw.filter(lambda row: row["fold"] == "train", desc="UDD train fold")
        heldout_rows = raw.filter(
            lambda row: row["fold"] == "heldout",
            desc="UDD heldout fold",
        )
    else:
        train_rows, heldout_rows = raw, None

    tokenizer = DocumentTokenizer.from_pretrained(args.tokenizer)
    student_config = StudentConfig.from_blueprint(blueprint)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = (
        torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if args.student_checkpoint:
        checkpoint_fingerprint = _checkpoint_tokenizer_fingerprint(
            args.student_checkpoint
        )
        if (
            checkpoint_fingerprint is not None
            and checkpoint_fingerprint != tokenizer.fingerprint
        ):
            raise SystemExit(
                "--student-checkpoint tokenizer fingerprint does not match --tokenizer"
            )
        student = DocumentVLMStudent.from_pretrained(
            args.student_checkpoint,
            map_location=device,
        )
    else:
        with torch.device(device):
            student = DocumentVLMStudent(student_config)
    if tokenizer.vocab_size > student.config.language.vocab_size:
        raise SystemExit("tokenizer vocabulary exceeds the student embedding table")

    collator_config = StudentCollatorConfig.from_blueprint(blueprint)
    collator = StudentCollator(tokenizer, collator_config)
    sequence_targets = blueprint["training"]["pretraining"]["distillation"].get(
        "sequence_targets",
        {},
    )
    train_dataset = UDDStudentDataset(
        train_rows,
        include_grounding=not args.no_grounding,
        teacher_target_probability=float(sequence_targets.get("probability", 0.0)),
        teacher_min_score=float(sequence_targets.get("min_score", 0.0)),
        teacher_target_seed=int(sequence_targets.get("seed", 0)),
    )
    teacher_targets = sum(
        source == "teacher" for source in train_dataset.target_sources
    )
    print(
        f"[sequence-distillation] selected={teacher_targets} "
        f"gold={len(train_dataset) - teacher_targets} "
        f"probability={float(sequence_targets.get('probability', 0.0)):.3f}",
        flush=True,
    )
    optimizer_raw = blueprint["training"]["pretraining"]["optimizer"]
    batch_size = args.batch_size or int(optimizer_raw["micro_batch_size"])
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(optimizer_raw["num_workers"])
    )
    overrides = {
        "resume_from": args.resume,
        "device": str(device),
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "target_source_counts": {
            "gold": len(train_dataset) - teacher_targets,
            "teacher": teacher_targets,
        },
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    config = PretrainConfig.from_blueprint(
        blueprint,
        args.output,
        **overrides,
    )
    balance_by = str(
        blueprint["training"]["pretraining"]["input_pipeline"]["balance_by"]
    )
    if config.adaptive_mixture.enabled and args.eval_group_by != balance_by:
        raise SystemExit(
            "adaptive mixture requires --eval-group-by to match "
            f"input_pipeline.balance_by ({balance_by!r})"
        )
    sampler = BalancedGroupBatchSampler.from_blueprint(
        train_dataset,
        blueprint,
        batch_size,
        grad_accum_steps=config.grad_accum_steps,
        epochs=config.epochs or 1,
        max_steps=config.max_steps,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    eval_loaders = _make_eval_loaders(
        heldout_rows,
        tokenizer,
        collator_config,
        batch_size,
        num_workers,
        args.eval_group_by,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
    )

    teacher = None
    distillation_loss = None
    if args.teacher_checkpoint:
        distillation_config = DistillationConfig.from_blueprint(blueprint)
        teacher_fingerprint = _checkpoint_tokenizer_fingerprint(
            args.teacher_checkpoint
        )
        if teacher_fingerprint != tokenizer.fingerprint:
            raise SystemExit(
                "online teacher checkpoint tokenizer fingerprint is missing or does "
                "not match --tokenizer; use sequence targets for cross-tokenizer teachers"
            )
        teacher_model = DocumentVLMStudent.from_pretrained(
            args.teacher_checkpoint,
            map_location=device,
        )
        teacher = NativeStudentTeacher(teacher_model, distillation_config)
        distillation_loss = DistillationLoss(
            student.config,
            teacher_model.config,
            distillation_config,
        )
    result = train_student(
        student,
        train_loader,
        config,
        teacher=teacher,
        distillation_loss=distillation_loss,
        eval_loaders=eval_loaders,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Finished step={result.global_step} "
            f"{result.token_unit}_tokens={result.budget_tokens_seen:,} "
            f"supervised_tokens={result.tokens_seen:,} "
            f"student_flops={result.student_flops_seen:,} "
            f"executed_student_flops="
            f"{result.executed_student_flops_seen:,} "
            f"checkpoint={result.last_checkpoint}"
        )


if __name__ == "__main__":
    main()

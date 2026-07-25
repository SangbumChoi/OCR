#!/usr/bin/env python3
"""Audit structured-target fit under evaluation and rollout token budgets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from docvlm_eval.architecture import load_blueprint
from docvlm_eval.benchmarks import load_jsonl
from docvlm_eval.generation_budget_audit import (
    GenerationBudgetPolicy,
    audit_generation_budget_coverage,
)
from docvlm_eval.student.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _named_paths(values: list[str], *, option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{option} must be NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path)
        if not name or name in result:
            raise SystemExit(f"{option} names must be non-empty and unique")
        if not path.is_file():
            raise SystemExit(f"{option} file does not exist: {path}")
        result[name] = path
    return result


def _overrides(values: list[str] | None) -> tuple[tuple[str, int], ...]:
    result = []
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--evaluation-token-budget must be PATTERN=TOKENS")
        pattern, raw_tokens = value.rsplit("=", 1)
        try:
            tokens = int(raw_tokens)
        except ValueError as error:
            raise SystemExit(
                "--evaluation-token-budget TOKENS must be an integer"
            ) from error
        result.append((pattern, tokens))
    return tuple(result)


def _blueprint_policy(
    blueprint: dict,
    stage: str,
) -> GenerationBudgetPolicy:
    rollout = blueprint["training"]["posttraining"][stage]["rollout"]
    return GenerationBudgetPolicy(
        name=stage,
        base_tokens=int(rollout["max_new_tokens"]),
        hard_cap=int(
            rollout.get(
                "max_new_tokens_hard_cap",
                rollout["max_new_tokens"],
            )
        ),
        by_answer_type=tuple(
            (str(pattern), int(tokens))
            for pattern, tokens in (
                rollout.get("max_new_tokens_by_answer_type") or {}
            ).items()
        ),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--calibration-split",
        action="append",
        dest="calibration_splits",
    )
    parser.add_argument("--evaluation-base-tokens", type=int, required=True)
    parser.add_argument("--evaluation-hard-cap", type=int, required=True)
    parser.add_argument(
        "--evaluation-token-budget",
        action="append",
        metavar="PATTERN=TOKENS",
    )
    parser.add_argument(
        "--target-mode",
        choices=["answer_only", "free_rationale", "evidence_linked"],
        default="evidence_linked",
    )
    parser.add_argument("--minimum-coverage", type=float, default=1.0)
    parser.add_argument("--recommendation-multiple", type=int, default=32)
    parser.add_argument("--max-overflow-examples", type=int, default=20)
    parser.add_argument(
        "--allow-policy-mismatch",
        action="store_true",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_paths = _named_paths(args.split, option="--split")
    samples_by_split = {
        name: load_jsonl(path)
        for name, path in split_paths.items()
    }
    blueprint = load_blueprint(args.config)
    tokenizer = DocumentTokenizer.from_pretrained(args.tokenizer)
    policies = [
        GenerationBudgetPolicy(
            name="evaluation",
            base_tokens=args.evaluation_base_tokens,
            hard_cap=args.evaluation_hard_cap,
            by_answer_type=_overrides(args.evaluation_token_budget),
        ),
        _blueprint_policy(blueprint, "preference"),
        _blueprint_policy(blueprint, "rlvr"),
    ]
    report = audit_generation_budget_coverage(
        samples_by_split,
        tokenizer,
        policies,
        target_mode=args.target_mode,
        calibration_splits=(
            tuple(args.calibration_splits)
            if args.calibration_splits
            else tuple(
                split
                for split in ("train", "validation")
                if split in samples_by_split
            )
        ),
        minimum_coverage=args.minimum_coverage,
        recommendation_multiple=args.recommendation_multiple,
        max_overflow_examples=args.max_overflow_examples,
        require_policy_consistency=not args.allow_policy_mismatch,
    )
    _write_json(args.output, report)
    print(
        f"generation budget audit: {report['gate']['status']} "
        f"({report['fingerprint']})"
    )
    if report["gate"]["status"] != "pass":
        raise SystemExit(
            "generation budget audit failed; inspect "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()

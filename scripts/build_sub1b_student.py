#!/usr/bin/env python3
"""Construct, count, selectively initialize, and optionally save the native sub-1B student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docvlm_eval.architecture import load_blueprint, validate_blueprint
from docvlm_eval.student.config import StudentConfig


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument("--tiny", action="store_true", help="Build a small contract-test model.")
    parser.add_argument(
        "--tiny-vocab-size",
        type=int,
        default=256,
        help="Vocabulary size for --tiny; use at least 260 with a trained byte tokenizer.",
    )
    parser.add_argument("--device", default="meta", choices=["meta", "cpu", "cuda"])
    parser.add_argument("--allow-full-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-arm", default="I0_random")
    parser.add_argument("--vision-source", type=Path)
    parser.add_argument("--vision-family", default="student", choices=["student", "siglip"])
    parser.add_argument("--language-source", type=Path)
    parser.add_argument(
        "--language-family",
        default="student",
        choices=["student", "llama", "lfm2"],
    )
    parser.add_argument(
        "--token-map",
        type=Path,
        help="JSON mapping of target token IDs to source token IDs for embedding transfer.",
    )
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    import torch

    from docvlm_eval.student.model import DocumentVLMStudent, count_unique_parameters
    from docvlm_eval.student.checkpoint import (
        checkpoint_content_identity,
        load_checkpoint_attention_geometry,
        load_checkpoint_state,
    )
    from docvlm_eval.student.transfer import selective_transfer

    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    torch.manual_seed(args.seed)
    blueprint = load_blueprint(args.config)
    estimates, errors = validate_blueprint(blueprint)
    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    config = (
        StudentConfig.tiny(vocab_size=args.tiny_vocab_size)
        if args.tiny
        else StudentConfig.from_blueprint(blueprint)
    )
    if not args.tiny and args.device != "meta" and not args.allow_full_memory:
        raise SystemExit("full construction allocates several GB; pass --allow-full-memory")
    with torch.device(args.device):
        model = DocumentVLMStudent(config)
    counts = count_unique_parameters(model)
    for name, count in counts.items():
        print(f"{name:>10}: {count:>12,} parameters")
    if not args.tiny:
        print(f" estimator: {estimates['total']:>12,} parameters")
        if counts["total"] >= int(blueprint["budget"]["max_parameters"]):
            raise SystemExit("constructed model exceeds the deployment parameter budget")

    arms = {arm["id"]: arm for arm in blueprint["initialization_arms"]}
    if args.init_arm not in arms:
        raise SystemExit(f"unknown initialization arm {args.init_arm!r}")
    arm = arms[args.init_arm]
    reports = []
    active_sources = {
        "vision": args.vision_source if arm["vision_transfer"] > 0 else None,
        "language": (
            args.language_source if arm["language_transfer"] > 0 else None
        ),
    }
    if args.device == "meta" and any(active_sources.values()):
        raise SystemExit("weight transfer requires a materialized cpu/cuda model")
    if args.vision_source and active_sources["vision"] is None:
        print(
            f"Skipping vision source: {args.init_arm} has zero vision transfer"
        )
    if args.language_source and active_sources["language"] is None:
        print(
            f"Skipping language source: {args.init_arm} has zero language transfer"
        )
    if active_sources["vision"]:
        vision_identity = checkpoint_content_identity(
            active_sources["vision"]
        )
        reports.append(
            selective_transfer(
                model,
                load_checkpoint_state(active_sources["vision"]),
                {"vision": arm["vision_transfer"]},
                family=args.vision_family,
                shape_policy=str(arm.get("shape_policy", "exact")),
                source_identity=vision_identity,
                require_attention_geometry=bool(
                    arm.get("require_attention_geometry", False)
                ),
                require_healthy_source_weights=bool(
                    arm.get("require_healthy_source_weights", False)
                ),
            ).to_dict()
        )
    if active_sources["language"]:
        language_identity = checkpoint_content_identity(
            active_sources["language"]
        )
        token_map = None
        if args.token_map:
            raw_token_map = json.loads(args.token_map.read_text(encoding="utf-8"))
            token_map = {int(target): int(source) for target, source in raw_token_map.items()}
        reports.append(
            selective_transfer(
                model,
                load_checkpoint_state(active_sources["language"]),
                {"language": arm["language_transfer"]},
                family=args.language_family,
                token_map=token_map,
                shape_policy=str(arm.get("shape_policy", "exact")),
                source_identity=language_identity,
                source_attention_geometry=(
                    load_checkpoint_attention_geometry(
                        active_sources["language"],
                        family=args.language_family,
                    )
                ),
                require_attention_geometry=bool(
                    arm.get("require_attention_geometry", False)
                ),
                require_healthy_source_weights=bool(
                    arm.get("require_healthy_source_weights", False)
                ),
            ).to_dict()
        )
    minimum_fractions = arm["minimum_component_parameter_fraction"]
    for report in reports:
        active_components = [
            component
            for component in ("vision", "language", "connector")
            if float(report["fractions"].get(component, 0.0)) > 0
        ]
        if len(active_components) != 1:
            raise SystemExit(
                "each transfer report must contain exactly one active component"
            )
        component = active_components[0]
        component_parameters = int(counts[component])
        realized_fraction = (
            int(report["copied_parameters"]) / component_parameters
            if component_parameters
            else 0.0
        )
        report["component"] = component
        report["target_component_parameters"] = component_parameters
        report["realized_component_parameter_fraction"] = realized_fraction
        report["minimum_component_parameter_fraction"] = float(
            minimum_fractions.get(component, 0.0)
        )
    if args.init_arm != "I0_random":
        required = []
        if arm["vision_transfer"] and not args.vision_source:
            required.append("--vision-source")
        if arm["language_transfer"] and not args.language_source:
            required.append("--language-source")
        if required:
            raise SystemExit(f"{args.init_arm} requires {' and '.join(required)}")
        empty_components = []
        for component in ("vision", "language"):
            if not arm[f"{component}_transfer"]:
                continue
            report = next(
                (
                    report
                    for report in reports
                    if report["fractions"].get(component, 0.0) > 0
                ),
                None,
            )
            if report is None or int(report["copied_parameters"]) <= 0:
                empty_components.append(component)
        if empty_components:
            raise SystemExit(
                "selective transfer copied zero parameters for required "
                f"components: {empty_components}"
            )
        underdosed_components = []
        for component in ("vision", "language"):
            if not arm[f"{component}_transfer"]:
                continue
            report = next(
                item for item in reports if item["component"] == component
            )
            if (
                report["realized_component_parameter_fraction"]
                < report["minimum_component_parameter_fraction"]
            ):
                underdosed_components.append(
                    {
                        "component": component,
                        "realized": report[
                            "realized_component_parameter_fraction"
                        ],
                        "minimum": report[
                            "minimum_component_parameter_fraction"
                        ],
                    }
                )
        if underdosed_components:
            raise SystemExit(
                "selective transfer parameter dose is below the required "
                f"component fraction: {underdosed_components}"
            )

    if args.save:
        if args.device == "meta":
            raise SystemExit("cannot save a meta-device model")
        metadata = {
            "blueprint": str(args.config),
            "initialization_arm": args.init_arm,
            "initialization_seed": args.seed,
            "transfer_reports": reports,
            "parameter_counts": counts,
        }
        model.save_pretrained(args.save, metadata=metadata)
        print(f"Saved {args.save}")
    elif reports:
        print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

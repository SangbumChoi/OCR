#!/usr/bin/env python3
"""Build the architecture-commonality and selective-transfer preflight report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from docvlm_eval.student.architecture_commonality import (
    architecture_commonality,
    load_architecture_catalog,
    profile_from_blueprint,
)


ROOT = Path(__file__).resolve().parents[1]


def _markdown(result: dict, profiles: list[dict]) -> str:
    profile_by_id = {profile["id"]: profile for profile in profiles}
    lines = [
        "# Small-VLM architecture commonality",
        "",
        "This report compares pinned public configs before any selective weight copy. "
        "A compatible tensor shape is necessary but not sufficient: module semantics, "
        "attention geometry, position convention, and tokenizer identity are checked separately.",
        "",
        "## Compared models",
        "",
        "| Model | Parameters | Vision | Connector | Language |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for profile in profiles:
        lines.append(
            f"| [{profile['id']}]({profile['config_url']}) | "
            f"{profile['parameters_millions']:.0f}M | "
            f"{profile['vision']['family']} / {profile['vision']['mixer']} | "
            f"{profile['connector']['family']} | "
            f"{profile['language']['family']} / {profile['language']['mixer']} |"
        )
    lines.extend(
        [
            "",
            "## Common characteristics",
            "",
            "| Feature | Modal value | Prevalence | Target match |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for item in result["common_features"]:
        value = json.dumps(item["value"], ensure_ascii=True)
        lines.append(
            f"| `{item['feature']}` | `{value}` | "
            f"{item['count']}/{result['source_count']} | "
            f"{'yes' if item['target_matches'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Transfer preflight",
            "",
            "| Source | Compatible subcomponents | Exact | Structured | Token rows | Distill only |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in result["compatibility"]:
        counts = Counter(
            decision["mode"]
            for decision in item["decisions"].values()
        )
        source = profile_by_id[item["source"]]
        lines.append(
            f"| [{item['source']}]({source['config_url']}) | "
            f"{item['compatible_subcomponents']}/{item['total_subcomponents']} | "
            f"{counts['exact']} | {counts['structured_mlp']} | "
            f"{counts['token_rows']} | {counts['distill_only']} |"
        )
    lines.extend(
        [
            "",
            "## Decision rules",
            "",
            "- Copy attention only when hidden width, query heads, KV heads, head dimension, "
            "normalization, RoPE convention, and RoPE base all match.",
            "- Reduce an MLP only as one complete SwiGLU group with one shared channel selection; "
            "never crop independent matrices.",
            "- Copy token embeddings only through an explicit tokenizer identity map.",
            "- Treat position embeddings, hybrid-convolution blocks, encoder-decoder text stacks, "
            "and non-identical connectors as distillation targets rather than weight-copy targets.",
            "- Run an initialization factorial against random, vision-only, language-only, dual, "
            "and selective controls; this report establishes compatibility, not downstream benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "small_vlm_architectures.yaml",
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=ROOT / "configs" / "sub1b_architecture.yaml",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "results"
            / "small_vlm_architecture_commonality.json"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "report"
            / "small_vlm_architecture_commonality.md"
        ),
    )
    args = parser.parse_args()
    profiles = load_architecture_catalog(args.catalog)
    blueprint = yaml.safe_load(args.blueprint.read_text(encoding="utf-8"))
    target = profile_from_blueprint(blueprint)
    result = architecture_commonality(profiles, target)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        _markdown(result, profiles),
        encoding="utf-8",
    )
    print(args.report_output)


if __name__ == "__main__":
    main()

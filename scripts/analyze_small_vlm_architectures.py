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
    lfm_meta_transfer_preflight,
    load_architecture_catalog,
    profile_from_blueprint,
)
from docvlm_eval.student.compute import (
    estimate_forward_flops,
    estimate_language_kv_cache_bytes,
    estimate_training_flops_breakdown,
)
from docvlm_eval.student.config import StudentConfig
from docvlm_eval.student.sweep import apply_json_patch


ROOT = Path(__file__).resolve().parents[1]


def _compatibility_table(
    result: dict,
    profile_by_id: dict[str, dict],
) -> list[str]:
    lines = [
        "| Source | Compatible subcomponents | Exact | Structured | Token rows | Distill only |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
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
    return lines


def _markdown(
    result: dict,
    profiles: list[dict],
    aligned: dict,
    preflight: dict,
    compute: dict,
) -> str:
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
            *_compatibility_table(result, profile_by_id),
        ]
    )
    lines.extend(
        [
            "",
            "## LFM-aligned sub-1B control",
            "",
            "The executable control keeps the document vision tower fixed and aligns the language "
            "operator to LFM2.5. It uses width 2048, 32 query heads, 8 KV heads, per-head Q/K "
            "RMSNorm, half-split RoPE at base 1,000,000, bias-free projections, kernel-3 short "
            "convolution, 12 layers, and a reduced 5,120-channel SwiGLU.",
            "",
            f"- Deployment parameters: `{preflight['target_parameters']:,}`.",
            f"- Copied language parameters: `{preflight['copied_parameters']:,}` "
            f"({preflight['copied_language_fraction']:.2%}).",
            f"- Exact tensors: `{preflight['exact_tensors']}`; structured tensors: "
            f"`{preflight['structured_tensors']}` across "
            f"`{preflight['structured_groups']}` SwiGLU groups.",
            f"- Shape skips: `{preflight['shape_skips']}`; semantic skips: "
            f"`{preflight['semantic_skips']}`; missing source keys: "
            f"`{preflight['missing_source_keys']}`.",
            f"- Representative forward FLOPs: `{compute['lfm_aligned']['forward_flops']:,}` "
            f"versus `{compute['native']['forward_flops']:,}` native "
            f"({compute['forward_ratio']:.2%}).",
            f"- 2,176-token language state: `{compute['lfm_aligned']['kv_cache_bytes']:,}` "
            f"bytes versus `{compute['native']['kv_cache_bytes']:,}` native "
            f"({compute['kv_cache_ratio']:.2%}).",
            "",
            *_compatibility_table(aligned, profile_by_id),
            "",
            "## Decision rules",
            "",
            "- Copy attention only when hidden width, query heads, KV heads, head dimension, "
            "normalization, RoPE convention, and RoPE base all match.",
            "- Reduce an MLP only as one complete SwiGLU group with one shared channel selection; "
            "never crop independent matrices.",
            "- Copy token embeddings only through an explicit tokenizer identity map.",
            "- Transfer short-convolution blocks only through like-typed depth mapping with the "
            "same width, kernel, bias, and normalization contract.",
            "- Treat incompatible position embeddings, encoder-decoder text stacks, and "
            "non-identical connectors as distillation targets rather than weight-copy targets.",
            "- Run an initialization factorial against random, vision-only, language-only, dual, "
            "and selective controls; this report establishes compatibility, not downstream benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def _aligned_blueprint(
    blueprint: dict,
    sweep_path: Path,
) -> dict:
    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))
    variant = next(
        item for item in sweep["variants"] if item["id"] == "lfm_random"
    )
    return apply_json_patch(blueprint, variant["blueprint_patches"])


def _compute_profile(blueprint: dict) -> dict[str, int]:
    config = StudentConfig.from_blueprint(blueprint)
    forward = estimate_forward_flops(
        config,
        text_tokens=2048,
        vision_tokens=2520,
    )
    training = estimate_training_flops_breakdown(
        config,
        text_tokens=2048,
        vision_tokens=2520,
    )
    return {
        "forward_flops": forward.total,
        "training_flops": training.algorithmic,
        "kv_cache_bytes": estimate_language_kv_cache_bytes(
            config,
            sequence_tokens=2176,
        ),
    }


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
        "--lfm-sweep",
        type=Path,
        default=(
            ROOT
            / "configs"
            / "sub1b_lfm_language_transfer_sweep.yaml"
        ),
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
    aligned_blueprint = _aligned_blueprint(blueprint, args.lfm_sweep)
    aligned_target = profile_from_blueprint(aligned_blueprint)
    aligned_target["id"] = "docvlm-lfm-aligned-814m"
    aligned_target["model_id"] = "local/docvlm-lfm-aligned-814m"
    aligned_target["config_url"] = (
        "configs/sub1b_lfm_language_transfer_sweep.yaml"
    )
    aligned = architecture_commonality(profiles, aligned_target)
    lfm_profile = next(
        profile for profile in profiles
        if profile["id"] == "lfm2.5-vl-1.6b"
    )
    preflight = lfm_meta_transfer_preflight(
        aligned_blueprint,
        lfm_profile,
    )
    native_compute = _compute_profile(blueprint)
    aligned_compute = _compute_profile(aligned_blueprint)
    compute = {
        "contract": {
            "text_tokens": 2048,
            "vision_tokens": 2520,
            "cache_sequence_tokens": 2176,
            "batch_size": 1,
            "multiply_add": "two_flops",
        },
        "native": native_compute,
        "lfm_aligned": aligned_compute,
        "forward_ratio": (
            aligned_compute["forward_flops"]
            / native_compute["forward_flops"]
        ),
        "training_ratio": (
            aligned_compute["training_flops"]
            / native_compute["training_flops"]
        ),
        "kv_cache_ratio": (
            aligned_compute["kv_cache_bytes"]
            / native_compute["kv_cache_bytes"]
        ),
    }
    output = {
        "schema_version": 2,
        "default_target": result,
        "lfm_aligned_target": aligned,
        "lfm_meta_preflight": preflight,
        "compute_comparison": compute,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_output.write_text(
        _markdown(result, profiles, aligned, preflight, compute),
        encoding="utf-8",
    )
    print(args.report_output)


if __name__ == "__main__":
    main()

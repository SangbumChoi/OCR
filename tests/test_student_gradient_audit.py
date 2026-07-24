import json
import sys
from pathlib import Path

import yaml

from docvlm_eval.student.gradient_audit import (
    aggregate_gradient_conflict_audit,
    write_gradient_conflict_audit,
)
from docvlm_eval.student.sweep import compile_sweep_plan


ROOT = Path(__file__).resolve().parents[1]


def _audit_plan(tmp_path: Path):
    raw = yaml.safe_load(
        (
            ROOT / "configs" / "sub1b_gradient_conflict_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    raw["output_root"] = str(tmp_path / "audit")
    config = tmp_path / "gradient-audit.yaml"
    config.write_text(
        yaml.safe_dump(raw, sort_keys=False),
        encoding="utf-8",
    )
    return compile_sweep_plan(
        config,
        repo_root=ROOT,
        python=sys.executable,
        compile_root=tmp_path / "compiled",
    )


def _write_run(
    run_root: Path,
    *,
    cosine: float | None,
    model_payload: bytes,
) -> None:
    pretrain = run_root / "artifacts" / "pretrain"
    checkpoint = pretrain / "checkpoints" / "step_00000010"
    student = checkpoint / "student"
    student.mkdir(parents=True)
    (student / "model.pt").write_bytes(model_payload)
    (pretrain / "latest_checkpoint.txt").write_text(
        str(checkpoint.resolve()) + "\n",
        encoding="utf-8",
    )
    if cosine is None:
        return
    records = [
        {
            "kind": "gradient_conflict",
            "gradient_probe/cosine/autoregressive__orientation": cosine,
            (
                "gradient_probe/overlap_elements/"
                "autoregressive__orientation"
            ): 32,
        }
        for _ in range(8)
    ]
    (pretrain / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_gradient_audit_compiles_replicates_and_promotes_persistent_conflict(
    tmp_path,
):
    plan = _audit_plan(tmp_path)

    assert len(plan.variants) == 9
    assert plan.baseline == "no_probe"
    for variant in plan.variants:
        probe = variant.plan.resolved_blueprint["training"]["pretraining"][
            "gradient_conflict_probe"
        ]
        assert probe["enabled"] is (variant.arm_id != "no_probe")
        assert probe["every_steps"] == 1000
        assert probe["components"] == (
            ["vision"]
            if variant.arm_id == "vision_anchor"
            else ["vision", "connector", "language"]
        )
        _write_run(
            Path(variant.plan.root),
            cosine=(
                None
                if variant.arm_id == "no_probe"
                else 0.10
                if variant.arm_id == "vision_anchor"
                else -0.20
            ),
            model_payload=f"model-{variant.replicate_id}".encode(),
        )

    audit = aggregate_gradient_conflict_audit(plan)

    assert audit["trajectory_invariance"]["status"] == "pass"
    assert audit["decision"] == "promote_gradient_surgery"
    pair = "autoregressive__orientation"
    assert pair in audit["evidence"]["persistent_conflict_pairs"]
    assert pair in audit["evidence"]["material_anchor_difference_pairs"]
    assert audit["arms"]["all_trunks"]["pairs"][pair]["measurements"] == 24
    written = write_gradient_conflict_audit(
        plan,
        output_dir=tmp_path / "reports",
    )
    assert written["decision"] == "promote_gradient_surgery"
    assert (tmp_path / "reports" / "gradient_conflict_audit.json").is_file()
    assert (tmp_path / "reports" / "gradient_conflict_audit.md").is_file()

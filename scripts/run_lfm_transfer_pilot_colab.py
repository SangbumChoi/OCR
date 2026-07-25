#!/usr/bin/env python3
"""Run a gated selective-transfer experiment with compact Colab output."""

from __future__ import annotations

import argparse
import json
import netrc
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from docvlm_eval.student.confirmatory_submission import (
    audit_smol_confirmatory_submission,
)
from docvlm_eval.student.sweep import compile_sweep_plan
from docvlm_eval.student.transfer_readiness import (
    audit_lfm_transfer_pilot,
    audit_smol_vision_transfer_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "configs" / "sub1b_lfm_language_transfer_pilot.yaml"
PREFLIGHT = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_lfm_real_source_preflight.json"
)
SOURCE_SELECTION = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_source_matrix.json"
)
SWEEP_ROOT = ROOT / "outputs" / "sweeps" / "docvlm-lfm-language-transfer-pilot"
SMOL_SWEEP = ROOT / "configs" / "sub1b_smol_vision_transfer_pilot.yaml"
SMOL_PREFLIGHT = (
    ROOT
    / "docs"
    / "results"
    / "selective_transfer_smol_vision_real_source_preflight.json"
)
SMOL_SWEEP_ROOT = (
    ROOT / "outputs" / "sweeps" / "docvlm-smol-vision-transfer-pilot"
)
SMOL_CONFIRMATORY_SWEEP = (
    ROOT / "configs" / "sub1b_smol_vision_transfer_sweep.yaml"
)
SMOL_CONFIRMATORY_SWEEP_ROOT = (
    ROOT / "outputs" / "sweeps" / "docvlm-smol-vision-transfer-sweep"
)
SMOL_EXECUTION = (
    ROOT
    / "docs"
    / "results"
    / "smol_vision_transfer_pilot_execution_state.json"
)
SMOL_PILOT_COMPARISON = SMOL_SWEEP_ROOT / "comparison.json"
PILOTS = {
    "lfm-language": {
        "sweep": SWEEP,
        "root": SWEEP_ROOT,
        "log_name": "colab_pilot.log",
        "readiness_prefix": "docvlm-lfm-colab-readiness-",
    },
    "smol-vision": {
        "sweep": SMOL_SWEEP,
        "root": SMOL_SWEEP_ROOT,
        "log_name": "colab_pilot.log",
        "readiness_prefix": "docvlm-smol-colab-readiness-",
    },
    "smol-confirmatory": {
        "sweep": SMOL_CONFIRMATORY_SWEEP,
        "root": SMOL_CONFIRMATORY_SWEEP_ROOT,
        "log_name": "colab_confirmatory.log",
        "readiness_prefix": "docvlm-smol-confirmatory-readiness-",
    },
}


def _wandb_credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY", "").strip():
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def _gpu_evidence() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch is not installed"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "available": True,
        "device": device,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": list(
            torch.cuda.get_device_capability(device)
        ),
        "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    variants = summary.get("variants") or []
    promotion = summary.get("promotion") or {}
    multiple_comparisons = promotion.get("multiple_comparisons") or {}
    return {
        "status": summary.get("status"),
        "variants": [
            {
                "run": item.get("run"),
                "status": item.get("status"),
                **(
                    {"error": str(item.get("error"))[:300]}
                    if item.get("error")
                    else {}
                ),
            }
            for item in variants
        ],
        **(
            {"comparison": summary["comparison"]}
            if summary.get("comparison")
            else {}
        ),
        **(
            {
                "promotion": {
                    "status": promotion.get("status"),
                    "selected_variants": promotion.get(
                        "selected_variants"
                    ),
                    "multiple_comparisons": {
                        "method": multiple_comparisons.get("method"),
                        "comparison_count": multiple_comparisons.get(
                            "comparison_count"
                        ),
                        "familywise_alpha": multiple_comparisons.get(
                            "familywise_alpha"
                        ),
                    },
                }
            }
            if promotion
            else {}
        ),
    }


def _tail(path: Path, *, lines: int = 20, width: int = 500) -> list[str]:
    try:
        values = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return []
    return [line[:width] for line in values[-lines:]]


def _readiness(pilot: str) -> dict[str, Any]:
    profile = PILOTS[pilot]
    with tempfile.TemporaryDirectory(
        prefix=str(profile["readiness_prefix"])
    ) as temporary:
        plan = compile_sweep_plan(
            profile["sweep"],
            repo_root=ROOT,
            python=sys.executable,
            compile_root=temporary,
        )
        if pilot == "smol-confirmatory":
            pilot_plan = compile_sweep_plan(
                SMOL_SWEEP,
                repo_root=ROOT,
                python=sys.executable,
                compile_root=Path(temporary) / "pilot",
            )
            return audit_smol_confirmatory_submission(
                pilot_plan,
                plan,
                pilot_readiness=json.loads(
                    (
                        ROOT
                        / "docs"
                        / "results"
                        / "smol_vision_transfer_pilot_readiness.json"
                    ).read_text(encoding="utf-8")
                ),
                pilot_execution=json.loads(
                    SMOL_EXECUTION.read_text(encoding="utf-8")
                ),
                pilot_comparison=(
                    _read_summary(SMOL_PILOT_COMPARISON) or None
                ),
            )
        if pilot == "smol-vision":
            return audit_smol_vision_transfer_pilot(
                plan,
                repo_root=ROOT,
                sweep_path=profile["sweep"],
                vision_preflight_path=SMOL_PREFLIGHT,
                language_preflight_path=PREFLIGHT,
                source_selection_path=SOURCE_SELECTION,
            )
        return audit_lfm_transfer_pilot(
            plan,
            repo_root=ROOT,
            sweep_path=profile["sweep"],
            preflight_path=PREFLIGHT,
            source_selection_path=SOURCE_SELECTION,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        choices=sorted(PILOTS),
        default="lfm-language",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--replicate", action="append", dest="replicates")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=300.0)
    parser.add_argument("--min-free-gib", type=float, default=25.0)
    parser.add_argument("--min-gpu-gib", type=float, default=14.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    profile = PILOTS[args.pilot]
    sweep = Path(profile["sweep"])
    sweep_root = Path(profile["root"])
    if args.log is None:
        args.log = sweep_root / str(profile["log_name"])
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.heartbeat_seconds < args.poll_seconds:
        parser.error("--heartbeat-seconds must be at least --poll-seconds")
    if args.min_free_gib <= 0 or args.min_gpu_gib <= 0:
        parser.error("memory thresholds must be positive")

    readiness = _readiness(args.pilot)
    print(
        json.dumps(
            {
                "readiness": readiness["overall_status"],
                "checks": readiness["counts"],
                "fingerprint": readiness["fingerprint"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if readiness["overall_status"] != "pass" and not args.dry_run:
        raise SystemExit(
            f"{args.pilot} submission readiness audit did not pass"
        )

    free_bytes = shutil.disk_usage(ROOT).free
    environment = {
        "free_disk_gib": round(free_bytes / 2**30, 2),
        "wandb_credentials": _wandb_credentials_available(),
        "gpu": _gpu_evidence(),
    }
    print(json.dumps({"environment": environment}, sort_keys=True), flush=True)
    if not args.dry_run:
        if free_bytes < args.min_free_gib * 2**30:
            raise SystemExit(
                f"need at least {args.min_free_gib:g} GiB free disk"
            )
        if not environment["wandb_credentials"]:
            raise SystemExit(
                "W&B credentials are missing; run wandb.login() or set "
                "WANDB_API_KEY"
            )
        gpu = environment["gpu"]
        if not gpu.get("available"):
            raise SystemExit(str(gpu.get("reason") or "CUDA is unavailable"))
        if not gpu.get("bfloat16_supported"):
            raise SystemExit(
                "the production pilot requires native CUDA bfloat16 support; "
                "T4 is not compatible, use an L4, A10, A100, or newer GPU"
            )
        if int(gpu["total_memory_bytes"]) < args.min_gpu_gib * 2**30:
            raise SystemExit(
                f"need at least {args.min_gpu_gib:g} GiB GPU memory"
            )

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_student_sweep.py"),
        "--sweep",
        str(sweep),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.no_resume:
        command.append("--no-resume")
    for variant in args.variants or []:
        command.extend(["--variant", variant])
    for replicate in args.replicates or []:
        command.extend(["--replicate", replicate])

    args.log = args.log.resolve()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_root / "sweep_run_summary.json"
    started = time.monotonic()
    last_heartbeat = started
    last_snapshot = None
    with args.log.open("a", encoding="utf-8") as log:
        log.write(
            "\n"
            + json.dumps(
                {
                    "launcher_started_at_unix": time.time(),
                    "pilot": args.pilot,
                    "command": command,
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            summary = (
                {}
                if args.dry_run
                else _compact_summary(_read_summary(summary_path))
            )
            snapshot = json.dumps(summary, sort_keys=True)
            if summary and snapshot != last_snapshot:
                print(snapshot, flush=True)
                last_snapshot = snapshot
                last_heartbeat = time.monotonic()
            elif time.monotonic() - last_heartbeat >= args.heartbeat_seconds:
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "elapsed_minutes": round(
                                (time.monotonic() - started) / 60,
                                1,
                            ),
                            "log_mib": round(
                                args.log.stat().st_size / 2**20,
                                2,
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            time.sleep(args.poll_seconds)
        return_code = process.wait()

    if args.dry_run:
        result = {
            "status": "completed" if return_code == 0 else "failed",
            "dry_run": True,
            "pilot": args.pilot,
            "log": str(args.log),
        }
    else:
        result = _compact_summary(_read_summary(summary_path))
        result["log"] = str(args.log)
    print(json.dumps(result, sort_keys=True), flush=True)
    if return_code:
        print("\n".join(_tail(args.log)), file=sys.stderr)
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()

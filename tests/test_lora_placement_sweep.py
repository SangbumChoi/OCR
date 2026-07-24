import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_lora_placement_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_lora_placement_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
compile_commands = MODULE.compile_commands


def _raw():
    return yaml.safe_load(
        (ROOT / "configs" / "lora_vision_connector_sweep.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_confirmatory_sweep_compiles_matched_paired_commands():
    commands = compile_commands(
        _raw(),
        python=sys.executable,
        repo_root=ROOT,
    )

    assert len(commands) == 6
    by_replicate = {}
    for item in commands:
        by_replicate.setdefault(item["replicate"], []).append(item)
        command = item["command"]
        assert "--lora-budget-reference" in command
        assert command[command.index("--lora-budget-reference") + 1] == "vision"
        assert item["record_key"].endswith(
            f"{item['variant']}:{item['replicate']}"
        )
    assert set(by_replicate) == {"seed_0", "seed_1", "seed_2"}
    for pair in by_replicate.values():
        assert {item["variant"] for item in pair} == {
            "vision",
            "vision_connector",
        }
        assert len({item["optimizer_seed"] for item in pair}) == 1
        assert len({item["data_seed"] for item in pair}) == 1


def test_confirmatory_sweep_rejects_unmatched_design():
    raw = _raw()
    raw["placements"] = ["vision", "connector"]

    with pytest.raises(ValueError, match="vision_connector"):
        compile_commands(raw, python=sys.executable, repo_root=ROOT)

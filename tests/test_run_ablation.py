import importlib.util
from pathlib import Path
from types import SimpleNamespace

from docvlm_eval.finetune.lora_vlm import LoraVLMConfig

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_a0_honors_step_cap_and_training_seed(monkeypatch):
    generated = []
    configs = []
    records = []

    def fake_generate(out_dir, seed, count):
        generated.append((Path(out_dir).name, seed, count))
        return "heldout.jsonl" if "_a0_heldout" in str(out_dir) else "train.jsonl"

    def fake_train(config, eval_specs):
        configs.append((config, eval_specs))
        return "", {
            "train": {"score": 0.5},
            "heldout": {"score": 0.4},
        }

    monkeypatch.setattr(MODULE, "_gen_realistic", fake_generate)
    monkeypatch.setattr(MODULE, "_n_samples", lambda _: 10)
    monkeypatch.setattr(
        MODULE,
        "_record",
        lambda model, arm, payload: records.append((model, arm, payload)),
    )
    monkeypatch.setitem(MODULE.HF_ID, "fixture", "fixture/model")
    args = SimpleNamespace(
        train_jsonl=None,
        a0_test_seed=999,
        a0_test_count=2,
        a0_sizes=[3],
        models=["fixture"],
        a0_epochs=3,
        steps=300,
        data_seed=107,
        placement="all",
        lora_r=16,
        lora_alpha=32,
        quantization_bits=4,
        lora_budget_reference=None,
        lr=0.0001,
        seed=7,
        wandb_project=None,
        wandb_run_prefix="",
        no_grad_ckpt=False,
        _mils=768,
        batch_size=2,
        eval_max_samples=64,
    )

    MODULE.run_a0(args, None, fake_train, LoraVLMConfig)

    assert generated == [
        ("_a0_heldout", 999, 2),
        ("realistic_cases", 107, 3),
    ]
    config, eval_specs = configs[0]
    assert config.max_steps == 300
    assert config.epochs == 3
    assert config.seed == 7
    assert eval_specs == [
        ("train", "train.jsonl"),
        ("heldout", "heldout.jsonl"),
    ]
    payload = records[-1][2]
    assert payload["max_micro_steps"] == 300
    assert payload["data_seed"] == 107
    assert payload["sizes"]["3"]["max_micro_steps"] == 300

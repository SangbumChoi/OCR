import importlib.util
import os
import socket
from dataclasses import replace
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="distributed student tests require torch",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_resume_worker(rank: int, world_size: int, port: int, root: str) -> None:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent
    from docvlm_eval.student.pretrain import PretrainConfig, train_student

    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    batch = {
        "input_ids": torch.tensor([[1, 7, 8, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 8, 2]], dtype=torch.long),
    }

    def loader():
        return DataLoader([batch, batch], batch_size=None)

    def config(output: Path, max_steps: int, resume: str | None = None):
        return PretrainConfig(
            output_dir=str(output),
            epochs=1,
            max_steps=max_steps,
            learning_rate=1e-3,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            warmup_tokens=0,
            total_tokens=100,
            grad_accum_steps=1,
            checkpoint_every_steps=1,
            eval_every_steps=0,
            log_every_steps=1,
            precision="float32",
            device="cpu",
            resume_from=resume,
            tokenizer_fingerprint="sha256:distributed-test",
            loss_weights={"autoregressive": 1.0},
        )

    model_config = StudentConfig.tiny()
    model_config = replace(
        model_config,
        language=replace(model_config.language, dropout=0.1),
    )
    root_path = Path(root)
    torch.manual_seed(71)
    resumed = DocumentVLMStudent(model_config)
    train_student(resumed, loader(), config(root_path / "resumed", 1))
    result = train_student(
        resumed,
        loader(),
        config(root_path / "resumed", 2, "latest"),
    )

    torch.manual_seed(71)
    uninterrupted = DocumentVLMStudent(model_config)
    train_student(uninterrupted, loader(), config(root_path / "full", 2))

    assert result.global_step == 2
    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, resumed.state_dict()[name]), name
    if rank == 0:
        payload = torch.load(
            Path(result.last_checkpoint) / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert len(payload["rng_states"]) == world_size
        assert all(state is not None for state in payload["rng_states"])
    dist.destroy_process_group()


def test_two_rank_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_resume_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    assert (
        tmp_path
        / "resumed"
        / "checkpoints"
        / "step-00000002"
        / "training_state.pt"
    ).exists()

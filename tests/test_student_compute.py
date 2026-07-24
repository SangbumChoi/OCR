from dataclasses import replace

import pytest

from docvlm_eval.student.compute import (
    compute_profile,
    estimate_batch_training_flops,
    estimate_batch_training_flops_breakdown,
    estimate_forward_flops,
    estimate_rlvr_step_flops,
    estimate_training_flops,
    estimate_training_flops_breakdown,
    visual_tokens_for_canvas,
)
from docvlm_eval.student.config import StudentConfig


def test_visual_canvas_token_count_is_square_and_capped():
    assert visual_tokens_for_canvas(448, 14, 4096) == 1024
    assert visual_tokens_for_canvas(672, 14, 4096) == 2304
    assert visual_tokens_for_canvas(896, 14, 4096) == 4096
    assert visual_tokens_for_canvas(1200, 14, 4096) == 4096

    with pytest.raises(ValueError, match="square"):
        visual_tokens_for_canvas(448, 14, 4000)


def test_forward_flops_increase_with_resolution_and_visual_latents():
    config = StudentConfig.tiny()
    low_resolution = estimate_forward_flops(
        config,
        text_tokens=32,
        vision_tokens=16,
    )
    high_resolution = estimate_forward_flops(
        config,
        text_tokens=32,
        vision_tokens=64,
    )
    more_latents = estimate_forward_flops(
        replace(
            config,
            connector=replace(config.connector, latent_tokens=16),
        ),
        text_tokens=32,
        vision_tokens=16,
    )

    assert high_resolution.vision > low_resolution.vision
    assert high_resolution.connector > low_resolution.connector
    assert high_resolution.total > low_resolution.total
    assert more_latents.connector > low_resolution.connector
    assert more_latents.language > low_resolution.language
    assert more_latents.total > low_resolution.total


def test_batch_training_flops_use_dense_padded_shapes():
    torch = pytest.importorskip("torch")
    config = StudentConfig.tiny()
    batch = {
        "input_ids": torch.ones(2, 24, dtype=torch.long),
        "pixel_values": torch.ones(2, 3, 32, 32),
    }

    assert estimate_batch_training_flops(config, batch) == (
        estimate_training_flops(
            config,
            text_tokens=24,
            vision_tokens=16,
            batch_size=2,
        )
    )


def test_checkpoint_recompute_is_reported_without_changing_compute_budget():
    torch = pytest.importorskip("torch")
    config = StudentConfig.tiny()
    batch = {
        "input_ids": torch.ones(2, 24, dtype=torch.long),
        "pixel_values": torch.ones(2, 3, 32, 32),
    }
    baseline = estimate_batch_training_flops_breakdown(config, batch)
    checkpointed = estimate_batch_training_flops_breakdown(
        config,
        batch,
        checkpoint_components=("vision", "connector", "language"),
    )
    language_only = estimate_training_flops_breakdown(
        config,
        text_tokens=24,
        vision_tokens=0,
        batch_size=2,
        checkpoint_components=("language",),
    )

    assert checkpointed.algorithmic == baseline.algorithmic
    assert checkpointed.checkpoint_recompute > 0
    assert checkpointed.executed == (
        checkpointed.algorithmic + checkpointed.checkpoint_recompute
    )
    assert language_only.checkpoint_recompute > 0
    with pytest.raises(ValueError, match="unique subset"):
        estimate_training_flops_breakdown(
            config,
            text_tokens=24,
            vision_tokens=16,
            checkpoint_components=("language", "language"),
        )


def test_rlvr_compute_tracks_generation_and_replay():
    config = StudentConfig.tiny()
    short = estimate_rlvr_step_flops(
        config,
        vision_tokens=16,
        prompt_tokens=24,
        completion_tokens=2,
        group_size=2,
    )
    long = estimate_rlvr_step_flops(
        config,
        vision_tokens=16,
        prompt_tokens=24,
        completion_tokens=4,
        group_size=2,
    )
    replay = estimate_rlvr_step_flops(
        config,
        vision_tokens=16,
        prompt_tokens=24,
        completion_tokens=2,
        group_size=2,
        replay_text_tokens=32,
        checkpoint_components=("vision", "connector", "language"),
    )

    assert long["rollout"] > short["rollout"]
    assert long["policy_update"] > short["policy_update"]
    assert long["total"] > short["total"]
    assert replay["supervised_replay"] > 0
    assert replay["total"] > short["total"]
    assert replay["checkpoint_recompute"] > 0
    assert replay["executed_total"] == (
        replay["total"] + replay["checkpoint_recompute"]
    )


def test_profile_exposes_phase_accounting_convention():
    profile = compute_profile(
        StudentConfig.tiny(),
        image_long_side=32,
        text_tokens=64,
        rlvr_completion_tokens=4,
        rlvr_group_size=2,
        rlvr_replay_every_steps=2,
        checkpoint_components=("vision", "connector", "language"),
    )

    assert profile["vision_tokens"] == 16
    assert profile["training_flops_per_sample"] == (
        3 * profile["forward_flops"]["total"]
    )
    assert profile["rlvr"]["expected_replay_per_step"] > 0
    assert profile["checkpoint_recompute_flops_per_sample"] > 0
    assert profile["executed_training_flops_per_sample"] > (
        profile["training_flops_per_sample"]
    )
    assert profile["rlvr"]["expected_executed_total_per_step"] > (
        profile["rlvr"]["expected_total_per_step"]
    )
    assert profile["convention"]["multiply_add"] == "two_flops"

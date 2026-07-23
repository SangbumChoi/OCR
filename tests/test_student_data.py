import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="student data tests require torch",
)


class _CharacterTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [3 + ord(character) % 240 for character in text]


def _udd_rows():
    import json

    from PIL import Image

    return [
        {
            "image": Image.new("RGB", (20, 10), "white"),
            "sample_id": "doc-1",
            "source": "docvqa",
            "task": "vqa",
            "instructions": ["What is the total?", "What currency is used?"],
            "answers": [["42"], ["USD", "US dollars"]],
            "elements_json": json.dumps(
                [
                    {
                        "key": "total",
                        "value": "42",
                        "bbox": [0.1, 0.2, 0.5, 0.6, True],
                        "kind": "field",
                    }
                ]
            ),
            "language": "en",
            "image_width": 20,
            "image_height": 10,
        },
        {
            "image": Image.new("RGB", (8, 16), "black"),
            "sample_id": "doc-2",
            "source": "synthdog_ko",
            "task": "recognition",
            "instructions": ["Transcribe the page."],
            "answers": [["문서"]],
            "elements_json": "[]",
            "language": "ko",
            "image_width": 8,
            "image_height": 16,
        },
    ]


def test_udd_student_dataset_expands_qas_and_grounding_without_losing_groups():
    from docvlm_eval.student.data import UDDStudentDataset

    dataset = UDDStudentDataset(_udd_rows())

    assert len(dataset) == 4
    assert dataset.tasks == ["vqa", "vqa", "localization", "recognition"]
    assert dataset.sources == ["docvqa", "docvqa", "docvqa", "synthdog_ko"]
    assert dataset.languages == ["en", "en", "en", "ko"]
    assert dataset.aspect_ratios == [2.0, 2.0, 2.0, 0.5]
    assert dataset.sample_ids == [
        "doc-1:qa0",
        "doc-1:qa1",
        "doc-1:box0",
        "doc-2:qa0",
    ]
    assert dataset[0].answer == "42"
    assert dataset[1].answer == "USD"
    assert 'total containing "42"' in dataset[2].prompt
    assert dataset[2].box == (0.1, 0.2, 0.5, 0.6)
    assert dataset[2].box_normalized is True
    assert dataset[0].image_key == dataset[2].image_key


def test_collator_config_is_owned_by_the_machine_readable_blueprint():
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.data import StudentCollatorConfig

    config = StudentCollatorConfig.from_blueprint(
        load_blueprint("configs/sub1b_architecture.yaml")
    )

    assert config.max_length == 2048
    assert config.max_image_long_side == 896
    assert config.patch_size == 14
    assert config.max_visual_tokens == 4096
    assert config.vocab_size == 64000
    assert config.rotation_probability == 1.0
    assert config.visual_canvas_mode == "batch_adaptive"
    assert config.visual_sequence_mode == "packed"
    assert config.packed_attention_backend == "auto"


def test_rotate_normalized_box_covers_all_quarter_turns():
    from docvlm_eval.student.data import rotate_normalized_box

    box = (0.1, 0.2, 0.4, 0.6)

    assert rotate_normalized_box(box, 0) == pytest.approx((0.1, 0.2, 0.4, 0.6))
    assert rotate_normalized_box(box, 1) == pytest.approx((0.4, 0.1, 0.8, 0.4))
    assert rotate_normalized_box(box, 2) == pytest.approx((0.6, 0.4, 0.9, 0.8))
    assert rotate_normalized_box(box, 3) == pytest.approx((0.2, 0.6, 0.6, 0.9))


def test_collator_masks_prompts_and_transforms_boxes_to_the_padded_canvas():
    import torch

    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        UDDStudentDataset,
    )

    dataset = UDDStudentDataset(_udd_rows())
    collator = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            max_length=128,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            rotation_probability=0.0,
        ),
    )

    batch = collator([dataset[0], dataset[2]])

    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["pixel_mask"].shape == (2, 32, 32)
    assert batch["pixel_mask"].sum(dim=(1, 2)).tolist() == [200, 200]
    assert batch["orientation_labels"].tolist() == [0, 0]
    assert batch["contrastive_ids"].tolist() == [0, 0]
    assert batch["box_target_mask"].tolist() == [False, True]
    assert torch.allclose(
        batch["box_targets"][1],
        torch.tensor([0.0625, 0.0625, 0.3125, 0.1875]),
    )
    for row, position in enumerate(batch["box_query_positions"].tolist()):
        assert torch.all(batch["labels"][row, : position + 1] == -100)
        assert batch["labels"][row, position + 1] != -100
    assert torch.all(batch["labels"][~batch["attention_mask"]] == -100)


def test_batch_adaptive_canvas_reduces_dense_visual_tokens_without_resizing():
    import torch

    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        UDDStudentDataset,
    )
    from docvlm_eval.student.compute import estimate_batch_training_flops
    from docvlm_eval.student.config import StudentConfig

    dataset = UDDStudentDataset(_udd_rows())
    adaptive = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            max_length=128,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            rotation_probability=0.0,
            visual_canvas_mode="batch_adaptive",
        ),
    )
    fixed = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            max_length=128,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            rotation_probability=0.0,
            visual_canvas_mode="fixed_square",
        ),
    )

    batch = adaptive([dataset[0], dataset[2]])
    fixed_batch = fixed([dataset[0], dataset[2]])

    assert batch["pixel_values"].shape == (2, 3, 16, 24)
    visual_batch = batch["metadata"]["visual_batch"]
    assert visual_batch["height"] == 16
    assert visual_batch["width"] == 24
    assert visual_batch["coordinate_canvas_height"] == 32
    assert visual_batch["coordinate_canvas_width"] == 32
    assert visual_batch["dense_patch_tokens_per_image"] == 6
    assert visual_batch["valid_pixel_fraction"] == pytest.approx(
        200 / (16 * 24)
    )
    assert torch.allclose(
        batch["box_targets"][1],
        torch.tensor([0.0625, 0.0625, 0.3125, 0.1875]),
    )
    assert torch.equal(batch["box_targets"], fixed_batch["box_targets"])
    assert estimate_batch_training_flops(
        StudentConfig.tiny(),
        batch,
    ) < estimate_batch_training_flops(
        StudentConfig.tiny(),
        fixed_batch,
    )


def test_packed_visual_sequences_match_dense_outputs_without_padding_compute():
    import torch

    from docvlm_eval.student.compute import estimate_batch_training_flops
    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        UDDStudentDataset,
        student_model_inputs,
    )
    from docvlm_eval.student.model import DocumentVLMStudent

    dataset = UDDStudentDataset(_udd_rows())
    common = {
        "max_length": 128,
        "max_image_long_side": 32,
        "patch_size": 8,
        "max_visual_tokens": 16,
        "rotation_probability": 0.0,
    }
    dense = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            **common,
            visual_canvas_mode="batch_adaptive",
            visual_sequence_mode="dense",
        ),
    )([dataset[0], dataset[3]])
    packed = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            **common,
            visual_sequence_mode="packed",
        ),
    )([dataset[0], dataset[3]])

    assert packed["packed_pixel_values"].shape == (8, 3, 8, 8)
    assert packed["packed_position_ids"].tolist() == [0, 1, 2, 4, 5, 6, 0, 4]
    assert packed["packed_cu_seqlens"].tolist() == [0, 6, 8]
    assert "pixel_values" not in packed
    assert packed["metadata"]["visual_batch"]["executed_patch_tokens"] == 8

    torch.manual_seed(83)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    with torch.no_grad():
        dense_output = model(
            **student_model_inputs(dense),
            feature_layers={"vision": [0, -1]},
        )
        packed_output = model(
            **student_model_inputs(packed),
            feature_layers={"vision": [0, -1]},
        )

    assert torch.allclose(packed_output.logits, dense_output.logits, atol=2e-5)
    assert torch.allclose(packed_output.loss, dense_output.loss, atol=2e-5)
    assert torch.allclose(
        packed_output.orientation_logits,
        dense_output.orientation_logits,
        atol=2e-5,
    )
    assert packed_output.vision_mask.shape == (8,)
    assert packed_output.visual_attention_backend == "loop"
    for layer in (0, -1):
        assert torch.allclose(
            packed_output.vision_features[layer],
            dense_output.vision_features[layer][dense_output.vision_mask],
            atol=2e-5,
        )
    dense_generated = model.generate(
        dense["input_ids"],
        pixel_values=dense["pixel_values"],
        pixel_mask=dense["pixel_mask"],
        max_new_tokens=2,
    )
    packed_generated = model.generate(
        packed["input_ids"],
        packed_pixel_values=packed["packed_pixel_values"],
        packed_position_ids=packed["packed_position_ids"],
        packed_cu_seqlens=packed["packed_cu_seqlens"],
        max_new_tokens=2,
    )
    assert torch.equal(packed_generated, dense_generated)
    assert estimate_batch_training_flops(
        StudentConfig.tiny(),
        packed,
    ) < estimate_batch_training_flops(
        StudentConfig.tiny(),
        dense,
    )


def test_collated_udd_batch_runs_all_available_student_losses():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        UDDStudentDataset,
        student_model_inputs,
    )
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(4)
    dataset = UDDStudentDataset(_udd_rows())
    collator = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(
            max_length=128,
            max_image_long_side=32,
            patch_size=8,
            max_visual_tokens=16,
            rotation_probability=1.0,
            contrastive=True,
        ),
    )
    batch = collator([dataset[0], dataset[2]])
    model = DocumentVLMStudent(StudentConfig.tiny())

    output = model(**student_model_inputs(batch))

    assert set(output.losses) == {
        "autoregressive",
        "box_regression",
        "orientation",
        "region_text_contrastive",
    }
    assert output.loss is not None and torch.isfinite(output.loss)
    assert torch.isfinite(output.box_predictions).all()


def test_visual_padding_mask_blocks_fully_padded_patches():
    import torch

    from docvlm_eval.student.config import StudentConfig
    from docvlm_eval.student.model import DocumentVLMStudent

    torch.manual_seed(8)
    model = DocumentVLMStudent(StudentConfig.tiny()).eval()
    pixels = torch.randn(1, 3, 32, 32)
    mask = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask[:, :16, :16] = True
    changed = pixels.clone()
    changed[:, :, 16:, :] = torch.randn_like(changed[:, :, 16:, :]) * 100
    changed[:, :, :16, 16:] = torch.randn_like(changed[:, :, :16, 16:]) * 100
    ids = torch.randint(0, 256, (1, 6))

    expected = model(ids, pixel_values=pixels, pixel_mask=mask).logits
    actual = model(ids, pixel_values=changed, pixel_mask=mask).logits
    _, patch_mask = model.vision(pixels, mask, return_mask=True)

    assert patch_mask.sum().item() == 4
    assert torch.allclose(expected, actual)


def test_multi_positive_contrastive_does_not_make_same_image_views_negatives():
    import torch

    from docvlm_eval.student.model import _multi_positive_contrastive_loss

    similarities = torch.tensor([[4.0, 1.0], [2.0, 3.0]])

    same_image = _multi_positive_contrastive_loss(
        similarities,
        torch.tensor([7, 7]),
    )
    different_images = _multi_positive_contrastive_loss(
        similarities,
        torch.tensor([7, 8]),
    )

    assert torch.allclose(same_image, torch.tensor(0.0))
    assert different_images > 0


def test_balanced_sampler_uses_explicit_group_weights_and_epoch_seed():
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    sampler = BalancedGroupBatchSampler(
        ["rare", "common", "common"],
        batch_size=4,
        group_weights={"rare": 1.0, "common": 0.0},
        num_batches=3,
        seed=5,
    )

    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert first_epoch == [[0, 0, 0, 0]] * 3
    assert second_epoch == [[0, 0, 0, 0]] * 3


def test_balanced_sampler_reads_the_blueprint_grouping_policy():
    from docvlm_eval.architecture import load_blueprint
    from docvlm_eval.student.data import BalancedGroupBatchSampler, UDDStudentDataset

    dataset = UDDStudentDataset(_udd_rows())
    sampler = BalancedGroupBatchSampler.from_blueprint(
        dataset,
        load_blueprint("configs/sub1b_architecture.yaml"),
        batch_size=2,
        num_batches=1,
    )

    assert sampler.group_names == ["localization", "recognition", "vqa"]
    assert sampler.aspect_ratio_bucketing is False
    assert len(list(sampler)) == 1


def test_aspect_bucket_sampler_keeps_each_global_batch_shape_homogeneous():
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    ratios = [0.25, 0.3, 1.0, 1.1, 3.5, 4.0]
    sampler = BalancedGroupBatchSampler(
        ["a", "b", "a", "b", "a", "b"],
        batch_size=6,
        num_batches=20,
        seed=19,
        aspect_ratios=ratios,
        sample_ids=[f"sample-{index}" for index in range(len(ratios))],
        aspect_ratio_bucketing=True,
        rotation_probability=0.0,
    )

    for batch in sampler:
        assert len({sampler._aspect_bucket(index) for index in batch}) == 1


def test_aspect_bucket_sampler_preserves_target_group_distribution_in_expectation():
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    groups = ["rare"] * 2 + ["common"] * 8
    ratios = [0.25, 4.0] + [0.25] * 7 + [4.0]
    sampler = BalancedGroupBatchSampler(
        groups,
        batch_size=1,
        group_weights={"rare": 3.0, "common": 1.0},
        num_batches=20_000,
        seed=29,
        aspect_ratios=ratios,
        sample_ids=[f"sample-{index}" for index in range(len(groups))],
        aspect_ratio_bucketing=True,
        rotation_probability=0.0,
    )

    rare_fraction = sum(
        groups[batch[0]] == "rare" for batch in sampler
    ) / len(sampler)

    assert rare_fraction == pytest.approx(0.75, abs=0.015)


def test_aspect_bucket_sampler_shares_a_bucket_across_distributed_ranks():
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    kwargs = {
        "groups": ["a", "b", "a", "b"],
        "batch_size": 2,
        "num_batches": 12,
        "seed": 31,
        "num_replicas": 2,
        "aspect_ratios": [0.25, 0.3, 3.5, 4.0],
        "sample_ids": ["p0", "p1", "l0", "l1"],
        "aspect_ratio_bucketing": True,
        "rotation_probability": 0.0,
    }
    rank_zero = BalancedGroupBatchSampler(rank=0, **kwargs)
    rank_one = BalancedGroupBatchSampler(rank=1, **kwargs)

    for zero_batch, one_batch in zip(rank_zero, rank_one):
        global_batch = [*zero_batch, *one_batch]
        assert len(
            {rank_zero._aspect_bucket(index) for index in global_batch}
        ) == 1


def test_aspect_buckets_follow_the_same_epoch_rotation_as_the_collator():
    from docvlm_eval.student.data import (
        BalancedGroupBatchSampler,
        StudentCollator,
        StudentCollatorConfig,
    )

    collator = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(rotation_probability=1.0, augmentation_seed=37),
    )
    sample_id = next(
        f"sample-{index}"
        for index in range(100)
        if collator._quarter_turns(f"sample-{index}") % 2
    )
    sampler = BalancedGroupBatchSampler(
        ["only"],
        batch_size=1,
        aspect_ratios=[4.0],
        sample_ids=[sample_id],
        aspect_ratio_bucketing=True,
        rotation_probability=1.0,
        augmentation_seed=37,
    )

    assert sampler._aspect_bucket(0) < 0
    sampler.set_epoch(3)
    collator.set_epoch(3)
    expected_sign = -1 if collator._quarter_turns(sample_id) % 2 else 1
    assert sampler._aspect_bucket(0) * expected_sign > 0


def test_balanced_sampler_applies_curriculum_weights_at_step_boundaries():
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(
                id="rare-first",
                until_fraction=0.5,
                group_weights={"rare": 1.0, "common": 0.0},
            ),
            CurriculumStage(
                id="common-last",
                until_fraction=1.0,
                group_weights={"rare": 0.0, "common": 1.0},
            ),
        )
    )
    sampler = BalancedGroupBatchSampler(
        ["rare", "common"],
        batch_size=1,
        num_batches=4,
        curriculum=schedule,
        grad_accum_steps=1,
        epochs=1,
        max_steps=4,
    )

    assert list(sampler) == [[0], [0], [1], [1]]


def test_balanced_sampler_rejects_unknown_curriculum_groups():
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    schedule = CurriculumSchedule(
        stages=(
            CurriculumStage(
                id="invalid",
                until_fraction=1.0,
                group_weights={"missing": 1.0},
            ),
        )
    )

    with pytest.raises(ValueError, match="unknown groups"):
        BalancedGroupBatchSampler(
            ["available"],
            batch_size=1,
            curriculum=schedule,
        )


def test_balanced_sampler_rejects_token_fraction_group_weights():
    from docvlm_eval.student.curriculum import CurriculumSchedule, CurriculumStage
    from docvlm_eval.student.data import BalancedGroupBatchSampler

    schedule = CurriculumSchedule(
        unit="training_token_fraction",
        stages=(
            CurriculumStage(
                id="not-prefetch-safe",
                until_fraction=1.0,
                group_weights={"available": 1.0},
            ),
        ),
    )

    with pytest.raises(ValueError, match="prefetched sampler"):
        BalancedGroupBatchSampler(
            ["available"],
            batch_size=1,
            curriculum=schedule,
        )


def test_exhaustive_sampler_covers_every_example_once_on_one_rank():
    from docvlm_eval.student.data import DeterministicDistributedBatchSampler

    sampler = DeterministicDistributedBatchSampler(
        dataset_size=7,
        batch_size=3,
        seed=41,
    )
    first_epoch = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert [len(batch) for batch in first_epoch] == [3, 3, 1]
    assert sorted(index for batch in first_epoch for index in batch) == list(range(7))
    assert sorted(index for batch in second_epoch for index in batch) == list(range(7))
    assert first_epoch != second_epoch


def test_exhaustive_sampler_pads_only_for_distributed_batch_alignment():
    from docvlm_eval.student.data import DeterministicDistributedBatchSampler

    rank_zero = DeterministicDistributedBatchSampler(
        dataset_size=5,
        batch_size=2,
        seed=43,
        num_replicas=2,
        rank=0,
    )
    rank_one = DeterministicDistributedBatchSampler(
        dataset_size=5,
        batch_size=2,
        seed=43,
        num_replicas=2,
        rank=1,
    )
    zero_batches = list(rank_zero)
    one_batches = list(rank_one)
    observed = [
        index
        for rank_batches in (zero_batches, one_batches)
        for batch in rank_batches
        for index in batch
    ]

    assert len(zero_batches) == len(one_batches) == 2
    assert all(len(batch) == 2 for batch in zero_batches + one_batches)
    assert set(observed) == set(range(5))
    assert len(observed) == 8


def test_text_only_replay_batch_omits_visual_targets():
    from docvlm_eval.student.data import (
        StudentCollator,
        StudentCollatorConfig,
        StudentExample,
    )

    collator = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(max_length=64),
    )
    batch = collator(
        [
            StudentExample(
                sample_id="text-1",
                source="replay",
                task="language",
                prompt="Continue:",
                answer="A compact language replay target.",
            )
        ]
    )

    assert "pixel_values" not in batch
    assert "pixel_mask" not in batch
    assert "orientation_labels" not in batch
    assert "box_targets" not in batch


def test_rotation_augmentation_is_stable_for_exact_resume():
    from docvlm_eval.student.data import StudentCollator, StudentCollatorConfig

    collator = StudentCollator(
        _CharacterTokenizer(),
        StudentCollatorConfig(rotation_probability=1.0, augmentation_seed=17),
    )

    epoch_zero = collator._quarter_turns("sample-42")
    assert collator._quarter_turns("sample-42") == epoch_zero
    collator.set_epoch(3)
    epoch_three = collator._quarter_turns("sample-42")
    assert collator._quarter_turns("sample-42") == epoch_three

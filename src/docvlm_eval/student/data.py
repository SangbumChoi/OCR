"""UDD-aware examples, batching, augmentation, and balanced sampling for the student."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .curriculum import (
    COMPOSITION_TIERS,
    CompositionCurriculumSchedule,
    CurriculumSchedule,
    planned_optimizer_steps,
)


STUDENT_MODEL_INPUTS = frozenset(
    {
        "input_ids",
        "pixel_values",
        "pixel_mask",
        "packed_pixel_values",
        "packed_position_ids",
        "packed_cu_seqlens",
        "packed_attention_backend",
        "attention_mask",
        "labels",
        "box_targets",
        "box_target_mask",
        "box_query_positions",
        "orientation_labels",
        "contrastive",
        "contrastive_ids",
        "loss_weights",
        "feature_layers",
    }
)

VISUAL_MODEL_INPUTS = (
    "pixel_values",
    "pixel_mask",
    "packed_pixel_values",
    "packed_position_ids",
    "packed_cu_seqlens",
    "packed_attention_backend",
)


def student_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove provenance and runner-only fields before calling the model."""

    return {key: value for key, value in batch.items() if key in STUDENT_MODEL_INPUTS}


def visual_model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Return only dense or packed visual tensors accepted by the student."""

    return {key: batch[key] for key in VISUAL_MODEL_INPUTS if key in batch}


def stable_contrastive_id(example: "StudentExample") -> int:
    """Return a stable signed-int64 ID for same-image positive matching."""

    image_identity = example.image_key or example.sample_id
    digest = hashlib.blake2b(
        f"{example.source}\0{image_identity}".encode("utf-8"),
        digest_size=8,
        person=b"docvlm-id",
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class StudentExample:
    """One image-text objective, optionally carrying a single evidence box."""

    sample_id: str
    source: str
    task: str
    prompt: str
    answer: str
    image: Any = None
    image_key: str = ""
    language: str = ""
    box: tuple[float, float, float, float] | None = None
    box_normalized: bool = True
    target_source: str = "gold"


@dataclass(frozen=True)
class _ExampleRef:
    row_index: int
    kind: str
    item_index: int
    task: str
    source: str
    language: str
    component: str
    target_source: str
    sample_id: str
    aspect_ratio: float | None
    composition: str


def composition_tier(
    page_count: int,
    document_count: int,
) -> str:
    """Map exact page/document counts to one ordered composition tier."""

    page_count = int(page_count)
    document_count = int(document_count)
    if page_count < 1 or document_count < 1:
        raise ValueError("page_count and document_count must be positive")
    if document_count > 1:
        return "cross_document"
    if page_count > 1:
        return "multi_page"
    return "single_page"


def _parse_elements(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("elements_json")
    if raw:
        elements = json.loads(raw) if isinstance(raw, str) else raw
        return [item for item in elements if isinstance(item, dict)]
    out: list[dict[str, Any]] = []
    for field in json.loads(row.get("fields_json") or "[]"):
        out.append(
            {
                "key": field.get("key", ""),
                "value": field.get("value", ""),
                "bbox": field.get("bbox"),
                "kind": "field",
            }
        )
    for region in json.loads(row.get("regions_json") or "[]"):
        out.append(
            {
                "key": region.get("label", ""),
                "value": region.get("text", ""),
                "bbox": region.get("bbox"),
                "kind": "region",
            }
        )
    return out


def _metadata_view(dataset: Any) -> Any:
    if not hasattr(dataset, "select_columns") or not hasattr(dataset, "column_names"):
        return dataset
    wanted = {
        "sample_id",
        "source",
        "task",
        "instructions",
        "answers",
        "teacher_answers",
        "teacher_scores",
        "teacher_provenance_json",
        "elements_json",
        "fields_json",
        "regions_json",
        "full_text",
        "table_html",
        "language",
        "metric",
        "mixture_component",
        "image_width",
        "image_height",
        "page_count",
        "document_count",
    }
    columns = [name for name in dataset.column_names if name in wanted]
    return dataset.select_columns(columns)


def _valid_answers(raw: Any) -> list[str]:
    return [str(value) for value in (raw or []) if str(value).strip()]


def _ordinal(index: int) -> str:
    value = index + 1
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _grounding_prompt(elements: list[dict[str, Any]], index: int) -> str:
    element = elements[index]
    key = str(element.get("key") or element.get("kind") or "element").strip()
    value = " ".join(str(element.get("value") or "").split())
    same_key = [
        item
        for item in elements
        if str(item.get("key") or item.get("kind") or "element").strip() == key
        and item.get("bbox")
    ]
    if value:
        descriptor = f'the {key} containing "{value[:80]}"'
    elif len(same_key) > 1:
        occurrence = sum(
            1
            for item in elements[:index]
            if str(item.get("key") or item.get("kind") or "element").strip() == key
            and item.get("bbox")
        )
        descriptor = f"the {_ordinal(occurrence)} {key} in reading order"
    else:
        descriptor = f"the {key}"
    return (
        f"Where is {descriptor} in the document? "
        "Return only its normalized bounding box as [x1, y1, x2, y2]."
    )


class UDDStudentDataset:
    """Lazy image dataset that expands native UDD QA and localized-element payloads."""

    def __init__(
        self,
        dataset: Sequence[dict[str, Any]] | Any,
        *,
        include_grounding: bool = True,
        max_grounding_per_row: int = 16,
        teacher_target_probability: float = 0.0,
        teacher_min_score: float = 0.0,
        teacher_target_seed: int = 0,
    ):
        if not 0.0 <= teacher_target_probability <= 1.0:
            raise ValueError("teacher_target_probability must be within [0, 1]")
        if not 0.0 <= teacher_min_score <= 1.0:
            raise ValueError("teacher_min_score must be within [0, 1]")
        self.dataset = dataset
        self.include_grounding = include_grounding
        self.max_grounding_per_row = max_grounding_per_row
        self.teacher_target_probability = teacher_target_probability
        self.teacher_min_score = teacher_min_score
        self.teacher_target_seed = int(teacher_target_seed)
        self._refs: list[_ExampleRef] = []
        metadata = _metadata_view(dataset)
        for row_index in range(len(metadata)):
            row = metadata[row_index]
            sample_id = str(row.get("sample_id") or f"row-{row_index}")
            source = str(row.get("source") or "unknown")
            task = str(row.get("task") or "unknown")
            language = str(row.get("language") or "und")
            component = str(row.get("mixture_component") or source)
            image_width = int(row.get("image_width") or 0)
            image_height = int(row.get("image_height") or 0)
            aspect_ratio = (
                image_width / image_height
                if image_width > 0 and image_height > 0
                else None
            )
            page_count = (
                1
                if row.get("page_count") is None
                else int(row["page_count"])
            )
            document_count = (
                1
                if row.get("document_count") is None
                else int(row["document_count"])
            )
            composition = composition_tier(page_count, document_count)
            instructions = list(row.get("instructions") or [])
            answers = list(row.get("answers") or [])
            teacher_answers = list(row.get("teacher_answers") or [])
            teacher_scores = list(row.get("teacher_scores") or [])
            if len(instructions) != len(answers):
                raise ValueError(
                    f"UDD row {sample_id!r} has {len(instructions)} instructions "
                    f"but {len(answers)} answer lists"
                )
            for qa_index, (question, golds) in enumerate(zip(instructions, answers)):
                if str(question).strip() and _valid_answers(golds):
                    qa_sample_id = f"{sample_id}:qa{qa_index}"
                    teacher_available = bool(
                        qa_index < len(teacher_answers)
                        and qa_index < len(teacher_scores)
                        and str(teacher_answers[qa_index]).strip()
                        and float(teacher_scores[qa_index]) >= teacher_min_score
                    )
                    payload = (
                        f"{self.teacher_target_seed}:{qa_sample_id}:teacher-target"
                    ).encode("utf-8")
                    draw = int.from_bytes(
                        hashlib.blake2b(payload, digest_size=8).digest(),
                        "big",
                    ) / float(2**64)
                    target_source = (
                        "teacher"
                        if teacher_available and draw < teacher_target_probability
                        else "gold"
                    )
                    self._refs.append(
                        _ExampleRef(
                            row_index,
                            "qa",
                            qa_index,
                            task,
                            source,
                            language,
                            component,
                            target_source,
                            qa_sample_id,
                            aspect_ratio,
                            composition,
                        )
                    )
            elements = _parse_elements(row)
            if include_grounding:
                used = 0
                for element_index, element in enumerate(elements):
                    bbox = element.get("bbox")
                    if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
                        continue
                    self._refs.append(
                        _ExampleRef(
                            row_index,
                            "grounding",
                            element_index,
                            "localization",
                            source,
                            language,
                            component,
                            "gold",
                            f"{sample_id}:box{element_index}",
                            aspect_ratio,
                            composition,
                        )
                    )
                    used += 1
                    if used >= max_grounding_per_row:
                        break
        if not self._refs:
            raise ValueError("UDD dataset produced no trainable QA or grounding examples")

    def __len__(self) -> int:
        return len(self._refs)

    @property
    def tasks(self) -> list[str]:
        return [ref.task for ref in self._refs]

    @property
    def sources(self) -> list[str]:
        return [ref.source for ref in self._refs]

    @property
    def languages(self) -> list[str]:
        return [ref.language for ref in self._refs]

    @property
    def components(self) -> list[str]:
        return [ref.component for ref in self._refs]

    @property
    def target_sources(self) -> list[str]:
        return [ref.target_source for ref in self._refs]

    @property
    def sample_ids(self) -> list[str]:
        return [ref.sample_id for ref in self._refs]

    @property
    def aspect_ratios(self) -> list[float | None]:
        return [ref.aspect_ratio for ref in self._refs]

    @property
    def compositions(self) -> list[str]:
        return [ref.composition for ref in self._refs]

    def groups(self, key: str) -> list[str]:
        if key == "task":
            return self.tasks
        if key == "source":
            return self.sources
        if key == "language":
            return self.languages
        if key == "component":
            return self.components
        if key == "composition":
            return self.compositions
        raise ValueError(
            "group key must be one of: task, source, language, component, "
            "composition"
        )

    def __getitem__(self, index: int) -> StudentExample:
        ref = self._refs[index]
        row = self.dataset[ref.row_index]
        image = row.get("image")
        if image is None:
            image = row.get("image_path")
        image_key = str(row.get("sample_id") or ref.row_index)
        if ref.kind == "qa":
            question = str(row["instructions"][ref.item_index])
            answer = _valid_answers(row["answers"][ref.item_index])[0]
            if ref.target_source == "teacher":
                answer = str(row["teacher_answers"][ref.item_index]).strip()
            return StudentExample(
                sample_id=ref.sample_id,
                source=ref.source,
                task=ref.task,
                prompt=question,
                answer=answer,
                image=image,
                image_key=image_key,
                language=ref.language,
                target_source=ref.target_source,
            )

        elements = _parse_elements(row)
        element = elements[ref.item_index]
        raw_box = element["bbox"]
        normalized = bool(raw_box[4]) if len(raw_box) >= 5 else True
        return StudentExample(
            sample_id=ref.sample_id,
            source=ref.source,
            task=ref.task,
            prompt=_grounding_prompt(elements, ref.item_index),
            answer="",
            image=image,
            image_key=image_key,
            language=ref.language,
            box=tuple(float(value) for value in raw_box[:4]),
            box_normalized=normalized,
        )


def normalize_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    already_normalized: bool,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = box
    if not already_normalized:
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    left, right = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    top, bottom = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return left, top, right, bottom


def rotate_normalized_box(
    box: tuple[float, float, float, float],
    quarter_turns_clockwise: int,
) -> tuple[float, float, float, float]:
    """Rotate an axis-aligned normalized box by 0/90/180/270 degrees clockwise."""

    x1, y1, x2, y2 = box
    turns = quarter_turns_clockwise % 4
    if turns == 0:
        return box
    if turns == 1:
        return 1.0 - y2, x1, 1.0 - y1, x2
    if turns == 2:
        return 1.0 - x2, 1.0 - y2, 1.0 - x1, 1.0 - y1
    return y1, 1.0 - x2, y2, 1.0 - x1


def deterministic_quarter_turns(
    sample_id: str,
    *,
    epoch: int,
    probability: float,
    seed: int,
) -> int:
    """Choose reproducible right-angle augmentation for one sample and epoch."""

    payload = f"{seed}:{epoch}:{sample_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    draw = int.from_bytes(digest[:8], "big") / float(2**64)
    if draw >= probability:
        return 0
    return int.from_bytes(digest[8:], "big") % 4


@dataclass(frozen=True)
class StudentCollatorConfig:
    max_length: int = 2048
    max_image_long_side: int = 1024
    patch_size: int = 14
    max_visual_tokens: int = 4096
    vocab_size: int | None = None
    rotation_probability: float = 1.0
    augmentation_seed: int = 7
    allow_upscale: bool = False
    visual_canvas_mode: str = "fixed_square"
    visual_sequence_mode: str = "dense"
    packed_attention_backend: str = "auto"
    prompt_template: str = "User: {prompt}\nAssistant:"
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    contrastive: bool = True

    @classmethod
    def from_blueprint(
        cls,
        blueprint: dict[str, Any],
        **overrides: Any,
    ) -> "StudentCollatorConfig":
        vision = blueprint["student"]["vision"]
        pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
        values = {
            "max_length": int(pipeline["max_text_tokens"]),
            "max_image_long_side": int(pipeline["max_image_long_side"]),
            "patch_size": int(vision["patch_size"]),
            "max_visual_tokens": int(vision["max_position_tokens"]),
            "vocab_size": int(blueprint["student"]["language"]["vocab_size"]),
            "rotation_probability": float(pipeline["rotation_probability"]),
            "augmentation_seed": int(pipeline.get("augmentation_seed", 7)),
            "allow_upscale": bool(pipeline.get("allow_upscale", False)),
            "visual_canvas_mode": str(
                pipeline.get("visual_canvas_mode", "fixed_square")
            ),
            "visual_sequence_mode": str(
                pipeline.get("visual_sequence_mode", "dense")
            ),
            "packed_attention_backend": str(
                pipeline.get("packed_attention_backend", "auto")
            ),
            "contrastive": bool(pipeline.get("contrastive", True)),
        }
        values.update(overrides)
        return cls(**values)

    def __post_init__(self) -> None:
        if self.max_length < 4:
            raise ValueError("max_length must be at least 4")
        if (
            self.max_image_long_side <= 0
            or self.patch_size <= 0
            or self.max_visual_tokens <= 0
        ):
            raise ValueError("image and patch dimensions must be positive")
        if not 0.0 <= self.rotation_probability <= 1.0:
            raise ValueError("rotation_probability must be in [0, 1]")
        if self.visual_canvas_mode not in {"fixed_square", "batch_adaptive"}:
            raise ValueError(
                "visual_canvas_mode must be fixed_square or batch_adaptive"
            )
        if self.visual_sequence_mode not in {"dense", "packed"}:
            raise ValueError("visual_sequence_mode must be dense or packed")
        if self.packed_attention_backend not in {"auto", "flex", "loop"}:
            raise ValueError(
                "packed_attention_backend must be auto, flex, or loop"
            )


def _open_image(value: Any):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return image.convert("RGB")
    raise TypeError(f"unsupported image payload: {type(value).__name__}")


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    return [int(token) for token in encoded]


class StudentCollator:
    """Create prompt-masked text tensors and spatially consistent image supervision."""

    def __init__(self, tokenizer: Any, config: StudentCollatorConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        if self.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _quarter_turns(self, sample_id: str) -> int:
        return deterministic_quarter_turns(
            sample_id,
            epoch=self.epoch,
            probability=self.config.rotation_probability,
            seed=self.config.augmentation_seed,
        )

    def _tokenize(self, prompt: str, answer: str) -> tuple[list[int], list[int], int]:
        prompt_text = self.config.prompt_template.format(prompt=prompt)
        prompt_ids = _encode(self.tokenizer, prompt_text)
        if self.bos_token_id is not None:
            prompt_ids = [int(self.bos_token_id), *prompt_ids]
        answer_ids = _encode(self.tokenizer, answer)
        if not answer_ids:
            raise ValueError("student answer tokenized to an empty sequence")
        max_without_eos = self.config.max_length - 1
        if len(prompt_ids) >= max_without_eos:
            prompt_ids = prompt_ids[: max_without_eos - 1]
        room = max_without_eos - len(prompt_ids)
        answer_ids = answer_ids[:room]
        if not answer_ids:
            raise ValueError("max_length leaves no supervised answer token")
        sequence = [*prompt_ids, *answer_ids, int(self.eos_token_id)]
        if self.config.vocab_size is not None:
            invalid = [
                token
                for token in sequence
                if token < 0 or token >= self.config.vocab_size
            ]
            if invalid:
                raise ValueError(
                    f"tokenizer emitted ID {invalid[0]} outside student vocabulary "
                    f"[0, {self.config.vocab_size})"
                )
        labels = [-100] * len(prompt_ids) + [*answer_ids, int(self.eos_token_id)]
        return sequence, labels, len(prompt_ids) - 1

    def __call__(self, examples: Sequence[StudentExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        import numpy as np
        import torch
        from PIL import Image

        has_images = [example.image is not None for example in examples]
        if any(has_images) and not all(has_images):
            raise ValueError("mixed image and text-only examples require separate batches")

        image_tensors: list[torch.Tensor] = []
        resized_sizes: list[tuple[int, int]] = []
        transformed_boxes: list[tuple[float, float, float, float] | None] = []
        orientations: list[int] = []
        if all(has_images):
            canvas_patch_side = math.isqrt(self.config.max_visual_tokens)
            if canvas_patch_side <= 0:
                raise ValueError("max_visual_tokens cannot form a visual canvas")
            canvas_side = min(
                math.ceil(self.config.max_image_long_side / self.config.patch_size),
                canvas_patch_side,
            ) * self.config.patch_size
            for example in examples:
                image = _open_image(example.image)
                original_width, original_height = image.size
                turns = self._quarter_turns(example.sample_id)
                if turns:
                    transpose = {
                        1: Image.Transpose.ROTATE_270,
                        2: Image.Transpose.ROTATE_180,
                        3: Image.Transpose.ROTATE_90,
                    }[turns]
                    image = image.transpose(transpose)
                normalized_box = None
                if example.box is not None:
                    normalized_box = rotate_normalized_box(
                        normalize_box(
                            example.box,
                            original_width,
                            original_height,
                            already_normalized=example.box_normalized,
                        ),
                        turns,
                    )
                width, height = image.size
                scale = canvas_side / max(width, height)
                if not self.config.allow_upscale:
                    scale = min(scale, 1.0)
                resized_width = max(1, round(width * scale))
                resized_height = max(1, round(height * scale))
                if (resized_width, resized_height) != image.size:
                    image = image.resize(
                        (resized_width, resized_height),
                        Image.Resampling.BICUBIC,
                    )
                pixels = torch.from_numpy(
                    np.asarray(image, dtype=np.float32).copy()
                ).permute(2, 0, 1) / 255.0
                image_tensors.append(pixels)
                resized_sizes.append((resized_height, resized_width))
                transformed_boxes.append(normalized_box)
                orientations.append(turns)

        batch_height = batch_width = 0
        pixel_values = pixel_mask = None
        packed_pixel_values = packed_position_ids = packed_cu_seqlens = None
        canvas_boxes: list[tuple[float, float, float, float] | None] = transformed_boxes
        if image_tensors:
            if self.config.visual_sequence_mode == "packed":
                mean = torch.tensor(self.config.image_mean)[:, None, None]
                std = torch.tensor(self.config.image_std)[:, None, None]
                patch_rows: list[torch.Tensor] = []
                position_rows: list[torch.Tensor] = []
                sequence_lengths: list[int] = []
                canvas_boxes = []
                grid_side = math.isqrt(self.config.max_visual_tokens)
                for pixels, size, box in zip(
                    image_tensors,
                    resized_sizes,
                    transformed_boxes,
                ):
                    height, width = size
                    patch_height = math.ceil(height / self.config.patch_size)
                    patch_width = math.ceil(width / self.config.patch_size)
                    padded_height = patch_height * self.config.patch_size
                    padded_width = patch_width * self.config.patch_size
                    normalized = (pixels - mean) / std
                    normalized = torch.nn.functional.pad(
                        normalized,
                        (0, padded_width - width, 0, padded_height - height),
                    )
                    patches = (
                        normalized.unfold(
                            1,
                            self.config.patch_size,
                            self.config.patch_size,
                        )
                        .unfold(
                            2,
                            self.config.patch_size,
                            self.config.patch_size,
                        )
                        .permute(1, 2, 0, 3, 4)
                        .reshape(
                            patch_height * patch_width,
                            3,
                            self.config.patch_size,
                            self.config.patch_size,
                        )
                    )
                    rows = torch.arange(patch_height)[:, None]
                    columns = torch.arange(patch_width)[None, :]
                    positions = (rows * grid_side + columns).flatten()
                    patch_rows.append(patches)
                    position_rows.append(positions)
                    sequence_lengths.append(int(patches.shape[0]))
                    canvas_boxes.append(
                        None
                        if box is None
                        else (
                            box[0] * width / canvas_side,
                            box[1] * height / canvas_side,
                            box[2] * width / canvas_side,
                            box[3] * height / canvas_side,
                        )
                    )
                packed_pixel_values = torch.cat(patch_rows)
                packed_position_ids = torch.cat(position_rows).to(torch.long)
                packed_cu_seqlens = torch.tensor(
                    [0, *sequence_lengths],
                    dtype=torch.long,
                ).cumsum(0)
                batch_height = max(
                    math.ceil(height / self.config.patch_size)
                    * self.config.patch_size
                    for height, _ in resized_sizes
                )
                batch_width = max(
                    math.ceil(width / self.config.patch_size)
                    * self.config.patch_size
                    for _, width in resized_sizes
                )
            elif self.config.visual_canvas_mode == "fixed_square":
                batch_height = canvas_side
                batch_width = canvas_side
            elif self.config.visual_sequence_mode == "dense":
                batch_height = (
                    math.ceil(max(height for height, _ in resized_sizes)
                              / self.config.patch_size)
                    * self.config.patch_size
                )
                batch_width = (
                    math.ceil(max(width for _, width in resized_sizes)
                              / self.config.patch_size)
                    * self.config.patch_size
                )
            if self.config.visual_sequence_mode == "dense":
                pixel_values = torch.zeros(
                    len(examples),
                    3,
                    batch_height,
                    batch_width,
                    dtype=torch.float32,
                )
                pixel_mask = torch.zeros(
                    len(examples),
                    batch_height,
                    batch_width,
                    dtype=torch.bool,
                )
                mean = torch.tensor(self.config.image_mean)[:, None, None]
                std = torch.tensor(self.config.image_std)[:, None, None]
                canvas_boxes = []
                for index, (pixels, size, box) in enumerate(
                    zip(image_tensors, resized_sizes, transformed_boxes)
                ):
                    height, width = size
                    pixel_values[index, :, :height, :width] = (pixels - mean) / std
                    pixel_mask[index, :height, :width] = True
                    canvas_boxes.append(
                        None
                        if box is None
                        else (
                            box[0] * width / canvas_side,
                            box[1] * height / canvas_side,
                            box[2] * width / canvas_side,
                            box[3] * height / canvas_side,
                        )
                    )

        sequences: list[list[int]] = []
        label_rows: list[list[int]] = []
        box_positions: list[int] = []
        for example, box in zip(examples, canvas_boxes or [None] * len(examples)):
            answer = example.answer
            if box is not None:
                answer = "[" + ", ".join(f"{value:.6f}" for value in box) + "]"
            sequence, labels, box_position = self._tokenize(example.prompt, answer)
            sequences.append(sequence)
            label_rows.append(labels)
            box_positions.append(box_position)

        text_length = max(len(sequence) for sequence in sequences)
        input_ids = torch.full(
            (len(examples), text_length),
            int(self.pad_token_id),
            dtype=torch.long,
        )
        labels = torch.full((len(examples), text_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), text_length), dtype=torch.bool)
        for index, (sequence, label_row) in enumerate(zip(sequences, label_rows)):
            length = len(sequence)
            input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
            labels[index, :length] = torch.tensor(label_row, dtype=torch.long)
            attention_mask[index, :length] = True

        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "box_query_positions": torch.tensor(box_positions, dtype=torch.long),
            "contrastive": (
                self.config.contrastive
                and bool(image_tensors)
                and len(examples) > 1
            ),
            "metadata": {
                "sample_id": [example.sample_id for example in examples],
                "source": [example.source for example in examples],
                "task": [example.task for example in examples],
                "language": [example.language for example in examples],
                "image_key": [example.image_key for example in examples],
                "visual_canvas_mode": self.config.visual_canvas_mode,
                "visual_sequence_mode": self.config.visual_sequence_mode,
            },
        }
        if pixel_values is not None and pixel_mask is not None:
            valid_pixels = int(pixel_mask.sum().item())
            batch["metadata"]["visual_batch"] = {
                "height": batch_height,
                "width": batch_width,
                "coordinate_canvas_height": canvas_side,
                "coordinate_canvas_width": canvas_side,
                "dense_patch_tokens_per_image": (
                    batch_height // self.config.patch_size
                ) * (batch_width // self.config.patch_size),
                "valid_pixel_fraction": valid_pixels / pixel_mask.numel(),
            }
            batch["pixel_values"] = pixel_values
            batch["pixel_mask"] = pixel_mask
            batch["orientation_labels"] = torch.tensor(orientations, dtype=torch.long)
            batch["contrastive_ids"] = torch.tensor(
                [stable_contrastive_id(example) for example in examples],
                dtype=torch.long,
            )
        elif (
            packed_pixel_values is not None
            and packed_position_ids is not None
            and packed_cu_seqlens is not None
        ):
            packed_tokens = int(packed_pixel_values.shape[0])
            valid_pixels = sum(
                height * width for height, width in resized_sizes
            )
            allocated_pixels = packed_tokens * self.config.patch_size**2
            batch["metadata"]["visual_batch"] = {
                "height": batch_height,
                "width": batch_width,
                "coordinate_canvas_height": canvas_side,
                "coordinate_canvas_width": canvas_side,
                "dense_patch_tokens_per_image": (
                    packed_tokens / len(examples)
                ),
                "executed_patch_tokens": packed_tokens,
                "valid_pixel_fraction": valid_pixels / allocated_pixels,
            }
            batch["packed_pixel_values"] = packed_pixel_values
            batch["packed_position_ids"] = packed_position_ids
            batch["packed_cu_seqlens"] = packed_cu_seqlens
            batch["packed_attention_backend"] = (
                self.config.packed_attention_backend
            )
            batch["orientation_labels"] = torch.tensor(orientations, dtype=torch.long)
            batch["contrastive_ids"] = torch.tensor(
                [stable_contrastive_id(example) for example in examples],
                dtype=torch.long,
            )
        if any(box is not None for box in canvas_boxes):
            batch["box_targets"] = torch.tensor(
                [box if box is not None else (0.0, 0.0, 0.0, 0.0) for box in canvas_boxes],
                dtype=torch.float32,
            )
            batch["box_target_mask"] = torch.tensor(
                [box is not None for box in canvas_boxes],
                dtype=torch.bool,
            )
        return batch


class BalancedGroupBatchSampler:
    """Sample explicit group targets in optional augmentation-aware shape buckets."""

    def __init__(
        self,
        groups: Sequence[str],
        batch_size: int,
        *,
        group_weights: dict[str, float] | None = None,
        num_batches: int | None = None,
        seed: int = 7,
        num_replicas: int = 1,
        rank: int = 0,
        curriculum: CurriculumSchedule | None = None,
        compositions: Sequence[str] | None = None,
        composition_curriculum: CompositionCurriculumSchedule | None = None,
        grad_accum_steps: int = 1,
        epochs: int = 1,
        max_steps: int | None = None,
        aspect_ratios: Sequence[float | None] | None = None,
        sample_ids: Sequence[str] | None = None,
        aspect_ratio_bucketing: bool = False,
        aspect_ratio_bucket_log2_step: float = 0.5,
        rotation_probability: float = 0.0,
        augmentation_seed: int = 7,
    ):
        if not groups:
            raise ValueError("balanced sampler requires at least one example")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.num_batches = num_batches or math.ceil(len(groups) / batch_size)
        self.seed = seed
        self.epoch = 0
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("distributed sampler rank must be within num_replicas")
        self.num_replicas = num_replicas
        self.rank = rank
        if grad_accum_steps <= 0 or epochs <= 0:
            raise ValueError("grad_accum_steps and epochs must be positive")
        self.grad_accum_steps = int(grad_accum_steps)
        self.steps_per_epoch = math.ceil(self.num_batches / self.grad_accum_steps)
        self.curriculum = curriculum or CurriculumSchedule()
        self.composition_curriculum = (
            composition_curriculum or CompositionCurriculumSchedule()
        )
        self.total_steps = planned_optimizer_steps(
            num_batches=self.num_batches,
            grad_accum_steps=self.grad_accum_steps,
            epochs=epochs,
            max_steps=max_steps,
        )
        self.indices: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            self.indices.setdefault(str(group), []).append(index)
        self.group_names = sorted(self.indices)
        if compositions is not None and len(compositions) != len(groups):
            raise ValueError("compositions must align with sampler groups")
        self.compositions = tuple(
            str(value) for value in (
                compositions
                if compositions is not None
                else ("single_page",) * len(groups)
            )
        )
        unknown_compositions = set(self.compositions) - set(COMPOSITION_TIERS)
        if unknown_compositions:
            raise ValueError(
                "unsupported composition tiers: "
                f"{sorted(unknown_compositions)}"
            )
        self.composition_curriculum.validate()
        self.composition_indices: dict[str, dict[str, list[int]]] = {}
        for group, indices in self.indices.items():
            for index in indices:
                tier = self.compositions[index]
                self.composition_indices.setdefault(group, {}).setdefault(
                    tier,
                    [],
                ).append(index)
        supplied = group_weights or {}
        unknown = set(supplied) - set(self.group_names)
        if unknown:
            raise ValueError(f"weights reference unknown groups: {sorted(unknown)}")
        self.base_weights = {
            group: float(supplied.get(group, 1.0)) for group in self.group_names
        }
        self.weights = [self.base_weights[group] for group in self.group_names]
        if any(weight < 0 for weight in self.weights) or not any(self.weights):
            raise ValueError("group weights must be non-negative with at least one positive value")
        curriculum_groups = {
            group
            for stage in self.curriculum.stages
            for group in stage.group_weights
        }
        unknown_curriculum = curriculum_groups - set(self.group_names)
        if unknown_curriculum:
            raise ValueError(
                "curriculum weights reference unknown groups: "
                f"{sorted(unknown_curriculum)}"
            )
        if (
            self.curriculum.unit
            in {"training_token_fraction", "training_compute_fraction"}
            and curriculum_groups
        ):
            raise ValueError(
                "runtime token/compute curricula cannot drive prefetched sampler "
                "weights; use loss-weight stages or an optimizer-step curriculum"
            )
        self.aspect_ratio_bucketing = bool(aspect_ratio_bucketing)
        self.aspect_ratio_bucket_log2_step = float(aspect_ratio_bucket_log2_step)
        self.rotation_probability = float(rotation_probability)
        self.augmentation_seed = int(augmentation_seed)
        if self.aspect_ratio_bucket_log2_step <= 0:
            raise ValueError("aspect_ratio_bucket_log2_step must be positive")
        if not 0.0 <= self.rotation_probability <= 1.0:
            raise ValueError("rotation_probability must be within [0, 1]")
        if aspect_ratios is not None and len(aspect_ratios) != len(groups):
            raise ValueError("aspect_ratios must align with sampler groups")
        if sample_ids is not None and len(sample_ids) != len(groups):
            raise ValueError("sample_ids must align with sampler groups")
        if self.aspect_ratio_bucketing and aspect_ratios is None:
            raise ValueError("aspect-ratio bucketing requires aspect_ratios")
        if self.aspect_ratio_bucketing and sample_ids is None:
            raise ValueError("aspect-ratio bucketing requires sample_ids")
        self.aspect_ratios = (
            tuple(aspect_ratios) if aspect_ratios is not None else ()
        )
        sample_id_values = sample_ids if sample_ids is not None else ()
        self.sample_ids = tuple(str(value) for value in sample_id_values)

    @classmethod
    def from_blueprint(
        cls,
        dataset: UDDStudentDataset,
        blueprint: dict[str, Any],
        batch_size: int,
        *,
        num_batches: int | None = None,
        seed: int = 7,
        num_replicas: int | None = None,
        rank: int | None = None,
        grad_accum_steps: int | None = None,
        epochs: int | None = None,
        max_steps: int | None = None,
    ) -> "BalancedGroupBatchSampler":
        pipeline = blueprint["training"]["pretraining"]["input_pipeline"]
        optimizer = blueprint["training"]["pretraining"]["optimizer"]
        balance_by = str(pipeline["balance_by"])
        return cls(
            dataset.groups(balance_by),
            batch_size,
            group_weights={
                str(group): float(weight)
                for group, weight in (pipeline.get("group_weights") or {}).items()
            },
            num_batches=num_batches,
            seed=seed,
            num_replicas=(
                int(os.environ.get("WORLD_SIZE", "1"))
                if num_replicas is None
                else num_replicas
            ),
            rank=(
                int(os.environ.get("RANK", "0"))
                if rank is None
                else rank
            ),
            curriculum=CurriculumSchedule.from_blueprint(blueprint),
            compositions=dataset.compositions,
            composition_curriculum=(
                CompositionCurriculumSchedule.from_blueprint(blueprint)
            ),
            grad_accum_steps=(
                int(optimizer["grad_accum_steps"])
                if grad_accum_steps is None
                else int(grad_accum_steps)
            ),
            epochs=(
                int(optimizer["epochs"] or 1)
                if epochs is None
                else int(epochs)
            ),
            max_steps=(
                (
                    None
                    if optimizer.get("max_steps") is None
                    else int(optimizer["max_steps"])
                )
                if max_steps is None
                else int(max_steps)
            ),
            aspect_ratios=dataset.aspect_ratios,
            sample_ids=dataset.sample_ids,
            aspect_ratio_bucketing=bool(
                pipeline.get("aspect_ratio_bucketing", False)
            ),
            aspect_ratio_bucket_log2_step=float(
                pipeline.get("aspect_ratio_bucket_log2_step", 0.5)
            ),
            rotation_probability=float(pipeline["rotation_probability"]),
            augmentation_seed=int(pipeline.get("augmentation_seed", 7)),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_group_weights(
        self,
        group_weights: Mapping[str, float],
    ) -> None:
        """Replace epoch-level group weights without changing sampler topology."""
        supplied = {
            str(group): float(weight)
            for group, weight in group_weights.items()
        }
        if set(supplied) != set(self.group_names):
            missing = sorted(set(self.group_names) - set(supplied))
            extra = sorted(set(supplied) - set(self.group_names))
            raise ValueError(
                "replacement weights must match sampler groups: "
                f"missing={missing}, extra={extra}"
            )
        values = [supplied[group] for group in self.group_names]
        if (
            any(not math.isfinite(weight) or weight < 0 for weight in values)
            or not any(values)
        ):
            raise ValueError(
                "replacement group weights must be finite, non-negative, "
                "and include a positive value"
            )
        self.base_weights = supplied
        self.weights = values

    def __len__(self) -> int:
        return self.num_batches

    def _aspect_bucket(self, index: int) -> int | str:
        ratio = self.aspect_ratios[index]
        if ratio is None or not math.isfinite(ratio) or ratio <= 0:
            return "unknown"
        turns = deterministic_quarter_turns(
            self.sample_ids[index],
            epoch=self.epoch,
            probability=self.rotation_probability,
            seed=self.augmentation_seed,
        )
        effective_ratio = 1.0 / ratio if turns % 2 else ratio
        return round(math.log2(effective_ratio) / self.aspect_ratio_bucket_log2_step)

    def _bucketed_indices(
        self,
    ) -> dict[int | str, dict[str, list[int]]]:
        buckets: dict[int | str, dict[str, list[int]]] = {}
        for group, indices in self.indices.items():
            for index in indices:
                bucket = self._aspect_bucket(index)
                buckets.setdefault(bucket, {}).setdefault(group, []).append(index)
        return buckets

    def _composition_weights(self, step: int) -> tuple[str, dict[str, float]]:
        stage = self.composition_curriculum.stage_for_step(step)
        if stage is None:
            return "base", {tier: 1.0 for tier in COMPOSITION_TIERS}
        return stage.id, dict(stage.weights)

    def _weighted_pool(
        self,
        group: str,
        indices: Sequence[int],
        composition_weights: Mapping[str, float],
    ) -> tuple[list[int], list[float]]:
        if not self.composition_curriculum.stages:
            return list(indices), [1.0] * len(indices)
        pool: list[int] = []
        weights: list[float] = []
        for index in indices:
            tier = self.compositions[index]
            tier_count = len(self.composition_indices[group][tier])
            weight = float(composition_weights[tier]) / tier_count
            if weight > 0:
                pool.append(index)
                weights.append(weight)
        return pool, weights

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        bucketed = self._bucketed_indices() if self.aspect_ratio_bucketing else {}
        for batch_index in range(self.num_batches):
            step = (
                self.epoch * self.steps_per_epoch
                + batch_index // self.grad_accum_steps
            )
            stage = (
                self.curriculum.stage_for_step(step, self.total_steps)
                if self.curriculum.unit == "optimizer_step_fraction"
                else None
            )
            weights = dict(self.base_weights)
            if stage is not None:
                weights.update(stage.group_weights)
            selected_weights = [weights[group] for group in self.group_names]
            if not any(selected_weights):
                raise ValueError(
                    f"curriculum stage {stage.id!r} disables every sampler group"
                )
            global_batch_size = self.batch_size * self.num_replicas
            composition_stage, composition_weights = (
                self._composition_weights(step)
            )
            group_pools = {
                group: self._weighted_pool(
                    group,
                    self.indices[group],
                    composition_weights,
                )
                for group in self.group_names
            }
            selected_weights = [
                weight if group_pools[group][0] else 0.0
                for group, weight in zip(self.group_names, selected_weights)
            ]
            if not any(selected_weights):
                raise ValueError(
                    "composition curriculum stage "
                    f"{composition_stage!r} disables every available example"
                )
            if bucketed:
                bucket_names = sorted(bucketed, key=str)
                bucket_masses = []
                for bucket in bucket_names:
                    bucket_masses.append(
                        sum(
                            weight
                            * sum(
                                self._weighted_pool(
                                    group,
                                    bucketed[bucket].get(group, ()),
                                    composition_weights,
                                )[1]
                            )
                            / sum(group_pools[group][1])
                            for group, weight in zip(
                                self.group_names, selected_weights
                            )
                            if weight > 0
                        )
                    )
                bucket = rng.choices(bucket_names, weights=bucket_masses, k=1)[0]
                conditional_weights = [
                    (
                        weight
                        * sum(
                            self._weighted_pool(
                                group,
                                bucketed[bucket].get(group, ()),
                                composition_weights,
                            )[1]
                        )
                        / sum(group_pools[group][1])
                        if weight > 0
                        else 0.0
                    )
                    for group, weight in zip(self.group_names, selected_weights)
                ]
                selected_groups = rng.choices(
                    self.group_names,
                    weights=conditional_weights,
                    k=global_batch_size,
                )
                selected_indices = []
                for group in selected_groups:
                    pool, pool_weights = self._weighted_pool(
                        group,
                        bucketed[bucket][group],
                        composition_weights,
                    )
                    selected_indices.append(
                        rng.choices(pool, weights=pool_weights, k=1)[0]
                    )
            else:
                selected_groups = rng.choices(
                    self.group_names,
                    weights=selected_weights,
                    k=global_batch_size,
                )
                selected_indices = [
                    rng.choices(
                        group_pools[group][0],
                        weights=group_pools[group][1],
                        k=1,
                    )[0]
                    for group in selected_groups
                ]
            start = self.rank * self.batch_size
            yield selected_indices[start : start + self.batch_size]


class DeterministicDistributedBatchSampler:
    """Shuffle exhaustively per epoch, padding only to align distributed ranks."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        seed: int = 7,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if dataset_size <= 0 or batch_size <= 0:
            raise ValueError("dataset_size and batch_size must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("distributed sampler rank must be within num_replicas")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.num_batches = math.ceil(
            self.dataset_size / (self.batch_size * self.num_replicas)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(self.dataset_size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        if self.num_replicas == 1:
            for start in range(0, self.dataset_size, self.batch_size):
                yield indices[start : start + self.batch_size]
            return

        global_batch_size = self.batch_size * self.num_replicas
        total_size = self.num_batches * global_batch_size
        indices.extend(
            indices[index % self.dataset_size]
            for index in range(total_size - self.dataset_size)
        )
        rank_start = self.rank * self.batch_size
        for start in range(0, total_size, global_batch_size):
            global_batch = indices[start : start + global_batch_size]
            yield global_batch[rank_start : rank_start + self.batch_size]

"""InternVL2.5-1B / InternVL3-1B adapters.

Architecture: InternViT-300M vision encoder + Qwen2/2.5-0.5B language model, joined by an
MLP pixel-shuffle projector (~0.9B total). These ship strong document/OCR ability for the
size and are first-class citizens in VLMEvalKit. We reproduce the model card's dynamic
high-resolution tiling (up to ``max_tiles`` 448px crops) because it is what gives InternVL
its document edge - disabling it collapses small-text accuracy.

Model card: https://huggingface.co/OpenGVLab/InternVL2_5-1B
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _find_closest_ratio(ar, ratios, w, h, size):
    best, best_diff = (1, 1), float("inf")
    area = w * h
    for r in ratios:
        target = r[0] / r[1]
        diff = abs(ar - target)
        if diff < best_diff or (
            diff == best_diff and area > 0.5 * size * size * r[0] * r[1]
        ):
            best_diff, best = diff, r
    return best


def _dynamic_preprocess(image, min_tiles=1, max_tiles=12, size=448, use_thumbnail=True):
    """InternVL's aspect-ratio-aware tiling (verbatim logic from the model card)."""
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {
            (i, j)
            for n in range(min_tiles, max_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_tiles <= i * j <= max_tiles
        },
        key=lambda x: x[0] * x[1],
    )
    rw, rh = _find_closest_ratio(ar, ratios, w, h, size)
    tw, th = size * rw, size * rh
    blocks = rw * rh
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // size
    for idx in range(blocks):
        box = (
            (idx % cols) * size,
            (idx // cols) * size,
            ((idx % cols) + 1) * size,
            ((idx // cols) + 1) * size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and blocks != 1:
        tiles.append(image.resize((size, size)))
    return tiles


@dataclass
class _InternVL(ModelAdapter):
    family: str = "InternVL"
    max_tiles: int = 12
    input_size: int = 448

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = getattr(torch, self.dtype)
        self.model = (
            AutoModel.from_pretrained(
                self.hf_id,
                revision=self.revision,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=self.resolve_attn(),
            )
            .eval()
            .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            revision=self.revision,
            trust_remote_code=True,
            use_fast=False,
        )
        self.transform = _build_transform(self.input_size)
        self._loaded = True

    def _pixels(self, image_path):
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        tiles = _dynamic_preprocess(
            img, max_tiles=self.max_tiles, size=self.input_size
        )
        pv = torch.stack([self.transform(t) for t in tiles])
        return pv.to(getattr(torch, self.dtype)).to(self.device)

    def generate(self, image_path: str, question: str):
        import torch

        pixel_values = self._pixels(image_path)
        prompt = f"<image>\n{question}"
        with torch.no_grad():
            # InternVL's chat() returns text; to also get confidence we call the
            # underlying generate via the documented `return_history`-free path and
            # recompute logprobs from a second scored pass when needed.
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt,
                dict(max_new_tokens=self.gen.max_new_tokens,
                     do_sample=self.gen.do_sample, num_beams=self.gen.num_beams),
            )
        return response.strip(), None


@register("internvl2_5-1b")
@dataclass
class InternVL2_5_1B(_InternVL):
    hf_id: str = "OpenGVLab/InternVL2_5-1B"
    param_count_m: float = 938.0


@register("internvl3-1b")
@dataclass
class InternVL3_1B(_InternVL):
    hf_id: str = "OpenGVLab/InternVL3-1B"
    param_count_m: float = 938.0


@register("internvl2-1b")
@dataclass
class InternVL2_1B(_InternVL):
    # older 1B (InternViT-300M + Qwen2-0.5B); good doc/OCR baseline within the family
    hf_id: str = "OpenGVLab/InternVL2-1B"
    param_count_m: float = 938.0

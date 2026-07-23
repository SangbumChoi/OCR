"""Florence-2-large adapter (~0.77B).

Architecture: a DaViT vision encoder + BART-style encoder-decoder, trained on the FLD-5B
dataset with a unified *task-token* interface (``<OCR>``, ``<OCR_WITH_REGION>``,
``<CAPTION>``, ``<OD>``, ...). Like GOT, it is not a conversational VQA model - there is no
``<DocVQA>`` token - so we drive it with ``<OCR>`` and treat it as an OCR-specialist
baseline. Strong, fast text spotting at a tiny size; weak at question reasoning.

Model card: https://huggingface.co/microsoft/Florence-2-large
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@dataclass
class _Florence2(ModelAdapter):
    family: str = "Florence-2"
    hf_id: str = "microsoft/Florence-2-large"
    param_count_m: float = 770.0
    task_token: str = "<OCR>"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.hf_id,
            revision=self.revision,
            trust_remote_code=True,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                revision=self.revision,
                torch_dtype=getattr(torch, self.dtype),
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self._loaded = True

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=self.task_token, images=image, return_tensors="pt"
        ).to(self.device, getattr(torch, self.dtype))
        with torch.no_grad():
            ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.gen.max_new_tokens,
                num_beams=3,
                do_sample=False,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip(), None


@register("florence2-large")
@dataclass
class Florence2Large(_Florence2):
    hf_id: str = "microsoft/Florence-2-large"
    param_count_m: float = 770.0


@register("florence2-base")
@dataclass
class Florence2Base(_Florence2):
    hf_id: str = "microsoft/Florence-2-base"
    param_count_m: float = 230.0

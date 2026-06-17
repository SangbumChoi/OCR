"""SmolVLM-256M / SmolVLM-500M adapters.

Architecture: SigLIP vision encoder + SmolLM2 language model in the Idefics3 framework,
with aggressive pixel-shuffle token compression for edge/on-device use. These are the
smallest serious VLMs available and anchor the "edge deployment" end of the comparison.

Model card: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
Uses the standard HF ``AutoModelForVision2Seq`` + chat template, so confidence (token
logprobs) is available for calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@dataclass
class _SmolVLM(ModelAdapter):
    family: str = "SmolVLM"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                self.hf_id, torch_dtype=getattr(torch, self.dtype)
            )
            .eval()
            .to(self.device)
        )
        self._loaded = True

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(
            self.device
        )
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=self.gen.do_sample,
                num_beams=self.gen.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        text = self.processor.batch_decode(
            out.sequences[:, input_len:], skip_special_tokens=True
        )[0]
        conf = self._confidence_from_scores(out.scores, out.sequences, input_len)
        return text.strip(), conf


@register("smolvlm-256m")
@dataclass
class SmolVLM256M(_SmolVLM):
    hf_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    param_count_m: float = 256.0


@register("smolvlm-500m")
@dataclass
class SmolVLM500M(_SmolVLM):
    hf_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    param_count_m: float = 500.0

"""LLaVA-OneVision-0.5B adapter.

Architecture: SigLIP-SO400M vision encoder + Qwen2-0.5B LM with AnyRes high-resolution
tiling (~0.9B total). A strong general single-image/multi-image/video VLM; included as the
"general-purpose small VLM" baseline against the document specialists.

Model card: https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@register("llava-ov-0.5b")
@dataclass
class LlavaOneVision05B(ModelAdapter):
    family: str = "LLaVA-OneVision"
    hf_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    param_count_m: float = 894.0

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.hf_id,
            revision=self.revision,
        )
        self.model = (
            LlavaOnevisionForConditionalGeneration.from_pretrained(
                self.hf_id,
                revision=self.revision,
                torch_dtype=getattr(torch, self.dtype),
                low_cpu_mem_usage=True,
                attn_implementation=self.resolve_attn(),
            )
            .eval()
            .to(self.device)
        )
        self._loaded = True

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, getattr(torch, self.dtype)
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

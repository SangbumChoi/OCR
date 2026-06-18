"""Ovis2-1B adapter.

Ovis (AIDC-AI) structurally *aligns* visual and textual embeddings via a learned visual
embedding table. Ovis2-1B = AIMv2-large vision encoder + Qwen2.5-0.5B LM (~1.0B total), with
notably strong OCR and structured table/chart extraction for its size. It is the only Ovis
variant at/below ~1B (Ovis2.5's smallest is 2B).

Ovis exposes a custom interface (``preprocess_inputs`` + separate text/visual tokenizers), so
this adapter follows the model card rather than the plain HF ``generate`` path.
Model card: https://huggingface.co/AIDC-AI/Ovis2-1B
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@register("ovis2-1b")
@dataclass
class Ovis2_1B(ModelAdapter):
    family: str = "Ovis2"
    hf_id: str = "AIDC-AI/Ovis2-1B"
    param_count_m: float = 1000.0
    max_partition: int = 9

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.hf_id,
                torch_dtype=getattr(torch, self.dtype),
                multimodal_max_length=8192,
                trust_remote_code=True,
                # default is flash_attention_2 (CUDA-only); resolve to sdpa/eager unless opted in
                attn_implementation=self.resolve_attn(),
            )
            .eval()
            .to(self.device)
        )
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self._loaded = True

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        query = f"<image>\n{question}"
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, [image], max_partition=self.max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=self.device, dtype=self.visual_tokenizer.dtype
            )
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                pixel_values=[pixel_values],
                attention_mask=attention_mask,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=self.gen.do_sample,
                num_beams=self.gen.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=self.text_tokenizer.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
            )
        text = self.text_tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        conf = self._confidence_from_scores(out.scores, out.sequences, input_ids.shape[1])
        return text.strip(), conf

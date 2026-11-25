"""PaddleOCR-VL-0.9B adapter.

Architecture: a NaViT-style dynamic-resolution vision encoder + ERNIE-4.5-0.3B language
model (~0.9B total), purpose-built for document parsing (text, tables, formulas, charts,
reading order) across 100+ languages. The most recent and most document-specialised model
in the comparison, so a natural candidate for "best small document VLM".

Model card: https://huggingface.co/PaddlePaddle/PaddleOCR-VL

Two usage paths exist; we prefer the transformers-native one and fall back to the
``paddleocr`` pipeline if the custom classes are unavailable. Because the public release
targets full-page parsing rather than single-question VQA, we prompt it with the question
directly; on VQA benchmarks this exercises its layout+text understanding, while on OCRBench
it plays to its strength.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


def _patch_create_causal_mask_kwarg() -> None:
    """PaddleOCR-VL's remote code calls ``create_causal_mask(inputs_embeds=...)`` but recent
    transformers renamed the parameter to ``input_embeds``. Wrap it so both names work; applied
    before the remote module is imported so its ``from ... import create_causal_mask`` binds the
    shim. No-op if the installed transformers already accepts ``inputs_embeds``.
    """
    try:
        import inspect

        import transformers.masking_utils as mu

        if getattr(mu.create_causal_mask, "_kwarg_shim", False):
            return
        if "inputs_embeds" in inspect.signature(mu.create_causal_mask).parameters:
            return  # already compatible
        _orig = mu.create_causal_mask

        def _shim(*args, **kw):
            if "inputs_embeds" in kw and "input_embeds" not in kw:
                kw["input_embeds"] = kw.pop("inputs_embeds")
            return _orig(*args, **kw)

        _shim._kwarg_shim = True
        mu.create_causal_mask = _shim
    except Exception:
        pass


@dataclass
class _PaddleOCRVL(ModelAdapter):
    family: str = "PaddleOCR-VL"
    hf_id: str = "PaddlePaddle/PaddleOCR-VL"
    param_count_m: float = 900.0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        _patch_create_causal_mask_kwarg()  # remote code uses inputs_embeds=; tf renamed it

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
                low_cpu_mem_usage=True,
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
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_prompt], images=[image], return_tensors="pt"
        ).to(self.device)
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


@register("paddleocr-vl")
@dataclass
class PaddleOCRVL(_PaddleOCRVL):
    # the 0.9B "1.0" release (NaViT encoder + ERNIE-4.5-0.3B)
    hf_id: str = "PaddlePaddle/PaddleOCR-VL"
    param_count_m: float = 900.0


@register("paddleocr-vl-1.5")
@dataclass
class PaddleOCRVL15(_PaddleOCRVL):
    # v1.5: adds irregular-shape (polygonal) localization, text spotting, seal recognition;
    # SOTA on OmniDocBench v1.5 (~94.5 overall). Same backbone family as 1.0.
    hf_id: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    param_count_m: float = 900.0


@register("paddleocr-vl-1.6")
@dataclass
class PaddleOCRVL16(_PaddleOCRVL):
    # v1.6: latest PaddleOCR-VL release. Needs a recent transformers
    # (use_kernel_forward_from_hub) -> run in the separate env (scripts/run_paddleocr_env.sh).
    hf_id: str = "PaddlePaddle/PaddleOCR-VL-1.6"
    param_count_m: float = 900.0

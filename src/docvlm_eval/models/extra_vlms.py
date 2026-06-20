"""Adapters for newer sub-~1.6B document/vision-language models (2025-2026 releases).

These all use the **modern unified HF chat-template** path
(``processor.apply_chat_template(..., tokenize=True, return_dict=True)`` with the image embedded
in the message content), so one base class — :class:`_HFChatVLM` — covers them. Confidence (token
logprobs) is exposed via ``output_scores`` for calibration, exactly like the SmolVLM adapter.

Registered here:
  * ``minicpm-v-4_6``   openbmb/MiniCPM-V-4.6            (~1.3B)
  * ``lfm2_5-vl-1.6b``  LiquidAI/LFM2.5-VL-1.6B          (~1.6B)
  * ``qwen3_5-0.8b``    Qwen/Qwen3.5-0.8B  (VL variant)  (~0.87B, sub-1B)
  * ``lightonocr-1b``   lightonai/LightOnOCR-1B-1025     (~1.16B, OCR specialist)

Ovis2.5 (custom ``preprocess_inputs`` interface) lives in ``ovis.py`` next to Ovis2.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@dataclass
class _HFChatVLM(ModelAdapter):
    """Generic adapter for VLMs that follow the standard AutoProcessor + chat-template API.

    Subclasses set ``hf_id`` / ``param_count_m`` / ``family``. ``trust_remote`` is on by default
    because these are bleeding-edge checkpoints whose code may not yet be in a stable transformers
    release (harmless when the model is already built in)."""

    family: str = "VLM"
    trust_remote: bool = True

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as _AutoVLM
        except ImportError:  # older transformers
            from transformers import AutoModelForVision2Seq as _AutoVLM

        self.processor = AutoProcessor.from_pretrained(
            self.hf_id, trust_remote_code=self.trust_remote
        )
        self.model = (
            _AutoVLM.from_pretrained(
                self.hf_id, torch_dtype=getattr(torch, self.dtype),
                trust_remote_code=self.trust_remote, attn_implementation=self.resolve_attn(),
            )
            .eval()
            .to(self.device)
        )
        self._loaded = True

    def _build_inputs(self, image, question: str):
        """Construct model inputs, preferring the unified tokenized chat template and falling back
        to the two-step (template->string, then processor(text, images)) path for older processors."""
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }]
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            return inputs.to(self.device)
        except Exception:
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
                add_generation_prompt=True, tokenize=False,
            )
            return self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._build_inputs(image, question)
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


@register("minicpm-v-4_6")
@dataclass
class MiniCPMV4_6(_HFChatVLM):
    # OpenBMB MiniCPM-V 4.6: SigLIP-style encoder + MiniCPM4 LM with high-res slicing. Over the
    # <1B budget (kept as a stronger reference point); standard AutoModelForImageTextToText API.
    family: str = "MiniCPM-V"
    hf_id: str = "openbmb/MiniCPM-V-4.6"
    param_count_m: float = 1300.0


@register("lfm2_5-vl-1.6b")
@dataclass
class LFM2_5_VL_1_6B(_HFChatVLM):
    # LiquidAI LFM2.5-VL: LFM2 hybrid-conv backbone + SigLIP2 encoder, edge-oriented.
    family: str = "LFM2-VL"
    hf_id: str = "LiquidAI/LFM2.5-VL-1.6B"
    param_count_m: float = 1597.0


@register("qwen3_5-0.8b")
@dataclass
class Qwen3_5_0_8B(_HFChatVLM):
    # Qwen3.5-0.8B is a vision-language variant (config carries vision_config) — the smallest
    # Qwen3.5 multimodal checkpoint, comfortably sub-1B.
    family: str = "Qwen3.5-VL"
    hf_id: str = "Qwen/Qwen3.5-0.8B"
    param_count_m: float = 873.0


@register("lightonocr-1b")
@dataclass
class LightOnOCR1B(_HFChatVLM):
    # LightOn LightOnOCR-1B: Mistral3/Pixtral-style OCR specialist (vision encoder + small LM),
    # purpose-built for full-page transcription. Tests an OCR-first design at ~1B.
    family: str = "LightOnOCR"
    hf_id: str = "lightonai/LightOnOCR-1B-1025"
    param_count_m: float = 1161.0

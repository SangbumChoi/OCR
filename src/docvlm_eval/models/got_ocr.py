"""GOT-OCR2.0 adapter (~580M).

Architecture: a ~80M ViT-based vision encoder + ~500M Qwen-0.5B decoder, trained
end-to-end as a *pure OCR transcription* model ("General OCR Theory"). It does NOT take a
free-form question - it transcribes the page (plain text or formatted/markdown). We
include it as an **OCR-specialist baseline**: it should dominate OCRBench-style text
recognition yet underperform on DocVQA/InfoVQA, because answering a question requires
reasoning the model was never trained to do. That contrast is itself a finding.

For VQA benchmarks we feed the question as context but the model effectively returns the
transcription; scoring then measures whether the answer string happens to appear in the
transcript (a generous lower bound for an OCR-only model).

Model card: https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf (transformers-native class).
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import ModelAdapter
from .registry import register


@register("got-ocr2")
@dataclass
class GotOCR2(ModelAdapter):
    family: str = "GOT-OCR2.0"
    hf_id: str = "stepfun-ai/GOT-OCR-2.0-hf"
    param_count_m: float = 580.0
    ocr_format: bool = False  # True -> request markdown/formatted output

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.hf_id, torch_dtype=getattr(torch, self.dtype), low_cpu_mem_usage=True
            )
            .eval()
            .to(self.device)
        )
        self._loaded = True

    def generate(self, image_path: str, question: str):
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        # GOT ignores the question; it transcribes. `format=True` -> structured output.
        inputs = self.processor(
            image, return_tensors="pt", format=self.ocr_format
        ).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=self.gen.max_new_tokens,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
            )
        text = self.processor.decode(
            ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # No reliable per-answer confidence for a transcription task.
        return text.strip(), None

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class VisionTextCollator:
    """
    Convert (image, text) -> model inputs.

    Notes:
    - Many VLM models/processors exist, so by default we call processor(images=..., text=...).
    - For models that need a prompt, prefix the label via `prompt` to enable supervised training.
    """

    processor: Any
    prompt: str = ""
    max_label_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [f["image"] for f in features]
        texts = [f["text"] for f in features]
        if self.prompt:
            texts = [self.prompt + t for t in texts]

        batch = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_label_length is not None,
            max_length=self.max_label_length,
        )

        # some processors don't create labels automatically, so set labels from input_ids
        if "labels" not in batch:
            if "input_ids" not in batch:
                raise ValueError("processor output has no input_ids/labels. Check processor compatibility.")
            labels = batch["input_ids"].clone()
            pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            batch["labels"] = labels
            # remove input_ids so it isn't used as decoder input (needed for some models)
            # but some models need input_ids, so keep it
        else:
            labels = batch["labels"]
            if isinstance(labels, torch.Tensor):
                # -100 masking fix (in case padding comes in as 0)
                pass

        return batch



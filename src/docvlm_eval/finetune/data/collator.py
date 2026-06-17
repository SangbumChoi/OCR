from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass
class VisionTextCollator:
    """
    (image, text) -> model inputs 변환.

    주의:
    - 다양한 VLM 모델/프로세서가 존재하므로, 기본적으로 processor(images=..., text=...) 형태를 사용합니다.
    - prompt가 필요한 모델은 `prompt`로 라벨 앞에 prefix를 붙여 supervised 학습이 가능하도록 합니다.
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

        # 일부 processor는 labels를 자동으로 만들지 않아서, input_ids를 labels로 세팅
        if "labels" not in batch:
            if "input_ids" not in batch:
                raise ValueError("processor output에 input_ids/labels가 없습니다. processor 호환 여부를 확인하세요.")
            labels = batch["input_ids"].clone()
            pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100
            batch["labels"] = labels
            # decoder input으로 쓰이지 않도록 input_ids 제거(모델에 따라 필요)
            # 단, 어떤 모델은 input_ids가 필요할 수 있어 그대로 둡니다.
        else:
            labels = batch["labels"]
            if isinstance(labels, torch.Tensor):
                # -100 마스킹 보정(패딩이 0으로 들어오는 경우 대비)
                pass

        return batch



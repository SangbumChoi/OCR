from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class OcrExample:
    image_path: str
    text: str
    id: Optional[str] = None


class JsonlOcrDataset(Dataset):
    """
    JSONL 기반 OCR 데이터셋.

    각 라인 예시:
      {"image_path": "path/to/img.png", "text": "GT text", "id": "optional"}
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        *,
        base_dir: str | Path | None = None,
        image_mode: str = "RGB",
        max_samples: int | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.base_dir = Path(base_dir) if base_dir is not None else self.jsonl_path.parent
        self.image_mode = image_mode
        self.samples: List[OcrExample] = self._load(max_samples=max_samples)

    def _load(self, *, max_samples: int | None) -> List[OcrExample]:
        out: List[OcrExample] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                image_path = obj["image_path"]
                text = obj["text"]
                ex_id = obj.get("id")
                out.append(OcrExample(image_path=image_path, text=text, id=ex_id))
                if max_samples is not None and len(out) >= max_samples:
                    break
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.samples[idx]
        img_path = self._resolve_image_path(ex.image_path)
        image = Image.open(img_path).convert(self.image_mode)
        return {
            "image": image,
            "text": ex.text,
            "id": ex.id if ex.id is not None else str(idx),
            "image_path": str(img_path),
        }



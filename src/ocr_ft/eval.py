from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .metrics import compute_ocr_metrics
from .modeling import infer_one_deepseek_ocr


@dataclass
class EvalConfig:
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown. "
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    test_compress: bool = False


def evaluate_dataset_with_infer(
    *,
    model: Any,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    cfg: EvalConfig,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    preds: List[str] = []
    refs: List[str] = []
    rows: List[Dict[str, Any]] = []

    it = items[: max_samples if max_samples is not None else len(items)]
    for ex in tqdm(it, desc="eval"):
        pred = infer_one_deepseek_ocr(
            model=model,
            tokenizer=tokenizer,
            image_file=ex["image_path"],
            prompt=cfg.prompt,
            output_path=" ",
            base_size=cfg.base_size,
            image_size=cfg.image_size,
            crop_mode=cfg.crop_mode,
            test_compress=cfg.test_compress,
            save_results=False,
        )
        ref = ex["text"]
        preds.append(pred)
        refs.append(ref)
        rows.append(
            {
                "id": ex.get("id"),
                "image_path": ex["image_path"],
                "ref": ref,
                "pred": pred,
            }
        )

    metrics = compute_ocr_metrics(preds, refs)
    return {
        "metrics": metrics,
        "num_samples": len(it),
        "examples": rows,
    }


def save_eval_report(report: Dict[str, Any], report_path: str | Path) -> None:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)



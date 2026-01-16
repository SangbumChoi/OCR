from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_ft.data import JsonlOcrDataset
from ocr_ft.eval import EvalConfig, evaluate_dataset_with_infer, save_eval_report
from ocr_ft.modeling import ModelLoadConfig, load_deepseek_ocr_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare vanilla vs finetuned DeepSeek-OCR on same eval set")
    p.add_argument("--base_model_id", type=str, required=True)
    p.add_argument("--finetuned_model_id", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--report_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)

    p.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
    p.add_argument("--base_size", type=int, default=1024)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--crop_mode", type=int, default=1)
    p.add_argument("--test_compress", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ds = JsonlOcrDataset(args.val_jsonl, max_samples=args.max_samples)
    items = []
    for i in range(len(ds)):
        ex = ds[i]
        items.append({"id": ex["id"], "image_path": ex["image_path"], "text": ex["text"]})

    cfg = EvalConfig(
        prompt=args.prompt,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=bool(args.crop_mode),
        test_compress=bool(args.test_compress),
    )

    base_model, base_tok = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.base_model_id,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            device="cuda",
        )
    )
    ft_model, ft_tok = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.finetuned_model_id,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            device="cuda",
        )
    )

    base_report = evaluate_dataset_with_infer(model=base_model, tokenizer=base_tok, items=items, cfg=cfg, max_samples=args.max_samples)
    ft_report = evaluate_dataset_with_infer(model=ft_model, tokenizer=ft_tok, items=items, cfg=cfg, max_samples=args.max_samples)

    out = {
        "base": {"model_id": args.base_model_id, **base_report},
        "finetuned": {"model_id": args.finetuned_model_id, **ft_report},
        "delta": {
            "cer": ft_report["metrics"]["cer"] - base_report["metrics"]["cer"],
            "wer": ft_report["metrics"]["wer"] - base_report["metrics"]["wer"],
        },
    }

    p = Path(args.report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



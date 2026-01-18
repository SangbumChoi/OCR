from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ocr_ft.modeling import ModelLoadConfig, infer_one_deepseek_ocr, load_deepseek_ocr_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepSeek-OCR HuggingFace inference example (model.infer)")
    p.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--image_file", type=str, required=True)
    p.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)

    p.add_argument("--base_size", type=int, default=1024)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--crop_mode", type=int, default=1)
    p.add_argument("--test_compress", type=int, default=0)
    p.add_argument("--save_results", type=int, default=0)
    p.add_argument("--output_path", type=str, default=" ")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.save_results:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    text = infer_one_deepseek_ocr(
        model=model,
        tokenizer=tokenizer,
        image_file=args.image_file,
        prompt=args.prompt,
        output_path=args.output_path,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=bool(args.crop_mode),
        test_compress=bool(args.test_compress),
        save_results=bool(args.save_results),
    )
    print(text)


if __name__ == "__main__":
    main()


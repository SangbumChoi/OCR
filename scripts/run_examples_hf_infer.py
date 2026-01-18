from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from ocr_ft.modeling import ModelLoadConfig, infer_one_deepseek_ocr, load_deepseek_ocr_model_and_tokenizer


def _iter_images(examples_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    return sorted([p for p in examples_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run DeepSeek-OCR HF inference on examples/ and save to results/")
    p.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--examples_dir", type=str, default="examples")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)

    p.add_argument("--base_size", type=int, default=1024)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--crop_mode", type=int, default=1)
    p.add_argument("--test_compress", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()

    examples_dir = Path(args.examples_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            device=device,
        )
    )

    images = _iter_images(examples_dir)
    if len(images) == 0:
        raise SystemExit(f"이미지를 찾지 못했습니다: {examples_dir}")

    rows: List[Dict[str, str]] = []
    for img_path in tqdm(images, desc="examples"):
        text = infer_one_deepseek_ocr(
            model=model,
            tokenizer=tokenizer,
            image_file=str(img_path),
            prompt=args.prompt,
            output_path=" ",
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=bool(args.crop_mode),
            test_compress=bool(args.test_compress),
            save_results=False,
        )

        out_txt = results_dir / f"{img_path.stem}.txt"
        out_txt.write_text(text, encoding="utf-8")
        rows.append({"image": str(img_path), "result_txt": str(out_txt), "text": text})

    out_jsonl = results_dir / "examples_hf_outputs.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} outputs to: {results_dir}")
    print(f"- per-image txt: {results_dir}/*.txt")
    print(f"- jsonl summary: {out_jsonl}")


if __name__ == "__main__":
    main()


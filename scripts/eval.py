from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

from docvlm_eval.finetune.data import JsonlOcrDataset
from docvlm_eval.finetune.eval import EvalConfig, evaluate_dataset_with_infer, save_eval_report
from docvlm_eval.finetune.modeling import ModelLoadConfig, load_deepseek_ocr_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate DeepSeek-OCR using model.infer + CER/WER")
    p.add_argument("--model_id", type=str, required=True)
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
    model, tokenizer = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            device="cuda",
        )
    )

    ds = JsonlOcrDataset(args.val_jsonl, max_samples=args.max_samples)
    items = []
    for i in range(len(ds)):
        ex = ds[i]
        items.append({"id": ex["id"], "image_path": ex["image_path"], "text": ex["text"]})

    report = evaluate_dataset_with_infer(
        model=model,
        tokenizer=tokenizer,
        items=items,
        cfg=EvalConfig(
            prompt=args.prompt,
            base_size=args.base_size,
            image_size=args.image_size,
            crop_mode=bool(args.crop_mode),
            test_compress=bool(args.test_compress),
        ),
        max_samples=args.max_samples,
    )
    save_eval_report(report, args.report_path)


if __name__ == "__main__":
    main()



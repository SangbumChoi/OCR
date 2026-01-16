from __future__ import annotations

import argparse

from ocr_ft.train import TrainConfig, train_lora


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepSeek-OCR LoRA finetune (pytorch-native via accelerate)")
    p.add_argument("--model_id", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, default=None)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)

    p.add_argument("--prompt", type=str, default="<image>\n<|grounding|>Convert the document to markdown. ")
    p.add_argument("--base_size", type=int, default=1024)
    p.add_argument("--image_size", type=int, default=640)
    p.add_argument("--crop_mode", type=int, default=1)
    p.add_argument("--test_compress", type=int, default=0)

    p.add_argument("--use_lora", type=int, default=1)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    p.add_argument("--use_wandb", type=int, default=0)
    p.add_argument("--wandb_project", type=str, default="deepseek-ocr-finetune")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every_steps", type=int, default=200)
    p.add_argument("--max_samples_train", type=int, default=None)
    p.add_argument("--max_samples_val", type=int, default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = TrainConfig(
        model_id=args.model_id,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_train_steps=args.max_train_steps,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        prompt=args.prompt,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=bool(args.crop_mode),
        test_compress=bool(args.test_compress),
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple([s.strip() for s in args.lora_target_modules.split(",") if s.strip()]),
        use_wandb=bool(args.use_wandb),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_every=args.log_every,
        eval_every_steps=args.eval_every_steps,
        max_samples_train=args.max_samples_train,
        max_samples_val=args.max_samples_val,
    )
    train_lora(cfg)


if __name__ == "__main__":
    main()



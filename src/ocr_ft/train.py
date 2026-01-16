from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import JsonlOcrDataset, VisionTextCollator
from .eval import EvalConfig, evaluate_dataset_with_infer, save_eval_report
from .modeling import ModelLoadConfig, load_deepseek_ocr_model_and_tokenizer, try_load_deepseek_ocr_processor


@dataclass
class TrainConfig:
    model_id: str
    train_jsonl: str
    val_jsonl: Optional[str] = None
    output_dir: str = "outputs/exp"

    # optimization
    seed: int = 42
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 1
    train_batch_size: int = 1
    grad_accum_steps: int = 8
    max_train_steps: Optional[int] = None
    max_samples_train: Optional[int] = None
    max_samples_val: Optional[int] = None

    # model load
    dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None

    # DeepSeek-OCR prompt / infer options (eval용)
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown. "
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    test_compress: bool = False

    # LoRA
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # logging
    use_wandb: bool = False
    wandb_project: str = "deepseek-ocr-finetune"
    wandb_run_name: Optional[str] = None
    log_every: int = 10
    eval_every_steps: int = 200


def _maybe_init_wandb(cfg: TrainConfig) -> Any:
    if not cfg.use_wandb:
        return None
    import wandb

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config={
            "model_id": cfg.model_id,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "num_train_epochs": cfg.num_train_epochs,
            "train_batch_size": cfg.train_batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "max_train_steps": cfg.max_train_steps,
            "dtype": cfg.dtype,
            "attn_implementation": cfg.attn_implementation,
            "use_lora": cfg.use_lora,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "lora_target_modules": list(cfg.lora_target_modules),
            "prompt": cfg.prompt,
            "base_size": cfg.base_size,
            "image_size": cfg.image_size,
            "crop_mode": cfg.crop_mode,
            "test_compress": cfg.test_compress,
        },
    )
    return wandb


def _load_jsonl_as_items(jsonl_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = JsonlOcrDataset(jsonl_path, max_samples=max_samples)
    # eval은 infer가 image_file path를 받으므로, path를 그대로 제공
    items = []
    for i in range(len(ds)):
        ex = ds[i]
        items.append(
            {
                "id": ex["id"],
                "image_path": ex["image_path"],
                "text": ex["text"],
            }
        )
    return items


def _save_train_config(cfg: TrainConfig) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    with Path(cfg.output_dir, "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2, default=str)


def _apply_lora(model: Any, cfg: TrainConfig) -> Any:
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=list(cfg.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def train_lora(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    _save_train_config(cfg)

    accelerator = Accelerator(gradient_accumulation_steps=cfg.grad_accum_steps)
    wandb = _maybe_init_wandb(cfg) if accelerator.is_main_process else None

    model, tokenizer = load_deepseek_ocr_model_and_tokenizer(
        ModelLoadConfig(
            model_id=cfg.model_id,
            dtype=cfg.dtype,
            attn_implementation=cfg.attn_implementation,
            device=str(accelerator.device),
        )
    )

    if cfg.use_lora:
        model = _apply_lora(model, cfg)

    # 학습 입력 구성: (1) model.processor가 있으면 사용, (2) AutoProcessor가 있으면 사용, (3) 아니면 명확한 에러
    processor = getattr(model, "processor", None)
    if processor is None:
        processor = try_load_deepseek_ocr_processor(cfg.model_id)
    if processor is None:
        raise RuntimeError(
            "학습용 processor를 찾지 못했습니다. "
            "DeepSeek-OCR은 HF model card에서 infer API만 예시로 제공되므로, "
            "학습용 forward 입력 포맷(pixel_values/labels 등)에 맞춘 processor 또는 collator 구현이 필요합니다."
        )

    train_ds = JsonlOcrDataset(cfg.train_jsonl, max_samples=cfg.max_samples_train)
    collator = VisionTextCollator(processor=processor, prompt=cfg.prompt)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    steps_per_epoch = math.ceil(len(train_dl) / 1)
    total_steps = cfg.max_train_steps or (cfg.num_train_epochs * steps_per_epoch)

    global_step = 0
    model.train()
    pbar = tqdm(range(total_steps), disable=not accelerator.is_main_process, desc="train")

    while global_step < total_steps:
        for batch in train_dl:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and (global_step % cfg.log_every == 0):
                if wandb is not None:
                    wandb.log({"train/loss": float(loss.detach().cpu().item()), "step": global_step})

            if (
                accelerator.is_main_process
                and cfg.val_jsonl is not None
                and cfg.eval_every_steps > 0
                and (global_step > 0)
                and (global_step % cfg.eval_every_steps == 0)
            ):
                # eval은 infer 기반으로 수행 (학습 중엔 느릴 수 있음)
                base_model = accelerator.unwrap_model(model)
                eval_items = _load_jsonl_as_items(cfg.val_jsonl, max_samples=cfg.max_samples_val)
                report = evaluate_dataset_with_infer(
                    model=base_model,
                    tokenizer=tokenizer,
                    items=eval_items,
                    cfg=EvalConfig(
                        prompt=cfg.prompt,
                        base_size=cfg.base_size,
                        image_size=cfg.image_size,
                        crop_mode=cfg.crop_mode,
                        test_compress=cfg.test_compress,
                    ),
                    max_samples=cfg.max_samples_val,
                )
                save_eval_report(report, Path(cfg.output_dir) / f"eval_step_{global_step}.json")
                if wandb is not None:
                    wandb.log(
                        {
                            "eval/cer": report["metrics"]["cer"],
                            "eval/wer": report["metrics"]["wer"],
                            "step": global_step,
                        }
                    )

            global_step += 1
            pbar.update(1)
            if global_step >= total_steps:
                break

    pbar.close()

    # 저장: LoRA면 adapter 저장, 아니면 full model 저장
    if accelerator.is_main_process:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        if cfg.use_lora and isinstance(unwrapped, PeftModel):
            unwrapped.save_pretrained(out)
            tokenizer.save_pretrained(out)
        else:
            unwrapped.save_pretrained(out)
            tokenizer.save_pretrained(out)

    accelerator.wait_for_everyone()



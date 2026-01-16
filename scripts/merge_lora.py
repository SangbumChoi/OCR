from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model and export merged checkpoint")
    p.add_argument("--base_model_id", type=str, required=True)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--merged_out_dir", type=str, required=True)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)
    return p


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(dtype)


def main() -> None:
    args = build_parser().parse_args()
    torch_dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)

    model_kwargs = dict(trust_remote_code=True, use_safetensors=True)
    if args.attn_implementation is not None:
        model_kwargs["_attn_implementation"] = args.attn_implementation
    base = AutoModel.from_pretrained(args.base_model_id, **model_kwargs).eval().cuda().to(torch_dtype)

    peft_model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = peft_model.merge_and_unload()

    out = Path(args.merged_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)


if __name__ == "__main__":
    main()



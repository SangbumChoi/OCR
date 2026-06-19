from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer


@dataclass
class ModelLoadConfig:
    model_id: str
    dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"
    device: str = "cuda"  # "cuda" | "cpu" | "mps"


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_deepseek_ocr_model_and_tokenizer(cfg: ModelLoadConfig) -> Tuple[Any, Any]:
    """
    DeepSeek-OCR loader.
    - Uses trust_remote_code=True, matching the HF model-card example.
    - The attn implementation (e.g. flash_attention_2) is optional.
    """
    torch_dtype = _resolve_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)

    model_kwargs = dict(trust_remote_code=True, use_safetensors=True)
    if cfg.attn_implementation is not None:
        model_kwargs["_attn_implementation"] = cfg.attn_implementation

    model = AutoModel.from_pretrained(cfg.model_id, **model_kwargs)
    model = model.eval()
    if cfg.device != "cpu":
        model = model.to(cfg.device)
    model = model.to(torch_dtype)
    return model, tokenizer


def try_load_deepseek_ocr_processor(model_id: str) -> Any | None:
    """
    The DeepSeek-OCR model card only shows a tokenizer, but
    some versions / derivatives may provide an AutoProcessor, so try it first.
    Returns None on failure.
    """
    try:
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        return None


@torch.inference_mode()
def infer_one_deepseek_ocr(
    *,
    model: Any,
    tokenizer: Any,
    image_file: str,
    prompt: str,
    output_path: str = " ",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    test_compress: bool = False,
    save_results: bool = False,
) -> str:
    """
    Calls DeepSeek-OCR's custom `model.infer(...)`.
    The return type may be a string/dict/etc., so handle it defensively.
    """
    if not hasattr(model, "infer"):
        raise AttributeError("this model has no infer(...). Check that it is a DeepSeek-OCR model.")

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_file,
        output_path=output_path,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        test_compress=test_compress,
        save_results=save_results,
    )

    # The HF card example uses `res` directly, so it is likely a string, but
    # depending on the implementation it may be a dict/list, so coerce to str safely.
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("text", "result", "prediction", "pred"):
            if k in res and isinstance(res[k], str):
                return res[k]
        return str(res)
    return str(res)



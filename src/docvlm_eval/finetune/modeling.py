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
    raise ValueError(f"지원하지 않는 dtype: {dtype}")


def load_deepseek_ocr_model_and_tokenizer(cfg: ModelLoadConfig) -> Tuple[Any, Any]:
    """
    DeepSeek-OCR 로더.
    - HF model card 예시와 동일하게 trust_remote_code=True를 사용합니다.
    - attn 구현(예: flash_attention_2)은 옵션입니다.
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
    DeepSeek-OCR은 model card에서 tokenizer만 예시로 나오지만,
    일부 버전/파생 모델은 AutoProcessor를 제공할 수 있어 우선 시도합니다.
    실패 시 None 반환.
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
    DeepSeek-OCR의 custom `model.infer(...)`를 호출합니다.
    반환 타입이 문자열/딕셔너리 등으로 달라질 수 있어 최대한 방어적으로 처리합니다.
    """
    if not hasattr(model, "infer"):
        raise AttributeError("이 모델에는 infer(...)가 없습니다. DeepSeek-OCR 모델인지 확인하세요.")

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

    # 허깅페이스 카드 예시는 `res`를 바로 사용하므로 문자열일 가능성이 높지만,
    # 구현에 따라 dict/list일 수 있어 안전하게 문자열로 변환합니다.
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in ("text", "result", "prediction", "pred"):
            if k in res and isinstance(res[k], str):
                return res[k]
        return str(res)
    return str(res)



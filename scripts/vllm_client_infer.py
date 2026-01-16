from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def _img_to_data_url(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext in ("png",):
        mime = "image/png"
    elif ext in ("webp",):
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_vllm_chat_completions(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: str,
    api_key: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _img_to_data_url(image_path)}},
                ],
            }
        ],
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM OpenAI-compatible multimodal client (image -> OCR text)")
    p.add_argument("--base_url", type=str, required=True, help='예: "http://localhost:8000/v1"')
    p.add_argument("--model", type=str, required=True, help='vLLM에 로드된 모델명')
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    out = call_vllm_chat_completions(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        image_path=args.image_path,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 표준 OpenAI 응답 형태에서 텍스트만 출력
    try:
        print(out["choices"][0]["message"]["content"])
    except Exception:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



from __future__ import annotations

import argparse

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepSeek-OCR vLLM local inference example (Python API)")
    p.add_argument("--model_id", type=str, default="deepseek-ai/DeepSeek-OCR")
    p.add_argument("--image_file", type=str, required=True)
    p.add_argument("--prompt", type=str, default="<image>\nFree OCR.")
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--skip_special_tokens", type=int, default=0)

    # DeepSeek-OCR vLLM에서 권장되는 NGram logits processor 옵션
    p.add_argument("--ngram_size", type=int, default=30)
    p.add_argument("--window_size", type=int, default=90)
    p.add_argument("--whitelist_token_ids", type=str, default="128821,128822", help="예: '128821,128822'")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # vLLM import는 설치 환경에 따라 달라질 수 있어 런타임 import
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

    whitelist = {int(x.strip()) for x in args.whitelist_token_ids.split(",") if x.strip()}

    llm = LLM(
        model=args.model_id,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
    )

    image = Image.open(args.image_file).convert("RGB")
    model_input = [{"prompt": args.prompt, "multi_modal_data": {"image": image}}]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_args=dict(
            ngram_size=args.ngram_size,
            window_size=args.window_size,
            whitelist_token_ids=whitelist,
        ),
        skip_special_tokens=bool(args.skip_special_tokens),
    )

    outputs = llm.generate(model_input, sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()


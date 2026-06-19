# OCR (DeepSeek-OCR fine-tuning scaffold)

> **Status — legacy scaffold (kept for reference).** This is the *original* LoRA fine-tuning
> scaffold the repo started from. The scripts below still exist and run, but the project has since
> moved to:
> - the unified **`src/docvlm_eval/finetune`** subpackage (install with `pip install -e ".[finetune]"`),
> - the literature-grounded **Part-2 plan** in
>   [`docs/report/part2_ablation_plan.md`](report/part2_ablation_plan.md) (A1–A7 ablations → staircase), and
> - the **synthetic training-data generator** with built-in ground truth
>   ([`scripts/make_realistic_cases.py --count N`](../scripts/make_realistic_cases.py), see
>   [`data/benchmarks/realistic_cases/`](../data/benchmarks/realistic_cases/README.md)).
>
> Prefer those for new work; treat this page as the historical "how the pieces fit" reference.

An **end-to-end scaffold** for fine-tuning DeepSeek-OCR (a VLM).

**Included features**
- **Fine-tuning logic:** a LoRA (PEFT) `transformers` `Seq2SeqTrainer` pipeline.
- **Custom dataset loader:** loads image + target text from JSONL.
- **W&B logging:** train/eval loss and CER/WER logging (optional).
- **Eval metric logging:** computes CER/WER and saves a report.
- **Vanilla vs fine-tuned comparison:** a script to compare both on the same eval set.
- **LoRA apply + merge:** merge the adapter and export a single model.
- **Dataset collection (crawler):** URL-seed-based file download + metadata (basic).
- **vLLM client inference:** calls an OpenAI-compatible `/v1/chat/completions` endpoint.

## Quick start

### 1) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[models,finetune]"     # unified package + LoRA fine-tuning deps
```

### 2) Dataset format

Expects `data/my_dataset/train.jsonl`, `data/my_dataset/val.jsonl`.

Each line has these fields:
- `image_path`: path to the image file (absolute or relative).
- `text`: the target text.

Example:

```json
{"image_path":"data/my_dataset/images/0001.png","text":"Hello world"}
```

### 3) LoRA fine-tuning

```bash
python scripts/finetune_lora.py \
  --model_id deepseek-ai/DeepSeek-OCR \
  --train_jsonl data/my_dataset/train.jsonl \
  --val_jsonl data/my_dataset/val.jsonl \
  --output_dir outputs/exp1 \
  --use_wandb 0
```

### 4) Evaluation

```bash
python scripts/eval.py \
  --model_id outputs/exp1 \
  --val_jsonl data/my_dataset/val.jsonl \
  --report_path outputs/exp1/eval_report.json
```

### 5) Vanilla vs fine-tuned comparison

```bash
python scripts/compare.py \
  --base_model_id deepseek-ai/DeepSeek-OCR \
  --finetuned_model_id outputs/exp1 \
  --val_jsonl data/my_dataset/val.jsonl \
  --report_path outputs/compare_report.json
```

### 6) LoRA merge (export)

```bash
python scripts/merge_lora.py \
  --base_model_id deepseek-ai/DeepSeek-OCR \
  --adapter_dir outputs/exp1 \
  --merged_out_dir outputs/exp1-merged
```

### 7) Data collection (crawler)

```bash
python scripts/crawl.py \
  --seed_url "https://example.com" \
  --out_dir data/crawled \
  --max_pages 50
```

### 8) vLLM (OpenAI-compatible) client inference

```bash
python scripts/vllm_client_infer.py \
  --base_url "http://localhost:8000/v1" \
  --model "deepseek-ocr" \
  --image_path "data/my_dataset/images/0001.png" \
  --prompt "Extract all text from the image."
```

## Inference examples (local)

Two local inference paths that follow the DeepSeek-OCR HF model-card examples.
Reference: [`deepseek-ai/DeepSeek-OCR` model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

### A) HuggingFace (`model.infer`)

```bash
python scripts/hf_infer.py \
  --model_id deepseek-ai/DeepSeek-OCR \
  --image_file "your_image.jpg" \
  --prompt "<image>\n<|grounding|>Convert the document to markdown. "
```

### B) vLLM Python API (local)

```bash
python scripts/vllm_local_infer.py \
  --model_id deepseek-ai/DeepSeek-OCR \
  --image_file "your_image.jpg" \
  --prompt "<image>\nFree OCR."
```

### C) Batch inference over `examples/` → saved to `docs/results/`

```bash
python scripts/run_examples_hf_infer.py \
  --model_id deepseek-ai/DeepSeek-OCR \
  --examples_dir examples \
  --results_dir docs/results \
  --prompt "<image>\n<|grounding|>Convert the document to markdown. "
```

## Notes
- DeepSeek-OCR's exact prompt / special-token conventions can vary by model and serving mode, so
  this scaffold defaults to a generic "image → text" VLM OCR pipeline.
- Provide these two details and the prompt / input format can be tailored to DeepSeek-OCR:
  - (A) the **base model ID** to use (HF repo or local path);
  - (B) whether the training labels are **plain text** or a structure such as **markdown / HTML / JSON**.

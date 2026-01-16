# OCR (DeepSeek OCR Finetune Scaffold)

DeepSeek OCR(VLM) 파인튜닝을 위한 **엔드투엔드 스캐폴딩**입니다.

포함 기능
- **파인튜닝 로직**: LoRA(PEFT) 기반 `transformers` `Seq2SeqTrainer` 파이프라인
- **커스텀 데이터셋 로더**: JSONL 기반 이미지+정답 텍스트 로딩
- **W&B 로깅**: 학습/평가 손실 및 CER/WER 로깅(옵션)
- **평가 메트릭 로깅**: CER/WER 계산 및 리포트 저장
- **바닐라 vs 파인튜닝 비교**: 동일 eval set에서 비교 스크립트 제공
- **LoRA 적용 및 merge**: 어댑터 merge 후 단일 모델로 export
- **데이터셋 수집(크롤러)**: URL seed 기반 파일 다운로드/메타 기록(기본형)
- **vLLM 클라이언트 추론**: OpenAI-compatible `/v1/chat/completions` 호출 스크립트

## 빠른 시작

### 1) 환경 준비

```bash
cd /Users/sangbumchoi/Documents/OCR
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2) 데이터셋 포맷

`data/my_dataset/train.jsonl`, `data/my_dataset/val.jsonl` 형태를 기대합니다.

각 라인은 다음 필드를 가집니다:
- `image_path`: 이미지 파일 경로(절대/상대 모두 가능)
- `text`: 정답 텍스트

예시:

```json
{"image_path":"data/my_dataset/images/0001.png","text":"Hello world"}
```

### 3) LoRA 파인튜닝

```bash
python scripts/finetune_lora.py \
  --model_id deepseek-ai/DeepSeek-OCR \
  --train_jsonl data/my_dataset/train.jsonl \
  --val_jsonl data/my_dataset/val.jsonl \
  --output_dir outputs/exp1 \
  --use_wandb 0
```

### 4) 평가

```bash
python scripts/eval.py \
  --model_id outputs/exp1 \
  --val_jsonl data/my_dataset/val.jsonl \
  --report_path outputs/exp1/eval_report.json
```

### 5) 바닐라 vs 파인튜닝 비교

```bash
python scripts/compare.py \
  --base_model_id deepseek-ai/DeepSeek-OCR \
  --finetuned_model_id outputs/exp1 \
  --val_jsonl data/my_dataset/val.jsonl \
  --report_path outputs/compare_report.json
```

### 6) LoRA merge(export)

```bash
python scripts/merge_lora.py \
  --base_model_id deepseek-ai/DeepSeek-OCR \
  --adapter_dir outputs/exp1 \
  --merged_out_dir outputs/exp1-merged
```

### 7) 데이터 수집(크롤러)

```bash
python scripts/crawl.py \
  --seed_url "https://example.com" \
  --out_dir data/crawled \
  --max_pages 50
```

### 8) vLLM(OpenAI-compatible) 클라이언트 추론

```bash
python scripts/vllm_client_infer.py \
  --base_url "http://localhost:8000/v1" \
  --model "deepseek-ocr" \
  --image_path "data/my_dataset/images/0001.png" \
  --prompt "Extract all text from the image."
```

## 중요 메모
- DeepSeek-OCR의 **정확한 프롬프트/특수 토큰 규약**은 모델/서빙 방식에 따라 달라질 수 있어, 이 스캐폴딩은 기본적으로 “image → text” 형태의 일반 VLM OCR 파이프라인을 제공합니다.
- 다음 정보 2가지만 주시면, DeepSeek-OCR에 맞춘 프롬프트/입력 포맷을 즉시 반영해 드릴게요:
  - (A) 사용할 **베이스 모델 ID**(HF repo 또는 로컬 경로)
  - (B) 학습 라벨이 **plain text**인지, 아니면 **markdown/html/JSON** 같은 구조인지

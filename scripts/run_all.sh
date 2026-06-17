#!/usr/bin/env bash
# Full reproduction: build benchmarks -> evaluate every model -> build comparison table.
# Designed for a free Colab/Kaggle T4. Override LIMIT for a fast smoke pass.
#
#   bash scripts/run_all.sh             # default: 300 samples/benchmark
#   LIMIT=20 bash scripts/run_all.sh    # quick sanity pass
set -euo pipefail

LIMIT="${LIMIT:-300}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
RESULTS="${RESULTS:-results}"

MODELS=(internvl2_5-1b internvl3-1b smolvlm-500m smolvlm-256m llava-ov-0.5b got-ocr2 florence2-large paddleocr-vl)
BENCHMARKS=(docvqa infovqa chartqa ocrbench)

echo "==> Building benchmarks (limit=$LIMIT)"
for b in "${BENCHMARKS[@]}"; do
  python scripts/build_benchmarks.py --benchmark "$b" --limit "$LIMIT"
done

echo "==> Building robustness probe from DocVQA"
python scripts/build_robustness_set.py --base data/benchmarks/docvqa.jsonl \
  --out-dir data/robustness/docvqa --limit 100

echo "==> Evaluating models"
for m in "${MODELS[@]}"; do
  for b in "${BENCHMARKS[@]}"; do
    python scripts/evaluate.py --model "$m" --benchmark "data/benchmarks/${b}.jsonl" \
      --benchmark-name "$b" --out "${RESULTS}/${m}/${b}" \
      --device "$DEVICE" --dtype "$DTYPE" --limit "$LIMIT" || echo "[skip] $m/$b failed"
  done
  # robustness only needs to run for the leading candidate(s); run for all that succeed
  python scripts/evaluate.py --model "$m" --benchmark data/robustness/docvqa/robustness.jsonl \
    --benchmark-name robustness --out "${RESULTS}/${m}/robustness" \
    --device "$DEVICE" --dtype "$DTYPE" || echo "[skip] $m/robustness failed"
done

echo "==> Building comparison table"
python scripts/make_comparison_table.py --results-dir "$RESULTS" --out-dir "$RESULTS"
echo "==> Done. See ${RESULTS}/comparison_table.md"

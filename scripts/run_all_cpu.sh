#!/usr/bin/env bash
# Attempt EVERY registered model on CPU over a benchmark, smallest-first, with a per-model
# wall-clock budget so a slow/hung 1B model can't block the sweep. Results are stored per model
# and aggregated cumulatively into results/matrix_<bench>.{md,json} (run_matrix scans all that
# have results). Models that time out / fail are recorded in the run status, not fatal.
#
#   bash scripts/run_all_cpu.sh                       # default: capability probe
#   BENCH=data/benchmarks/all_preview.jsonl TIMEOUT=1800 bash scripts/run_all_cpu.sh
set -uo pipefail

BENCH="${BENCH:-data/benchmarks/capability_probe/capability.jsonl}"
TIMEOUT="${TIMEOUT:-1500}"     # seconds per model
MNT="${MNT:-64}"              # max new tokens

# smallest -> largest so the matrix fills early
MODELS=(
  dummy-echo
  smolvlm-256m smoldocling-256m florence2-base smolvlm-500m got-ocr2
  florence2-large h2ovl-0.8b llava-ov-0.5b
  internvl2-1b internvl2_5-1b internvl3-1b ovis2-1b
  paddleocr-vl paddleocr-vl-1.5
)

for m in "${MODELS[@]}"; do
  echo "==> [$(date +%H:%M:%S)] $m (budget ${TIMEOUT}s)"
  timeout "${TIMEOUT}" python3 scripts/run_matrix.py --models "$m" \
      --benchmark "$BENCH" --device cpu --dtype float32 --max-new-tokens "$MNT" \
      >/dev/null 2>"results/_log_${m}.txt"
  code=$?
  if [ $code -eq 124 ]; then echo "    TIMEOUT after ${TIMEOUT}s"; fi
  if [ $code -ne 0 ] && [ $code -ne 124 ]; then echo "    exit=$code (see results/_log_${m}.txt)"; fi
done

# final cumulative aggregation
python3 scripts/run_matrix.py --models dummy-echo --benchmark "$BENCH" --device cpu >/dev/null 2>&1
echo "==> DONE. matrix at results/matrix_$(basename "${BENCH%.jsonl}").md"

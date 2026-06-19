#!/usr/bin/env bash
# PaddleOCR-VL needs transformers==4.55 (config-declared) — newer than the <5 pin the other
# remote-code models require, and 5.x breaks its rope config. So run the PaddleOCR-VL family in
# an ISOLATED venv, then aggregate into the shared docs/results/ matrix.
#
#   bash scripts/run_paddleocr_env.sh
set -uo pipefail

ENVDIR="${ENVDIR:-.venv-paddle}"
BENCH="${BENCH:-data/benchmarks/capability_probe/capability.jsonl}"
PY="$ENVDIR/bin/python"

if [ ! -x "$PY" ]; then
  echo "==> creating $ENVDIR (inherits system torch/pillow; adds transformers 4.55)"
  python3 -m venv --system-site-packages "$ENVDIR"
  "$ENVDIR/bin/pip" install -q -U pip
  "$ENVDIR/bin/pip" install -q "transformers==4.55.0" protobuf accelerate safetensors einops
fi
echo "==> $($PY -c 'import transformers;print("transformers",transformers.__version__)')"

for m in paddleocr-vl paddleocr-vl-1.5 paddleocr-vl-1.6; do
  echo "==> [$(date +%H:%M:%S)] $m"
  timeout "${TIMEOUT:-1800}" "$PY" scripts/run_matrix.py --models "$m" \
    --benchmark "$BENCH" --device cpu --dtype float32 --max-new-tokens 64 --no-resume \
    >/dev/null 2>"docs/results/_log_${m}.txt" || echo "   exit=$? (see docs/results/_log_${m}.txt)"
done

# aggregate (main interpreter is fine — just reads stored per_sample.json)
python3 scripts/run_matrix.py --models dummy-echo --benchmark "$BENCH" --device cpu >/dev/null 2>&1
echo "==> DONE. PaddleOCR-VL rows in docs/results/matrix_$(basename "${BENCH%.jsonl}").md"

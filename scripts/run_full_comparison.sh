#!/usr/bin/env bash
# Full sub-1B (+ a few slightly-larger reference) VLM comparison, measuring score + inference time
# + memory (the model wrapper records load/latency/peak-CPU/peak-GPU into each summary.json).
#
# Designed to run from a FRESH CLONE on a GPU box (e.g. Colab T4). Three transformers passes are
# needed because model families pin different versions:
#   pass 1  established chat VLMs           -> transformers 4.49
#   pass 2  PaddleOCR-VL 1.0/1.5/1.6        -> transformers 4.57
#   pass 3  newer 2025-26 VLMs (Ovis2.5,    -> transformers latest
#           MiniCPM-V-4.6, LFM2.5-VL,
#           Qwen3.5-VL, LightOnOCR)
#
# Every (model x benchmark) run prints a global progress line "[done/total] stage: model x bench"
# so it is clear how many experiments remain and which stage is running.
#
#   DEVICE=cuda bash scripts/run_full_comparison.sh
set -uo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MNT="${MNT:-64}"
CAP=data/probes/capability_probe/capability.jsonl
SCP=data/probes/spatial_context_probe/probe.jsonl
CEV=data/probes/custom_eval/custom_eval.jsonl
OOV=data/probes/oov_probe/oov.jsonl
WEB=data/probes/webui_probe/webui.jsonl

CHAT=(internvl2-1b internvl2_5-1b internvl3-1b smolvlm-256m smolvlm-500m smoldocling-256m
      llava-ov-0.5b got-ocr2 florence2-base florence2-large h2ovl-0.8b ovis2-1b)
PADDLE=(paddleocr-vl paddleocr-vl-1.5 paddleocr-vl-1.6)
# Newer releases (2025-26). Note: Ovis2.5 has no 1B (smallest is 2B); MiniCPM-V-4.6 / LFM2.5-VL
# are ~1.3-1.6B — kept as stronger reference points. Qwen3.5-0.8B is sub-1B.
NEW=(ovis2_5-2b minicpm-v-4_6 lfm2_5-vl-1.6b qwen3_5-0.8b lightonocr-1b)

# --- progress accounting: total (model x benchmark) experiments across all three passes ---
CHAT_BENCHES=5; PADDLE_BENCHES=3; NEW_BENCHES=3
TOTAL=$(( ${#CHAT[@]} * CHAT_BENCHES + ${#PADDLE[@]} * PADDLE_BENCHES + ${#NEW[@]} * NEW_BENCHES ))
DONE=0
STAGE="init"

run() {  # run <model> <benchmark.jsonl>
  DONE=$((DONE + 1))
  local left=$(( TOTAL - DONE ))
  local bench; bench=$(basename "$2" .jsonl)
  echo "[$DONE/$TOTAL] (${left} left) ${STAGE}: $1 x ${bench}"
  python scripts/run_matrix.py --models "$1" --benchmark "$2" \
        --device "$DEVICE" --dtype "$DTYPE" --max-new-tokens "$MNT" --no-resume || true
}

echo "== plan: $TOTAL experiments  (chat ${#CHAT[@]}x${CHAT_BENCHES}, paddle ${#PADDLE[@]}x${PADDLE_BENCHES}, new ${#NEW[@]}x${NEW_BENCHES}) =="

echo "== deps for the custom set (CJK fonts + QR/barcode) =="
(apt-get -qq update && apt-get -qq install -y fonts-noto-cjk) >/dev/null 2>&1 || true
pip -q install qrcode python-barcode >/dev/null 2>&1 || true

echo "== probes =="
python scripts/make_capability_probe.py
python scripts/make_spatial_context_probe.py
python scripts/make_custom_eval.py
python scripts/make_oov_probe.py
python scripts/make_webui_probe.py

STAGE="pass1/chat@tf4.49"
echo "== ${STAGE}: ${#CHAT[@]} models x ${CHAT_BENCHES} benchmarks =="
pip -q install "transformers==4.49.0" peft >/dev/null 2>&1 || true
for m in "${CHAT[@]}"; do
  run "$m" "$CAP"; run "$m" "$SCP"; run "$m" "$CEV"; run "$m" "$OOV"; run "$m" "$WEB"; done

STAGE="pass2/paddle@tf4.57"
echo "== ${STAGE}: ${#PADDLE[@]} models x ${PADDLE_BENCHES} benchmarks =="
pip -q install "transformers==4.57.1" protobuf >/dev/null 2>&1 || true
for m in "${PADDLE[@]}"; do run "$m" "$CAP"; run "$m" "$SCP"; run "$m" "$CEV"; done

STAGE="pass3/new@tf-latest"
echo "== ${STAGE}: ${#NEW[@]} models x ${NEW_BENCHES} benchmarks =="
# newest releases want a recent transformers; upgrade and let per-model failures be captured as data
pip -q install -U transformers accelerate timm >/dev/null 2>&1 || true
for m in "${NEW[@]}"; do run "$m" "$CAP"; run "$m" "$SCP"; run "$m" "$CEV"; done

echo "== aggregate + analysis =="
for b in "$CAP" "$SCP" "$CEV" "$OOV" "$WEB"; do
  python scripts/run_matrix.py --models dummy-echo --benchmark "$b" --device cpu >/dev/null 2>&1
done
python scripts/analyze_probe_signals.py --probe probe || true
python scripts/analyze_custom_eval.py || true
python scripts/build_insights.py || true
echo "== DONE ($DONE/$TOTAL experiments run): docs/results/matrix_*.md, probe_signals.json, custom_eval_breakdown.md, docs/report/insights.md =="

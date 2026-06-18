#!/usr/bin/env bash
# Full sub-1B VLM comparison incl. PaddleOCR-VL 1.0/1.5/1.6, measuring score + inference time +
# memory (the model wrapper records load/latency/peak-CPU/peak-GPU into each summary.json).
#
# Designed to run from a FRESH CLONE on a GPU box (e.g. Colab T4). Two transformers passes are
# needed because the chat VLMs want transformers<5 while PaddleOCR-VL wants 4.57.
#
#   DEVICE=cuda bash scripts/run_full_comparison.sh
set -uo pipefail

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MNT="${MNT:-64}"
CAP=data/benchmarks/capability_probe/capability.jsonl
SCP=data/benchmarks/spatial_context_probe/probe.jsonl
CEV=data/benchmarks/custom_eval/custom_eval.jsonl

CHAT=(internvl2-1b internvl2_5-1b internvl3-1b smolvlm-256m smolvlm-500m smoldocling-256m
      llava-ov-0.5b got-ocr2 florence2-base florence2-large h2ovl-0.8b ovis2-1b)
PADDLE=(paddleocr-vl paddleocr-vl-1.5 paddleocr-vl-1.6)

run() { python scripts/run_matrix.py --models "$1" --benchmark "$2" \
        --device "$DEVICE" --dtype "$DTYPE" --max-new-tokens "$MNT" --no-resume || true; }

echo "== deps for the custom set (CJK fonts + QR/barcode) =="
(apt-get -qq update && apt-get -qq install -y fonts-noto-cjk) >/dev/null 2>&1 || true
pip -q install qrcode python-barcode >/dev/null 2>&1 || true

echo "== probes =="
python scripts/make_capability_probe.py
python scripts/make_spatial_context_probe.py
python scripts/make_custom_eval.py

echo "== pass 1: chat VLMs (transformers 4.49) =="
pip -q install "transformers==4.49.0" peft >/dev/null 2>&1 || true
for m in "${CHAT[@]}"; do echo "-> $m"; run "$m" "$CAP"; run "$m" "$SCP"; run "$m" "$CEV"; done

echo "== pass 2: PaddleOCR-VL 1.0/1.5/1.6 (transformers 4.57) =="
pip -q install "transformers==4.57.1" protobuf >/dev/null 2>&1 || true
for m in "${PADDLE[@]}"; do echo "-> $m"; run "$m" "$CAP"; run "$m" "$CEV"; done

echo "== aggregate + analysis =="
python scripts/run_matrix.py --models dummy-echo --benchmark "$CAP" --device cpu >/dev/null 2>&1
python scripts/run_matrix.py --models dummy-echo --benchmark "$SCP" --device cpu >/dev/null 2>&1
python scripts/run_matrix.py --models dummy-echo --benchmark "$CEV" --device cpu >/dev/null 2>&1
python scripts/analyze_probe_signals.py --probe probe || true
python scripts/analyze_custom_eval.py || true
echo "== DONE: results/matrix_{capability,probe,custom_eval}.md, probe_signals.json, custom_eval_breakdown.md =="

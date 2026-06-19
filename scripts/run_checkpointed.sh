#!/usr/bin/env bash
# Resumable, git-checkpointed evaluation for GPU-limited / ephemeral environments (e.g. Colab T4).
#
# Problem: the eval container is reset when the free-GPU limit is hit, losing docs/results/.
# This wrapper makes a sweep survive that:
#   1. `git pull` first  -> restore any partial results pushed by an earlier (killed) session.
#   2. run each model, then `git add docs/results && commit && push` its predictions+summary.
# The pipeline already writes predictions.jsonl per sample (flush) and resumes by sample_id, so a
# re-run after a reset skips everything already done and continues from where it stopped.
# run_matrix aggregates the matrix from every summary.json on disk, so running one model at a time
# still yields the full cross-model matrix once the others are restored by the pull.
#
# Usage (Colab needs git identity + push credentials configured for the repo):
#   MODELS="smolvlm-256m smolvlm-500m internvl3-1b" \
#   BENCH=data/benchmarks/preview_eval.jsonl NAME=preview_eval DEVICE=cuda \
#   bash scripts/run_checkpointed.sh
#
# Env: MODELS, BENCH, NAME, DEVICE (cuda|cpu), BRANCH (default = current), EXTRA (extra run_matrix args).
set -uo pipefail
cd "$(dirname "$0")/.."

BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
BENCH="${BENCH:-data/benchmarks/preview_eval.jsonl}"
NAME="${NAME:-preview_eval}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-smolvlm-256m smolvlm-500m}"
EXTRA="${EXTRA:-}"

_push() { for i in 1 2 3 4; do git push origin "HEAD:$BRANCH" && return 0; sleep $((2 ** i)); done; echo "[warn] push failed (kept locally)"; }

echo "== checkpointed eval: branch=$BRANCH  bench=$NAME  device=$DEVICE  models=[$MODELS] =="
git pull --no-edit origin "$BRANCH" || echo "[warn] initial pull failed (continuing)"

for m in $MODELS; do
  echo "== model: $m =="
  # resume is on by default; per-sample flush means a mid-model kill keeps progress on disk
  python scripts/run_matrix.py --models "$m" --benchmark "$BENCH" --benchmark-name "$NAME" \
    --device "$DEVICE" $EXTRA || echo "[warn] $m errored — partial predictions kept for resume"
  git add docs/results
  if git diff --cached --quiet; then
    echo "   (no new results to checkpoint for $m)"
  else
    git commit -q -m "ckpt($NAME): $m" && _push
  fi
done
echo "== done: $NAME  (matrix: docs/results/matrix_${NAME}.md) =="

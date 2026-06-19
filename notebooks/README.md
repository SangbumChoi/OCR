# Running the comparison on a GPU (Colab T4)

I (the assistant) **cannot access or run your Colab/T4** — I'm in an isolated container with no
GPU and no Google auth. So this is set up to be one-click for you.

## Option A — open the ready notebook
Open [`colab_full_comparison.ipynb`](colab_full_comparison.ipynb) in Colab → Runtime → change to
**T4 GPU** → **Run all**. It clones the repo, installs it, runs `scripts/run_full_comparison.sh`,
prints the score+efficiency matrices, and downloads `docs/results/`.

## Option B — paste ONE cell into your existing Colab
```python
!git clone https://github.com/SangbumChoi/OCR.git 2>/dev/null; \
 cd OCR && git checkout claude/new-session-w79q0i && git pull --ff-only && \
 pip -q install -e '.[models,finetune]' protobuf && \
 DEVICE=cuda bash scripts/run_full_comparison.sh && \
 echo "===== CAPABILITY (scores + efficiency) =====" && cat docs/results/matrix_capability.md && \
 echo "===== SPATIAL/CONTEXT SIGNALS =====" && python scripts/analyze_probe_signals.py --probe probe
```

## What it produces
- `docs/results/matrix_capability.md` — per-model scores **+ an Efficiency table** (device, params,
  load(s), avg/p90 latency, peak CPU MB, peak GPU MB) measured by the model wrapper.
- `docs/results/matrix_probe.md` + `docs/results/probe_signals.json` — spatial/context shortcut-robust
  PASS/FAIL signals.
- `docs/results/<model>/<probe>/summary.json` — full per-model metrics incl. timing/memory.

It runs two transformers passes automatically (4.49 for the chat VLMs, 4.57 for PaddleOCR-VL
1.0/1.5/1.6), so all models — including the previously-failing ones — are compared in one go.

## Sharing results back
Either download the `docvlm_results.zip` the notebook offers, or commit them:
```python
!cd OCR && git add -f docs/results/matrix_*.md docs/results/*.json && \
 git -c user.email=you@example.com -c user.name=you commit -m "T4 results" && git push
```
(then I can read and interpret the numbers).

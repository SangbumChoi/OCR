# UDD ablation reproduction — docvlm-eval PoC (CPU verification + GPU measurement)

Step-by-step recipe to reproduce the public-data ablation (A1–A6 on
[`danelcsb/UDD`](https://huggingface.co/datasets/danelcsb/UDD)) from a fresh machine. Two tiers:
everything except the actual LoRA training verifies on a **CPU-only box**; the measurements run on
a **single T4** (Colab is enough). The one notebook that drives both:
`notebooks/udd_ablation.ipynb`.

## 0. What is being reproduced

Each ablation arm pair differs in exactly ONE factor at an equal sample count (equal-N control),
trains a LoRA on that mix, and is scored on the ENTIRE suite **before and after** training — the
claim under test is never "the arm's own axis improved" but "the whole capability surface moved":

| family | factor | arms |
|---|---|---|
| A1 | does grounding (WHERE) supervision help? | `spotting_on` vs `spotting_off` |
| A2 | rationale text vs answer-only, same records | `reason_chain` vs `reason_answer` |
| A3 | structured signals vs mere task mixing | `base` / `spot` / `reason` / `spot_reason` |
| A4 | language mixing (en vs en+X) | `en`, `en_ko`, `en_ja`, `en_zh`, … |
| A5 | LoRA placement, fixed composite mix | `vision` / `connector` / `llm_attn` / `llm_mlp` |
| A6 | LoRA rank (alpha = 2r), fixed composite mix | `r8` / `r16` / `r32` / `r64` |

Results land in `docs/results/udd_ablation_results.json` under `models.<model>.U-<arm>` with
`probes_before` / `probes` / `delta` (per-suite and per-axis) and `preds_dir` pointing at
per-sample `preds_before/` + `preds_after/` jsonls (the flipped-example evidence).

## 1. Setup

```bash
git clone https://github.com/SangbumChoi/OCR.git && cd OCR
git checkout claude/new-session-w79q0i
pip install -e '.[models,finetune]'
```

On Colab you skip all of this — the notebook's first cell clones + installs by itself, and pulls
the corpus from the Hub (`danelcsb/UDD`, 28,299 image-rows / 53,108 QAs / 32 sources) when no
local build exists. GPU caveat handled by section 6 automatically: LFM2.5-VL/Qwen3.5 need
`transformers>=5` and `torchao>=0.16` (Colab preinstalls older ones); the cell force-upgrades both
before training. Checked-in probe fixtures carry paths from the generating machine —
`load_jsonl` re-anchors them at the local repo root, so any clone location works.

## 2. CPU verification (no GPU, ~30 min total)

```bash
python -m pytest tests/ -q                      # ~240 offline tests must pass
jupyter nbconvert --to notebook --execute --inplace notebooks/udd_ablation.ipynb
```

or run the notebook interactively — sections 1–5 are the CPU tier:

1. **corpus sanity** — native `instructions`/`answers` lists, `validate_payload_shapes`, fold split;
2. **pools** — `build_task_trainsets --per-task -1` writes FULL per-task/per-language pools
   (equal-N moves to arm composition), plus the derived A2 chain/answer pair and text probes;
3. **arm compose (dry-run) + leakage check** — no heldout image may appear in any arm's mix;
4. **A2 factor isolation + text probes** — inspect the exact record pairs;
5. **mock multi-model eval** — deterministic mock readers (oracle / caseflip / wrapped / truncate /
   constant / echo-question) through the REAL metric dispatch; oracle must be ≈1.0, constant ≈0.0.

The cheapest real train+validate proof (also CPU, slow): one arm on the 256M smoke base —

```bash
python scripts/run_udd_ablation.py --arm A1_spotting_on --smoke --models smolvlm-256m \
       --results docs/results/udd_ablation_smoke.json
```

A committed run of exactly this lives at `docs/results/udd_ablation_smoke.json`, so notebook
sections 9–10 (full-comparison table/heatmap + flipped examples) render on a fresh clone.

## 3. GPU measurement (T4 / Colab)

Open `notebooks/udd_ablation.ipynb` in Colab with a GPU runtime and run all cells — with a GPU it
skips the dry-run and section 6 trains the ENTIRE ladder directly:

```bash
python scripts/run_udd_ablation.py --arm A1 A2 A3 A4 A5 A6 --steps 300
```

`--count 0` (default) = use ALL images: each FAMILY trains at the largest equal-N every one of its
arms supports (at the 28k corpus: A2 ≈ 7.3k, A5/A6 ≈ 4.1k, A1/A3 ≈ 2.1k, A4 = smallest language
pool). `--smoke` first (count=24, steps=8) is a 1–2 h wiring proof of all 24 arms if you want a
cheap end-to-end pass before the measurement. Single-factor knobs pass through:
`--placement`, `--lora-r/--lora-alpha`, `--max-image-long-side`.

## 4. Reading the results

Notebook sections 7–10: per-arm capability scores; the full comparison (coarse per-suite
`before → after (Δ)` table + fine-grained `suite:axis` delta heatmap over capability / spatial /
realistic / heldout); and `flipped_examples()` — join `preds_before`/`preds_after` by sample id,
rank by |Δscore|, display the images. A factor "helps" only if the heldout and cross-capability
deltas move, not just its own axis.

## 5. Rebuilding the data (optional)

```bash
python scripts/build_udd.py --per-bench 1000 --max-scan 40000   # re-stream sources, dedup-cached
python scripts/audit_udd_duplicates.py                          # 0 cross-source exact expected
```

The corpus release on the Hub already contains everything the ablation needs (`fold` gives the
leakage-safe public heldout); rebuilding is only for changing the scale or adding sources.

## Pitfalls that cost time (all fixed in the pipeline, kept for context)

- `transformers<5` (the `[models]` extra pins it for the remote-code sweep) blocks LFM/Qwen —
  section 6 upgrades it; `torchao 0.10` preinstalled on Colab is rejected by transformers 5 —
  upgraded in the same command.
- A failed arm prints the child's stderr tail directly under the notebook cell; the parent also
  prints the failing command + common causes. If an arm is "unknown", the pools were not built.
- datasets `map` caching served stale enrichment — every enrich/dedupe map runs with
  `load_from_cache_file=False`; delete stray `cache-*.arrow` files if disk fills.

# Probes & synthetic sets — *our own* evaluation data (not public benchmarks)

These are **rendered/derived by this repo**, not third-party public benchmarks, so they live here
rather than in [`../benchmarks/`](../benchmarks/README.md) (which holds real public datasets).
They are still catalogued in [`../../configs/benchmark_catalog.yaml`](../../configs/benchmark_catalog.yaml)
with `kind: probe` or `kind: synthetic`.

| Dir | kind | What it is |
| --- | ---- | ---------- |
| `capability_probe/`      | probe     | isolates document-VLM capability axes on controlled renders (exact GT incl. boxes) |
| `custom_eval/`           | probe     | proposed per-content-class eval format (rotation / language / reading-direction / spotting) |
| `oov_probe/`             | probe     | out-of-vocabulary / invented-script fallback behaviour |
| `webui_probe/`           | probe     | web UI/UX agent understanding |
| `spatial_context_probe/` | probe     | control-paired falsifiable probes (counterfactual / distractor / absence) |
| `realistic_cases/`       | probe     | realistic document special cases with built-in GT (HTML→WeasyPrint→Augraphy), paired clean/degraded |
| `recognition_fullpage/`  | synthetic | illustrative full-page printed-text recognition (no clean HF source) |
| `scenetext/`             | synthetic | illustrative scene-text detection+recognition |
| `robustness/`            | synthetic | a degraded copy derived from the DocVQA sample (robustness retention) |

Regenerate with the `scripts/make_*.py` generators (see each one's header).

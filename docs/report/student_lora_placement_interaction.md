# Vision and connector LoRA interaction

## Motivation

The preliminary LFM spotting experiment favors vision-only LoRA over connector-only LoRA, but it
does not test whether the connector adds value after the visual representation has adapted. The
existing `all` placement is not an answer because it also adapts the language model. It changes
three module families and cannot identify a vision-by-connector interaction.

The authenticated W&B snapshot is preserved in
[`../results/lfm_ablation_wandb_snapshot.json`](../results/lfm_ablation_wandb_snapshot.json), with
its fail-closed audit in
[`../results/lfm_ablation_wandb_analysis.md`](../results/lfm_ablation_wandb_analysis.md). Of ten
observed runs, five have heldout metrics, four are marked finished without evaluation, and the A0
baseline crashed. W&B run state is therefore not accepted as experimental completion.

Within the only directly comparable preliminary pair, both A1 spotting runs share the visible
epoch, learning-rate, rank, and training-file controls. Vision minus connector is +0.0486 overall,
+0.0607 grounding, +0.0234 L1-locate, -0.0581 L1-region, and 0.0000 KIE. This is one pair with
missing immutable seed, dose, heldout-ID, and base-revision controls. It establishes direction for
the confirmatory design, not a promotion result, and cross-arm A2/A4/A7 score comparisons remain
confounded.

The resolver now exposes `vision_connector` as the exact union of the two disjoint target sets.
This creates the direct comparison required by the current hypothesis:

```text
vision-only spotting adaptation
versus
vision + connector spotting adaptation
```

## Adapter-budget control

Applying the same rank to more modules increases trainable parameters. A gain from the combined arm
could therefore reflect adapter capacity rather than connector involvement. The confirmatory sweep
uses vision rank 16 as its trainable-parameter reference. For each placement, the runner computes:

```text
LoRA parameters per rank = sum(in_features + out_features)
target budget = vision parameters per rank * 16
effective rank = nearest integer(target budget / arm parameters per rank)
```

Alpha is rescaled to preserve the requested alpha-to-rank ratio. The run fails before training if
the analytical or realized PEFT trainable-parameter error exceeds 5%. Every adapter stores
`lora_budget.json`; W&B receives the same report under `config.lora_budget`.

This controls parameter count, not optimization geometry. The combined arm still distributes its
rank across a different set of matrices, which is the intended treatment.

## Base-weight memory contract

LoRA rank controls adapter parameters but does not by itself reduce the frozen LFM base-weight
footprint. The runner therefore defaults to four-bit NF4 QLoRA with double quantization and
bfloat16 compute. `--quantization-bits 16` is the explicit dense-base control. Four-bit mode
requires CUDA and fails before model loading on CPU. Each `lora_budget.json` records the requested
bits, realized loading mode, quantization type, double-quantization flag, and compute dtype.

## Paired design

[`../../configs/lora_vision_connector_sweep.yaml`](../../configs/lora_vision_connector_sweep.yaml)
defines two placements over three paired replicates. Within each replicate, both cells share:

- the LFM2.5-VL-1.6B base;
- A1 normalized-coordinate spotting targets;
- the synthetic training and heldout sets;
- optimizer seed, sample count, steps, image cap, grounding repeat, and evaluation cap;
- vision rank-16 trainable-parameter budget.

Only placement membership and the rank required to preserve that budget differ.

```bash
python scripts/run_lora_placement_sweep.py --dry-run
python scripts/run_lora_placement_sweep.py
python scripts/analyze_lora_placement_sweep.py
```

The full run comprises six GPU jobs. Distinct result keys and W&B names preserve both placement and
replicate. The dry run is the authoritative wiring check; no quality result is claimed until all
six jobs complete. The analyzer fails closed on missing cells or evidence and writes paired metric
distributions, deterministic bootstrap intervals, gate outcomes, and the promotion decision under
`docs/results/lora_vision_connector_sweep.{json,md}`.

## Decision

Promote `vision_connector` only if the paired heldout evidence shows:

1. positive mean grounding and `L1-locate` deltas versus vision-only;
2. no material regression on KIE, OCR, multilingual, or reading-order axes;
3. trainable-parameter budget error at or below 5% for every run;
4. no dependence on a single replicate;
5. the heldout gain is not accompanied by a larger train-minus-heldout gap.

If the combined arm fails these gates, keep vision-only LoRA. That outcome would support the
interpretation that the current connector is not the limiting spotting pathway.

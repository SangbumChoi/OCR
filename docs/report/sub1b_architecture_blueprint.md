# Sub-1B document VLM blueprint

## Decision

Build an approximately 800M-parameter student whose architecture, initialization, data mixture,
losses, and post-training rewards are controlled by
[`configs/sub1b_architecture.yaml`](../../configs/sub1b_architecture.yaml). LFM2.5-VL-1.6B remains
the fast experimental base and a teacher candidate; it is not the deployment-size student.

The first question is not whether transfer is allowed. It is **which inherited representations are
worth their dependency on a pretrained checkpoint**. Five initialization arms therefore span fully
random, vision-only, language-only, dual-tower, and alternating-block selective transfer. Every arm
uses the same architecture, token budget, offline sequence targets, and native objectives. Online
native-teacher feature and output distillation is a separate checkpoint-dependent experiment. This
separates initialization benefit from teacher supervision.

## Architecture and parameter budget

| Component | Default | Estimated parameters | Reason |
| --- | --- | ---: | --- |
| Vision tower | ViT, 12 layers, width 768, patch 14 | 88.7M | Retains fine glyph detail without spending most of the budget on natural-image capacity. |
| Language decoder | 23 layers, width 1536, GQA 24Q/8KV, 64k vocabulary | 677.5M | Holds multilingual emission, cross-region binding, arithmetic, and structured generation. |
| Connector | Two-layer gated resampler, 64 visual tokens | 33.2M | Makes compression an explicit, ablatable bottleneck instead of a fixed projection. |
| Temporary task heads | contrast, orientation, normalized box regression | 0.6M | Supplies dense pretraining signals; removable for deployment. |
| **Total** | | **799,919,882** | The instantiated model and independent estimator agree exactly; the model remains below one billion. |

The estimate is transparent and checked against the actual module graph. Run
`python scripts/validate_sub1b_blueprint.py` after changing dimensions or mixtures, then instantiate
on the PyTorch `meta` device to count exact tensors without allocating several GB:

```bash
python scripts/build_sub1b_student.py --device meta
python scripts/build_sub1b_student.py --tiny --device cpu --allow-full-memory
```

Both commands use [`docvlm_eval.student`](../../src/docvlm_eval/student), which implements the
vision tower, gated resampler, GQA decoder, causal multimodal loss, auxiliary heads, generation,
checkpoint round-trip, and selective initialization.

The all-attention language stack remains the default control. Set
`student.language.full_attention_layers` to a sorted zero-based subset to replace every other
layer with an LFM-style gated causal depthwise convolution. `conv_kernel_size` and `conv_bias`
remain explicit architecture controls. Attention layers use RoPE and compact GQA; convolution
layers retain a bounded recurrent state during generation. The four-arm fixed-FLOP comparison is
executable through
[`student_language_mixer_sweep.md`](student_language_mixer_sweep.md). No hybrid quality advantage
is assumed before that paired held-out experiment is complete.

Autoregressive generation encodes the image and prompt once, stores RoPE-applied keys and values at
the eight native GQA KV heads, and advances the decoder with one-token queries. The uncached
full-prefix path remains selectable as an explicit ablation. For the production RLVR shape
(2,520 visual patches, 256 prompt tokens, 128 completion tokens, group size eight), analytical
accounting estimates roughly 99% less rollout compute and a 161 MiB peak bf16 KV cache; completed
sequence policy/reference scoring remains unchanged.

The ViT's 4,096-position budget corresponds to a maximum square canvas of 64 by 64 patches, or
896 by 896 pixels at patch size 14. The input collator preserves aspect ratio, pads into that fixed
canvas, and supplies a pixel mask; padded patches are excluded from ViT self-attention, resampler
cross-attention, and pooled vision features.

### Why this split

The measured repository results show that small models often read a local value but fail grounding
and multi-region comparison. Spending almost the entire budget on the vision tower would improve
legibility but leave the integration bottleneck intact. Conversely, a tiny visual encoder would make
reasoning operate on lossy evidence. The default assigns roughly 11% to vision, 85% to language, and
4% to alignment and task heads, while the resampler exposes visual-token count as a direct latency
and accuracy control.

## Selective weight transfer

| Arm | Inherited weights | What it isolates | Main risk |
| --- | --- | --- | --- |
| `I0_random` | none | Whether the proposed corpus and objectives can create the capability | Highest data and compute demand |
| `I1_vision` | all compatible vision blocks | OCR and spatial sample efficiency | Natural-image features may discard tiny text |
| `I2_language` | all compatible LM blocks | Multilingual generation and reasoning priors | The decoder may guess from language priors |
| `I3_dual` | both towers; connector random | Standard warm-start ceiling | Weak evidence about what was actually learned |
| `I4_selective` | alternating depth-matched blocks | Minimum useful inherited representation | Width or depth mismatch needs projection/distillation |

Transfer is permitted only when tensor shape and semantic role match. Patch embeddings, attention
blocks, token embeddings, and output heads are treated separately. Vocabulary rows transfer by
exact token identity; unmatched rows remain random. A source with incompatible width is a teacher,
not a source of copied tensors. Connector weights remain random unless a controlled arm explicitly
tests connector transfer.

The implementation depth-maps a selected fraction of student blocks onto a deeper compatible
teacher and copies only exact-shape tensors. It includes canonical name adapters for native student,
Hugging Face SigLIP, and Llama-style checkpoints. Every run records copied keys, copied parameter
count, missing source keys, and shape mismatches. There is intentionally no hidden-width cropping or
interpolation: incompatible LFM or other teachers provide logits/features during distillation.

Example selective initialization:

```bash
python scripts/build_sub1b_student.py --tiny --device cpu --allow-full-memory \
  --init-arm I4_selective \
  --vision-source /path/to/vision/model.pt --vision-family siglip \
  --language-source /path/to/language/model.pt --language-family llama \
  --token-map /path/to/target_to_source_token_ids.json \
  --save /path/to/student_init
```

Native `model.pt`, PyTorch bin, single safetensors, and Hugging Face sharded checkpoint directories
are accepted. External token embeddings are never copied merely because their tensor shapes match.
`--token-map` must explicitly map each target vocabulary row to the same token's source row; without
that proof, the embedding and tied output head stay random.

The key comparison is not only final score. Report convergence tokens, held-out score, train versus
held-out gap, and robustness under counterfactual pixel edits. A transferred model that converges
quickly but relies more on priors is not automatically better.

Pinned source acquisition, zero-download shape analysis, fail-closed transfer validation, and the
five-arm matched suite are executable in
[`student_initialization_runner.md`](student_initialization_runner.md).

## Step 1: multimodal pretraining

Pretraining establishes perception and alignment before instruction behavior:

1. Start with clean local recognition, field boxes, reading order, and orientation.
2. Add dense pages, multilingual scripts, tables, charts, formulas, and long visual sequences.
3. Add acquisition degradation, distractors, relocated evidence, and cross-region questions.

These are executable stages rather than narrative labels. The full recipe advances loss stages by
the fraction of its effective-token budget consumed. Effective tokens are non-padding text tokens
plus the fixed resampled visual-prefix tokens for each image; raw image patches are reported
separately by architecture and are not mislabeled as language-sequence tokens. Step-fraction
curricula remain available for bounded ablations. Token-fraction sampler-weight changes are
rejected because worker prefetch could move their exact boundary; the full recipe uses
token-fraction loss changes and a fixed balanced sampler. The stage ID, progress, and every active
loss weight are logged. A curriculum and token-budget fingerprint in each checkpoint prevents
continuation under silently changed boundaries, units, or horizons. See
[`student_pretraining_runner.md`](student_pretraining_runner.md) for the schema and failure gates.

The default data mixture is 45% authored synthetic documents, 25% public document data, 15%
rendered text/formulas, 10% natural image-text replay, and 5% text-only replay. Synthetic data
provides exact boxes, table trees, chart values, formula source, and counterfactual pairs. Public
data constrains renderer bias. Replay protects general visual and language capability.

The autoregressive target is accompanied by region-text contrast, box regression, and orientation
losses. Same-tokenizer teacher KL and hidden-feature distillation are supported but remain zero
unless a native teacher checkpoint is explicitly configured. Losses are logged separately. This is
necessary because a low total loss can hide a connector or vision tower receiving almost no useful
gradient.

The native model exposes autoregressive, symmetric contrastive, four-way orientation, and normalized
box losses. Boxes are parameterized as start plus non-negative extent, so `x2 >= x1` and
`y2 >= y1` always hold; training combines smooth L1 with generalized IoU. The runner adds compressed
teacher KL and selected intermediate-feature alignment. Reading-order examples use answer-only
autoregressive supervision; there is no unimplemented standalone reading-order loss hidden inside
the weighted total.

LFM2.5-VL is connected through offline sequence distillation rather than invalid token-position
KL. Every image-question request is fingerprinted, generated resumably, scored with its native gold
metric, and accepted only above a configurable threshold. Accepted responses remain separate from
gold with model, confidence, score, and request provenance. A deterministic probability controls
how often pretraining consumes an accepted teacher response; rejected or absent responses fall
back to gold.

The UDD collator removes two subtle leakage/error paths. Box predictions pool the hidden state at
the end of the prompt, before any gold box tokens, and mixed QA views from the same image are
multi-positive pairs rather than false negatives in the contrastive objective.

## Step 2: post-training

### Grounded SFT

SFT teaches the response contract: concise extraction, evidence boxes, structured tables,
formula transcription, relational answers, calibrated abstention, and short evidence-linked
rationales. Rationales cite authored regions or cells. Free-form chain-of-thought is not a target
because fluent unsupported reasoning is particularly dangerous at this scale.

The executable runner keeps a strict `{"answer","evidence","rationale"}` JSON schema across all
three target ablations. It shuffles exhaustively per epoch rather than sampling curated SFT rows
with replacement. See
[`student_posttraining_runner.md`](student_posttraining_runner.md) for commands and contracts.

### RL with verifiable rewards

Run GRPO from the SFT checkpoint. All default rewards are computed from authored or normalized
ground truth:

- answer exactness and normalized text similarity;
- box IoU for evidence and spotting;
- tree similarity for tables;
- tolerance-aware numeric accuracy for charts;
- symbolic or normalized equivalence for formulas;
- rationale-to-evidence consistency;
- abstention utility on absent, unreadable, and contradictory inputs.

Final-answer and rationale rewards stay separate. Structural validity is a gate, not a bonus. Every
reward component is logged independently so held-out evaluation can expose reward hacking. The
runner combines KL to the frozen SFT policy with a periodic supervised multimodal replay loss.
The default applies one evidence-linked replay anchor every 20 rollout updates with coefficient
0.10. A separate replay JSONL can replace the active RLVR set, allowing general multimodal examples
to protect capabilities outside the reward dataset.

## Data construction for hard document reasoning

The generator should author a latent document graph before rendering:

```text
entities -> fields/cells/marks -> relations -> layout -> pixels
                                  |
                                  +-> questions, evidence, rationale, answer
```

This graph supports exact supervision for:

- tables: merged cells, nested headers, units, footnotes, cross-table joins;
- charts: source data, axis transforms, legends, uncertainty, interpolation, and visual decoys;
- formulas: source expression, rendered glyphs, equivalent forms, and symbol definitions;
- investment documents: period, currency, entity, metric, provenance, and derived ratios;
- multilingual pages: script mixing, bidirectional spans, vertical text, transliteration, and
  locale-specific numeric formats;
- scientific documents: figure-panel references, citations, equations, captions, and claims.

Difficulty is controlled by evidence count, spatial dispersion, distractor similarity, operation
depth, answer type, visual degradation, and whether the answer is absent. Train and held-out splits
must separate templates, fonts, value distributions, and document graphs, not merely random seeds.

## Required ablations

1. Cross initialization arm with data scale; compare sample efficiency and held-out robustness.
2. Sweep visual latent tokens and input resolution at a fixed total compute budget.
3. Remove each dense pretraining loss one at a time.
4. Compare answer-only, free rationale, and evidence-linked rationale SFT.
5. Compare SFT, correctness-only RLVR, and the full decomposed reward.
6. Evaluate every gain by document family, language, evidence count, and degradation.
7. Repeat the best recipe with LFM and at least one non-LFM teacher to detect teacher-specific gains.

The initialization-by-data-scale cross is compiled by
[`student_factorial_runner.md`](student_factorial_runner.md). The resolution-by-latent cross is
compiled with analytical runtime accounting and an overshoot gate by
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md). These suites must
still be executed before their effects can be claimed. The active native losses are isolated by
[`student_pretraining_loss_sweep.md`](student_pretraining_loss_sweep.md); online native-teacher
losses remain a separate checkpoint-dependent experiment. SFT target and RLVR reward effects are
isolated by [`student_posttraining_sweeps.md`](student_posttraining_sweeps.md), including a true
SFT-only evaluation path. Pinned LFM and non-LFM Qwen sequence targets are compared against
gold-only training at a fixed accepted-target dose by
[`student_sequence_teacher_sweep.md`](student_sequence_teacher_sweep.md). The current LoRA
ablations answer where
existing LFM capabilities are easiest to adapt. This blueprint answers the next question: which
capabilities can be built into a deployment-size student, what must be inherited, and what can be
learned from the controlled document curriculum.

## Current implementation boundary

The model constructor, selective initialization, tokenizer, UDD adapter, curriculum-aware samplers,
multimodal collator, same-tokenizer teacher interface, pretraining runner, structured SFT, strict
reward verifiers, supervised-replay-anchored single-update GRPO runner, and multi-split held-out
generation evaluator are executable and tested. The contracts are detailed in
[`student_input_pipeline.md`](student_input_pipeline.md) and
[`student_pretraining_runner.md`](student_pretraining_runner.md), and
[`student_posttraining_runner.md`](student_posttraining_runner.md). The next evidence-producing
step is to execute the compiled matched suites in
[`student_sweep_runner.md`](student_sweep_runner.md) at fixed token budgets and paired stochastic
replicates, including the data-scale cross in
[`student_factorial_runner.md`](student_factorial_runner.md) and the fixed-FLOP visual cross in
[`student_architecture_compute_sweep.md`](student_architecture_compute_sweep.md), then publish
held-out capability and efficiency curves with confidence intervals.
The language-mixer comparison in
[`student_language_mixer_sweep.md`](student_language_mixer_sweep.md) uses the same compute-budget
and paired-seed discipline for all-attention, alternating, and LFM-ratio decoders.

The current input path uses per-image packed patch sequences with stable two-dimensional visual
positions. Dense adaptive, aspect-bucketed, and fixed-square controls plus sequence-aware FLOP
accounting are executable through
[`student_visual_canvas_sweep.md`](student_visual_canvas_sweep.md). Packed numerical equivalence is
tested locally; compiled FlexAttention is attempted on supported CUDA systems with an explicit loop
fallback and resolved-backend logging. Target-GPU throughput and quality remain measured efficiency
ablations rather than claimed capabilities. Native RLVR is currently single-process. Formula reward
supports bounded elementary symbolic equivalence, but deliberately rejects unbounded calculus and
does not replace a full theorem prover.

The multi-objective pretraining losses also have an executable, trajectory-preserving conflict
probe. [`student_gradient_conflict_audit.md`](student_gradient_conflict_audit.md) measures weighted
loss cosines on shared-trunk anchors across paired replicates and requires material evidence before
PCGrad or GradNorm is added as an intervention.

Final deployment acceptance joins those two evidence streams. `eval_student.py` verifies the
benchmark's complete student configuration and canonical fingerprint, CUDA runtime, benchmark
dose, resolved backend, numerical delta, loop-relative median latency, and loop-relative peak
memory inside the same `gates.json` that evaluates held-out generalization and capability
retention. The same matched portrait/landscape patches also run through dense adaptive and fixed
square controls; packed deployment must outperform the adaptive control as well as its portable
loop. Three seed-deterministic order rotations produce same-round paired speed and memory ratios;
deployment requires both a median gain and no regressive round. Missing, legacy, or CPU-only
timing cannot pass; evidence from a different architecture fails. The production experiment makes
this gate a blocking first stage, so a fallback or efficiency regression cannot consume the much
larger pretraining budget.

## Evidence basis

The wider design-space review is maintained as a validated 100-method catalog:
[`frontier_method_survey.md`](frontier_method_survey.md). Its machine-readable source records the
benefit, sub-1B limitation, decision, and adjustable knobs for every method. The default blueprint
uses only entries marked `adopt`; entries marked `ablate` require matched evidence at the same
student size and compute budget.

The architecture follows the controlled connector, resolution, and data-mixture analysis in
[MM1](https://arxiv.org/abs/2403.09611), the compact visual-token findings in
[SmolVLM](https://arxiv.org/abs/2504.05299), and contrastive vision transfer established by
[CLIP](https://arxiv.org/abs/2103.00020). The rationale-distillation hypothesis follows
[Distilling Step-by-Step](https://arxiv.org/abs/2305.02301). These sources motivate design axes;
they do not establish that the same choices are optimal for document VLMs below one billion
parameters. The controlled ablations above are therefore part of the claim, not optional tuning.

# Sub-1B document VLM blueprint

## Decision

Build an approximately 790M-parameter student whose architecture, initialization, data mixture,
losses, and post-training rewards are controlled by
[`configs/sub1b_architecture.yaml`](../../configs/sub1b_architecture.yaml). LFM2.5-VL-1.6B remains
the fast experimental base and a teacher candidate; it is not the deployment-size student.

The first question is not whether transfer is allowed. It is **which inherited representations are
worth their dependency on a pretrained checkpoint**. Five initialization arms therefore span fully
random, vision-only, language-only, dual-tower, and alternating-block selective transfer. Every arm
uses the same architecture and token budget, then receives the same feature and output
distillation. This separates initialization benefit from teacher supervision.

## Architecture and parameter budget

| Component | Default | Estimated parameters | Reason |
| --- | --- | ---: | --- |
| Vision tower | ViT, 12 layers, width 768, patch 14 | 89M | Retains fine glyph detail without spending most of the budget on natural-image capacity. |
| Language decoder | 20 layers, width 1536, 64k vocabulary | 665M | Holds multilingual emission, cross-region binding, arithmetic, and structured generation. |
| Connector | Two-layer gated resampler, 64 visual tokens | 28M | Makes compression an explicit, ablatable bottleneck instead of a fixed projection. |
| Temporary task heads | region-text contrast, orientation | 8M | Supplies dense pretraining signals; removable for deployment. |
| **Total** | | **about 790M** | Leaves headroom below the strict one-billion deployment limit. |

The estimate is deliberately transparent, not a substitute for counting a constructed model. Run
`python scripts/validate_sub1b_blueprint.py` after changing dimensions or mixtures. Once model
construction exists, exact instantiated parameters must replace the estimate as the release gate.

### Why this split

The measured repository results show that small models often read a local value but fail grounding
and multi-region comparison. Spending almost the entire budget on the vision tower would improve
legibility but leave the integration bottleneck intact. Conversely, a tiny visual encoder would make
reasoning operate on lossy evidence. The default assigns roughly 11% to vision, 84% to language, and
5% to alignment and task heads, while the resampler exposes visual-token count as a direct latency
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

The key comparison is not only final score. Report convergence tokens, held-out score, train versus
held-out gap, and robustness under counterfactual pixel edits. A transferred model that converges
quickly but relies more on priors is not automatically better.

## Step 1: multimodal pretraining

Pretraining establishes perception and alignment before instruction behavior:

1. Start with clean local recognition, field boxes, reading order, and orientation.
2. Add dense pages, multilingual scripts, tables, charts, formulas, and long visual sequences.
3. Add acquisition degradation, distractors, relocated evidence, and cross-region questions.

The default data mixture is 45% authored synthetic documents, 25% public document data, 15%
rendered text/formulas, 10% natural image-text replay, and 5% text-only replay. Synthetic data
provides exact boxes, table trees, chart values, formula source, and counterfactual pairs. Public
data constrains renderer bias. Replay protects general visual and language capability.

The autoregressive target is accompanied by teacher KL, hidden-feature distillation,
region-text contrast, box regression, reading-order, and orientation losses. Losses are logged
separately and gradient norms are measured per module. This is necessary because a low total loss
can hide a connector or vision tower receiving almost no useful gradient.

## Step 2: post-training

### Grounded SFT

SFT teaches the response contract: concise extraction, evidence boxes, structured tables,
formula transcription, relational answers, calibrated abstention, and short evidence-linked
rationales. Rationales cite authored regions or cells. Free-form chain-of-thought is not a target
because fluent unsupported reasoning is particularly dangerous at this scale.

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
reward is also reported alone on held-out templates to expose reward hacking. KL to the SFT policy
and general multimodal replay constrain capability collapse.

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

The current LoRA ablations answer where existing LFM capabilities are easiest to adapt. This
blueprint answers the next question: which capabilities can be built into a deployment-size student,
what must be inherited, and what can be learned from the controlled document curriculum.

## Evidence basis

The architecture follows the controlled connector, resolution, and data-mixture analysis in
[MM1](https://arxiv.org/abs/2403.09611), the compact visual-token findings in
[SmolVLM](https://arxiv.org/abs/2504.05299), and contrastive vision transfer established by
[CLIP](https://arxiv.org/abs/2103.00020). The rationale-distillation hypothesis follows
[Distilling Step-by-Step](https://arxiv.org/abs/2305.02301). These sources motivate design axes;
they do not establish that the same choices are optimal for document VLMs below one billion
parameters. The controlled ablations above are therefore part of the claim, not optional tuning.

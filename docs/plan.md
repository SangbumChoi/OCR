# Project plan — from "OCR" to **document + UX understanding**

This is the north-star narrative for the whole repo: *why* each piece exists and *in what order*
to read it. Every phase links the document(s) that carry out that step. Treat this as the table of
contents for the thinking, not a duplicate of it.

> **Convention — English only.** Every document in this repo (all Markdown/reports under `docs/`
> and elsewhere, plus code comments and docstrings) is written in **English**. No mixed-language
> glosses. This keeps the project consistent and reviewable for any reader.

---

## 0. North star — redefine the target before touching a model

The task says "OCR", but the goal we are actually after is **document understanding**, and we
deliberately define "document" *broadly*:

- not just printed/scanned text (invoice, paper, receipt), but the **whole family of text-bearing
  artefacts** — IDs, cheques, prescriptions, webtoons, ancient manuscripts, LCD meters, …
- and, as a modern extension, **screens are documents too**: a website or a phone app is a
  text-bearing surface whose layout, reading order and affordances must be parsed. Reading them is
  a form of **UX understanding**. We want to aim *toward* this, not away from it.

So the premise is: **a "document" is any surface a person reads to extract meaning** — paper or
pixels. The rest of the plan follows from that one decision.

> The data-first conviction: **surveying the data/evaluations matters more than surveying models.**
> Models churn; the question of *what we must be able to read and how we'd know we succeeded* is the
> durable part. Hence phases 1–3 are all about data before phase 4 touches a model.

```
0 define "document" (incl. UX/screens)
        │
1 survey document VARIATIONS ─────────────►  report/document_type_taxonomy.md
        │
2 survey EVALUATIONS (data > models) ──────►  report/benchmark_taxonomy.md · benchmark_patterns.md
        │
3 understand evals → hunt SPECIAL CASES ───►  report/capability_axes.md (all capability axes)
        │                                      data/benchmarks/{custom_eval,oov_probe,realistic_cases}
4 evaluate ALL models × ALL metrics ───────►  report/results_analysis.md · insights.md · results/*
   (+ study model properties)                 report/technical_report.md (Appendix profiles)
        │
5 decide ARCHITECTURE: integrate, don't ───►  report/research_novelty.md · ablation_plan.md
   pipeline  (model + UDD input implemented)    sub1b_architecture_blueprint.md
        │
6 build/collect data → ablate → COMBINE ───►  scripts/make_realistic_cases.py · ablation_plan.md
   → spotting for human-in-the-loop verify     report/technical_report.md §Part 2.1b/1c
```

---

## 1. Survey the document **variations**

Under the broad definition, first map *what kinds of documents exist* and — crucially — what
**stressors** each imposes (layout, reading direction, language/script, degradation, non-text
classes, handwriting, hallucination risk, spotting need). A type is only as hard as the stressors
it combines, which turns an open-ended list into concrete evaluation requirements.

- **Read:** [`report/document_type_taxonomy.md`](report/document_type_taxonomy.md) — the
  type × stressor matrix (invoice → webtoon → ID → ancient → LCD → **UI/screen**), with deep dives
  on the four hardest families and a gap checklist.

## 2. Survey the **evaluations** — data before models

Because *what/how we measure* outlives any model, we catalogue the OCR/document benchmark
landscape: every category (full-page recognition, scene text, VQA, KIE, tables, charts, formulas,
end-to-end parsing) and the metric each uses (CER/WER/NED, ANLS, relaxed-acc, TEDS, OCRBench,
grounding IoU, calibration), then group benchmarks by *nature* and draw a priority map.

- **Read:** [`report/benchmark_taxonomy.md`](report/benchmark_taxonomy.md) — all benchmark *types*
  and *metrics*.
- **Read:** [`report/benchmark_patterns.md`](report/benchmark_patterns.md) — what each benchmark
  collects, visual-class diversity beyond text, VQA answer-natures (exact vs list/ANLS), grouping +
  priority graph.
- **Browse:** [`../data/benchmarks/README.md`](../data/benchmarks/README.md) — one inspectable
  sample (image + GT + metric note) per benchmark.

## 3. Understand the evals → hunt for **special cases & gaps**

With the evaluation space understood, ask: *what's missing? what totally special cases did the
field skip?* This is where we separate "reads the pixels" from "guesses from priors", isolate the
capability axes, and build controlled probes + our own evaluation format for the gaps.

- **Read:** [`report/capability_axes.md`](report/capability_axes.md) — the full capability-axis
  catalogue: text vs location, content reasoning, chart-value, spotting fair-comparison, **and**
  spatial & context understanding (falsifiable control-pair probes: counterfactual / distractor /
  position-bias).
- **Build/browse the gap sets:**
  [`../data/probes/custom_eval/`](../data/probes/custom_eval/README.md) (our proposed
  per-content-class format),
  [`../data/probes/realistic_cases/`](../data/probes/realistic_cases/README.md) (realistic
  *and* GT-exact special cases incl. the UX surfaces), and the OOV/web-UI probes.
- **Read (forward pointer):** [`report/research_novelty.md`](report/research_novelty.md) — the
  literature-grounded gaps these probes can uniquely test at ≤1B.

## 4. Evaluate **all models × all metrics** on the mockup evals (+ study the models)

Only now do we touch models: run every sub-1B candidate on the mockup/probe evaluations under
*every applicable metric*, read the results for flaws, and — in parallel — study each model's
properties (parameters, pretraining data, architecture) so the numbers have a *why*.

- **Run:** [`../scripts/run_matrix.py`](../scripts/run_matrix.py) (and
  [`build_realistic_benchmark.py`](../scripts/build_realistic_benchmark.py) to turn the synth GT
  into a benchmark) → matrices under [`results/`](results/).
- **Read:** [`report/results_analysis.md`](report/results_analysis.md) — measured GPU sweep (19
  models on a T4), inference bugs fixed, scoring flaws, per-model capability vectors.
- **Read:** [`report/insights.md`](report/insights.md) — auto cross-model insights (capability
  leaders, reasoning-emergence, grounding gap, efficiency frontier, OOV fallback).
- **Read:** [`report/technical_report.md`](report/technical_report.md) — the synthesis + Appendix
  model profiles.

## 5. Decide the **architecture**: integrate capabilities, don't pipeline

The functions we want (spotting, orientation, layout, then OCR, then re-extract) *could* be a
**pipeline** of specialist models — orientation model → layout model → OCR → re-read. We surveyed
that "semi-general-purpose" route, but a pipeline is **not actually general-purpose** (brittle
hand-offs, error compounding, no joint reasoning). The conclusion: **bake these capabilities *into*
the model** via targeted fine-tuning, rather than bolting on stages.

- **Decision:** prefer in-model capability injection over a model pipeline.
- **Implemented in the native student:** an explicit four-way **orientation head**; the UDD
  collator supplies controlled rotation labels and transforms evidence boxes with the image.
- **Read:** [`report/research_novelty.md`](report/research_novelty.md) — module-placement ×
  capability × scale, grounding-supervision causality (does "where" improve "whether"?).
- **Build:** [`report/sub1b_architecture_blueprint.md`](report/sub1b_architecture_blueprint.md) —
  the adjustable approximately 800M student, executable native model, selective-transfer controls,
  multimodal pretraining, grounded SFT, and verifiable-reward RL design. Its machine-readable source is
  [`../configs/sub1b_architecture.yaml`](../configs/sub1b_architecture.yaml).
- **Input:** [`report/student_input_pipeline.md`](report/student_input_pipeline.md) — the executable
  UDD QA/grounding adapter, new multilingual tokenizer, balanced sampler, prompt-masked collator,
  rotation/box transforms, and visual padding-mask contract.
- **Acquire data:** [`report/student_data_acquisition.md`](report/student_data_acquisition.md) —
  immutable Hub revisions, deterministic fold/filter sampling, schema and image validation,
  duplicate gates, and end-to-end component provenance.
- **Pretrain:** [`report/student_pretraining_runner.md`](report/student_pretraining_runner.md) —
  same-tokenizer online distillation, token-based learning-rate scheduling, deterministic
  sampler/loss curricula, mixed precision, `torchrun`, sample-weighted held-out evaluation, and
  exact checkpoint resume.
- **Fit optimizer state on the target GPU:**
  [`report/student_optimizer_memory.md`](report/student_optimizer_memory.md) — one fail-closed
  optimizer contract across pretraining and post-training, exact resume identity, and measured
  state/peak-memory deployment evidence.
- **Adapt the data mixture:**
  [`report/student_adaptive_mixture.md`](report/student_adaptive_mixture.md) — a separate
  optimizer-heldout validation split, EMA validation-loss feedback, epoch-boundary probability
  updates, exact resume, and a three-arm by three-seed matched sweep.
- **Schedule document composition:**
  [`report/student_composition_curriculum.md`](report/student_composition_curriculum.md) — exact
  page/document count propagation and a task-preserving secondary sampler that progresses from
  single pages through multi-page packets to cross-document dossiers, with a paired static-versus-
  staged sweep.
- **Close the validation-to-data loop:**
  [`report/student_failure_driven_synthesis.md`](report/student_failure_driven_synthesis.md) —
  leakage-safe matched baseline-to-final learning progress, residual-failure factor shrinkage
  across case, language, difficulty, layout, and composition arms, exact generator execution, and
  a content-addressed next-run plan.
- **Isolate pretraining losses:**
  [`report/student_pretraining_loss_sweep.md`](report/student_pretraining_loss_sweep.md) —
  fail-closed supervision provenance and paired leave-one-active-loss-out experiments.
- **Choose the contrastive objective:**
  [`report/student_contrastive_objective_sweep.md`](report/student_contrastive_objective_sweep.md) —
  paired fixed-compute SigLIP versus multi-positive softmax alignment.
- **Supply real contrastive negatives:**
  [`report/student_contrastive_memory.md`](report/student_contrastive_memory.md) — stable
  same-image IDs, an exact-resumable per-rank FIFO for one-image microbatches, explicit negative
  telemetry, compute accounting, and a two-arm by three-seed matched sweep.
- **Choose the connector family:**
  [`report/student_connector_family_sweep.md`](report/student_connector_family_sweep.md) —
  compute-matched gated attention pooling versus an ordered average-pool projector.
- **Calibrate deployment confidence:**
  [`report/student_temperature_calibration.md`](report/student_temperature_calibration.md) —
  leakage-safe heldout partitioning, scalar temperature fitting, paired ECE logging, and gates.
- **Test teacher dependence:**
  [`report/student_sequence_teacher_sweep.md`](report/student_sequence_teacher_sweep.md) — pinned
  LFM/Qwen versus gold-only supervision at a fixed request and accepted-target dose.
- **Transfer document relations:**
  [`report/student_token_relation_distillation_sweep.md`](report/student_token_relation_distillation_sweep.md)
  — same-native-teacher pointwise hidden anchors versus bounded token-relation KL at a fixed
  representation-loss weight.
- **Initialize:** [`report/student_initialization_runner.md`](report/student_initialization_runner.md)
  — pinned pretrained source acquisition, zero-download shape compatibility, fail-closed selective
  transfer, joint-salience SwiGLU reduction, initialization lineage, and matched baseline and
  structured-transfer suites.
- **Preflight transfer across architectures:**
  [`report/small_vlm_architecture_commonality.md`](report/small_vlm_architecture_commonality.md) —
  pinned small-VLM feature commonality and component-level copy-versus-distill decisions.
- **Measure real-weight commonality:**
  [`report/small_vlm_weight_commonality.md`](report/small_vlm_weight_commonality.md) —
  bounded safetensors range sketches across five pinned models, recurrent operator statistics,
  and a no-basis-assumption transfer contract.
- **Select transfer sources from combined evidence:**
  [`report/selective_transfer_source_matrix.md`](report/selective_transfer_source_matrix.md) —
  component-level copy, structured-transfer, identity-map, payload-preflight, and distillation
  decisions composed from pinned topology, sampled real-weight health, and executed payload
  evidence without treating population statistics as neuron-basis alignment.
- **Test LFM operator alignment:**
  [`report/student_lfm_language_transfer_sweep.md`](report/student_lfm_language_transfer_sweep.md)
  — a sub-1B LFM-compatible language control, zero-allocation transfer proof, and paired
  architecture and aligned-transfer sweep.
- **Test cross-model vision-block transfer:**
  [`report/student_smol_vision_transfer_pilot.md`](report/student_smol_vision_transfer_pilot.md)
  — exact SmolVLM2 transformer-block acquisition, real-payload copy verification, and a bounded
  matched pilot against the same LFM language initialization.
- **Guard long structured generation and rendering:**
  [`report/student_generation_rendering_safeguards.md`](report/student_generation_rendering_safeguards.md)
  — exact suffix-cycle termination, bounded task-aware token horizons, a leakage-safe tokenized
  target-fit gate, compact token-limit telemetry, all-page expansion, table-cell survival, and
  bounded-canvas checks.
- **Test structured initialization:**
  [`report/student_structured_mlp_transfer_sweep.md`](report/student_structured_mlp_transfer_sweep.md)
  — exact-shape selective transfer versus a shared-channel reduction of wider teacher MLPs.
- **Test source-aligned attention:**
  [`report/student_attention_geometry_transfer_factorial.md`](report/student_attention_geometry_transfer_factorial.md)
  — strict semantic transfer gates and a paired 2x2 geometry-by-transfer interaction.
- **Test the spotting module interaction:**
  [`report/student_lora_placement_interaction.md`](report/student_lora_placement_interaction.md)
  — an exact vision+connector target union, fail-closed LoRA parameter matching, and a paired
  three-seed confirmatory sweep on the LFM spotting arm. Its compact authenticated W&B snapshot
  audit rejects crashed runs and finished runs without heldout metrics, and limits the observed
  single-pair vision advantage to direction-only evidence.
- **Post-train:** [`report/student_posttraining_runner.md`](report/student_posttraining_runner.md) —
  exhaustive structured SFT, strict answer/evidence/rationale outputs, decomposed verifiable
  rewards, visual-prefix-cached rollout, verifier-ranked DPO/IPO, optional preference-to-GRPO
  sequencing with an immutable SFT reference, periodic supervised multimodal replay, bounded
  symbolic formula equivalence, exact checkpoint resume, and train/heldout generation evaluation.
- **Isolate post-training choices:**
  [`report/student_posttraining_sweeps.md`](report/student_posttraining_sweeps.md) — paired
  SFT-target, reward, advantage-estimator, compute-matched DPO-versus-GRPO, and matched
  DPO-versus-IPO and gold-anchor-versus-reference-only preference-source comparisons.
- **Run end to end:** [`report/student_experiment_runner.md`](report/student_experiment_runner.md) —
  one adjustable, validated, resumable DAG from independent hard-document synthesis through
  weighted UDD mixing, pretraining, SFT, optional preference optimization, RLVR, matched initial-versus-final
  train/validation/heldout evaluation, and next-batch synthesis planning.
- **Continue failure-driven rounds:**
  [`report/student_curriculum_runner.md`](report/student_curriculum_runner.md) — full-hash parent
  attestation, exact model/tokenizer preservation, validation-authorized generation, deterministic
  origin-stratified cumulative replay, and post-training-only continuation with an explicit
  optimizer reset policy.
- **Route verifier failures into data generation:**
  [`report/student_failure_driven_synthesis.md`](report/student_failure_driven_synthesis.md) —
  matched component-level reward deficits and progress drive only compatible exact generator
  families, while validation authorization and heldout isolation remain fail-closed.
- **Gate long structured generation:**
  [`report/student_generation_rendering_safeguards.md`](report/student_generation_rendering_safeguards.md)
  — matched candidate/reference token policies must preserve table, HTML, full-page, transcription,
  reading-order, long-context, and markup quality without repetition or max-token regressions.
- **Attest execution evidence:**
  [`report/student_experiment_evidence.md`](report/student_experiment_evidence.md) — deterministic
  source/stage/artifact/checkpoint hashes, semantic optimization and evaluation checks,
  resume-safe cumulative stage state, runtime parameter/trainability/deployment-count attestation
  at every native checkpoint, and a strict separation between pipeline execution and
  deployment-capability claims. Blueprint estimates alone cannot prove the sub-1B deployment
  contract.
- **Audit the complete objective:**
  [`report/end_to_end_goal_readiness.md`](report/end_to_end_goal_readiness.md) — requirement-level
  separation of implemented capability, sealed target-GPU execution, heldout quality, and
  multi-seed promotion evidence.
- **Run matched ablations:** [`report/student_sweep_runner.md`](report/student_sweep_runner.md) —
  RFC 6902 experiment/blueprint variants, explicit fixed-control gates, paired seed blocks,
  deterministic bootstrap intervals, independent resumable runs, W&B grouping, and baseline-delta
  train/heldout comparison, followed by multiplicity-corrected promotion and provenance-pinned
  canonical recipe materialization.
- **Audit robustness slices:**
  [`report/student_robustness_slices.md`](report/student_robustness_slices.md) — canonical
  document-family, language, evidence-count, and degradation metadata, matched train/heldout
  panels, and paired per-slice ablation intervals.
- **Match architecture compute:**
  [`report/student_architecture_compute_sweep.md`](report/student_architecture_compute_sweep.md) —
  analytical dense student FLOPs, compute-driven schedules and stopping, five
  resolution-by-latent profiles, paired seeds, and realized-budget overshoot gates.
- **Test LFM-style language mixers:**
  [`report/student_language_mixer_sweep.md`](report/student_language_mixer_sweep.md) —
  adjustable full-attention indices, gated causal short-convolution layers, hybrid generation
  state, exact parameter/cache accounting, and four compute-matched paired profiles.
- **Remove visual padding waste:**
  [`report/student_visual_canvas_sweep.md`](report/student_visual_canvas_sweep.md) —
  per-image packed patch sequences, stable two-dimensional positions, sequence-aware FLOPs, and
  paired dense adaptive/aspect-bucketed/fixed-square controls.
- **Measure sample efficiency:** [`report/student_factorial_runner.md`](report/student_factorial_runner.md)
  — fixed-heldout initialization-by-data-scale experiments, actual-row provenance, paired
  difference-in-differences, and capability-axis interactions.
- **Survey:** [`report/frontier_method_survey.md`](report/frontier_method_survey.md) — a validated
  catalog of 100 primary methods across vision, connectors, document models, compact LMs,
  transfer/distillation, data construction, RL, and reliability. Every entry records both the
  useful mechanism and the failure risk in the sub-1B document regime.
- **Trace adopted methods to executable evidence:**
  [`report/student_method_evidence.md`](report/student_method_evidence.md) — exact knob coverage,
  live implementation and test anchors, compact fingerprints, and a fail-closed pre-allocation
  experiment stage.
- **Audit multi-loss interference:**
  [`report/student_gradient_conflict_audit.md`](report/student_gradient_conflict_audit.md) —
  trajectory-preserving shared-trunk gradient telemetry, a three-arm by three-replicate anchor audit,
  and a falsifiable gate before PCGrad or GradNorm is introduced.
- **Read:** [`report/ablation_plan.md`](report/ablation_plan.md) — A1 spotting,
  A7 preprocessing, A5 LoRA placement are exactly the "fold the pipeline into the model" moves.

## 6. Build/collect data → ablate → **combine** → the staircase report

Finally, either **synthesize the fine-tuning data we want** (GT for free) or bring in real data,
apply the planned ablations one factor at a time, analyse each in isolation, then **stack the
winners** cumulatively. The deliverable is a report whose headline figure is a **staircase**: each
added factor lifting the score a step, ending well above the baseline.

- **Make data (GT built-in):**
  [`../scripts/make_realistic_cases.py`](../scripts/make_realistic_cases.py) `--config
  configs/synth_data.yaml [--ablation <id>] --count N` → large, label-exact training set whose GT
  carries every ablation factor (see [`report/synthetic_data_dto.md`](report/synthetic_data_dto.md)
  for the DTO + config-driven factor control, and
  [`../data/probes/realistic_cases/README.md`](../data/probes/realistic_cases/README.md));
  diversity is driven by [`report/prd_synthetic_diversity.md`](report/prd_synthetic_diversity.md)
  and the open-source technique survey [`report/synth_generation_survey.md`](report/synth_generation_survey.md)
  (simulation-only; LLM generators kept as future-optional seams).
- **Generate hard reasoning:** [`report/hard_synthetic_pipeline.md`](report/hard_synthetic_pipeline.md)
  documents the executable latent graph, five-level curriculum, hard table/chart/investment/science
  families, exact five-language locale projection, multi-box structured SFT/RLVR targets, and
  semantic train/heldout leakage gate. Required spatial labels use occurrence-aware color-probe
  recovery when a complex-script or wrapped span is absent from the PDF text search results.
  Photo-style cases can apply a deterministic same-frame perspective homography to pixels and all
  boxes, with paired hard documents sharing geometry and explicit off/on ablation arms.
  Each hard family also has three semantic-preserving structural layouts; separate visual
  fingerprints support layout-grouped splits and a strict cross-split layout-isolation gate while
  paired hard documents share their selected layout.
  Evidence-safe handwriting, stamp, and seal marks add grounded recognition targets before
  perspective and degradation; their explicit metadata forms a matched robustness slice.
  A three-page audit packet supplies exact cross-page reconciliation, with configurable vertical
  or compute-aware grid composition and page-count robustness reporting.
  A three-source investment dossier adds exact cross-document valuation, claim verification,
  source reliability, and next-action supervision. Document IDs, origins, and evidence ownership
  survive the one-image student input path and form a document-count robustness slice.
  A programmatic parallel-process diagram links every visible arrow rate to executable path
  products, parallel-path aggregation, topology lookup, and expected-count targets across three
  semantic-preserving layouts and five difficulty levels.
  Scientific result pages ground every mean and standard error separately, generate unambiguous
  confidence-interval and precision questions, and balance supported versus unsupported
  pooled-standard-error significance claims. Their interval and decision programs are independently
  re-executed by production RLVR rather than accepted from authored rationale text.
  The same latent results render a captioned quantitative figure and a separately grounded Results
  claim. Correct and incorrect manuscript claims are balanced, and a five-evidence program
  recomputes figure-table-claim consistency for a dedicated held-out and RLVR slice.
  Every executable graph query now preserves a fingerprinted reasoning trace through QA and sample
  conversion. Production RLVR independently re-executes that trace and combines evidence overlap,
  semantic rationale similarity, required-number recall, and hallucinated-number precision; a
  three-seed semantic-only verifier arm provides the matched control.
  Clean and degraded rasters are fail-closed on local evidence visibility; degraded copies also
  require clean-crop structure retention under bounded deterministic retries.
- **Ablate & combine:** [`report/ablation_plan.md`](report/ablation_plan.md)
  (A1–A7, `integration_order`) → [`../scripts/plot_ablation.py`](../scripts/plot_ablation.py).
- **Deliverable (updated):** the report is now **refocused on the single selected objective —
  spotting / grounding (L1) for human-in-the-loop verification** (see
  [`report/technical_report.md`](report/technical_report.md) §Part 2.1b/1c). The earlier
  **cumulative "staircase" is deprecated** as the headline; the deliverable is the before/after on
  **held-out spot-IoU** (the `plot_ablation.py` staircase stays in-repo for optional later use).

---

## Decisions & open TODOs

| Decision | Where |
| --- | --- |
| "Document" includes screens / UX surfaces (not just OCR) | this plan §0; realistic_cases `website`/`mobile_app` |
| Data/evaluation survey before model survey | §1–3 before §4 |
| Integrate capabilities into the model, **not** a specialist pipeline | §5; ablation_plan A1/A5/A7 |
| Orientation as an explicit four-way auxiliary signal | §5; native student model |
| Report's headline = **spotting/grounding for human-in-the-loop verification** (cumulative staircase **deprecated**) | §6; technical_report §Part 2.1b/1c |

## Reading order (index)

1. [`report/document_type_taxonomy.md`](report/document_type_taxonomy.md)
2. [`report/benchmark_taxonomy.md`](report/benchmark_taxonomy.md) ·
   [`report/benchmark_patterns.md`](report/benchmark_patterns.md)
3. [`report/capability_axes.md`](report/capability_axes.md) ·
   [`../data/probes/realistic_cases/README.md`](../data/probes/realistic_cases/README.md)
4. [`report/results_analysis.md`](report/results_analysis.md) ·
   [`report/insights.md`](report/insights.md) · [`results/comparison_table.md`](results/comparison_table.md)
5. [`report/research_novelty.md`](report/research_novelty.md) ·
   [`report/sub1b_architecture_blueprint.md`](report/sub1b_architecture_blueprint.md) ·
   [`report/student_data_acquisition.md`](report/student_data_acquisition.md) ·
   [`report/student_initialization_runner.md`](report/student_initialization_runner.md) ·
   [`report/small_vlm_weight_commonality.md`](report/small_vlm_weight_commonality.md) ·
   [`report/student_pretraining_runner.md`](report/student_pretraining_runner.md) ·
   [`report/student_gradient_conflict_audit.md`](report/student_gradient_conflict_audit.md) ·
   [`report/student_adaptive_mixture.md`](report/student_adaptive_mixture.md) ·
   [`report/student_composition_curriculum.md`](report/student_composition_curriculum.md) ·
   [`report/student_pretraining_loss_sweep.md`](report/student_pretraining_loss_sweep.md) ·
   [`report/student_contrastive_objective_sweep.md`](report/student_contrastive_objective_sweep.md) ·
   [`report/student_connector_family_sweep.md`](report/student_connector_family_sweep.md) ·
   [`report/student_temperature_calibration.md`](report/student_temperature_calibration.md) ·
   [`report/student_box_iou_loss_sweep.md`](report/student_box_iou_loss_sweep.md) ·
   [`report/student_sequence_teacher_sweep.md`](report/student_sequence_teacher_sweep.md) ·
   [`report/student_token_relation_distillation_sweep.md`](report/student_token_relation_distillation_sweep.md) ·
   [`report/student_structured_mlp_transfer_sweep.md`](report/student_structured_mlp_transfer_sweep.md) ·
   [`report/student_attention_geometry_transfer_factorial.md`](report/student_attention_geometry_transfer_factorial.md) ·
   [`report/student_lfm_language_transfer_sweep.md`](report/student_lfm_language_transfer_sweep.md) ·
   [`report/student_smol_vision_transfer_pilot.md`](report/student_smol_vision_transfer_pilot.md) ·
   [`report/student_lora_placement_interaction.md`](report/student_lora_placement_interaction.md) ·
   [`report/student_posttraining_runner.md`](report/student_posttraining_runner.md) ·
   [`report/student_posttraining_sweeps.md`](report/student_posttraining_sweeps.md) ·
   [`report/student_rlvr_advantage_sweep.md`](report/student_rlvr_advantage_sweep.md) ·
   [`report/student_preference_method_sweep.md`](report/student_preference_method_sweep.md) ·
   [`report/student_experiment_runner.md`](report/student_experiment_runner.md) ·
   [`report/student_architecture_compute_sweep.md`](report/student_architecture_compute_sweep.md) ·
   [`report/student_language_mixer_sweep.md`](report/student_language_mixer_sweep.md) ·
   [`report/student_visual_canvas_sweep.md`](report/student_visual_canvas_sweep.md) ·
   [`report/student_factorial_runner.md`](report/student_factorial_runner.md) ·
   [`report/frontier_method_survey.md`](report/frontier_method_survey.md) ·
   [`report/student_method_evidence.md`](report/student_method_evidence.md)
6. [`report/ablation_plan.md`](report/ablation_plan.md) ·
   [`report/prd_synthetic_diversity.md`](report/prd_synthetic_diversity.md) ·
   [`report/hard_synthetic_pipeline.md`](report/hard_synthetic_pipeline.md) ·
   [`report/synth_generation_survey.md`](report/synth_generation_survey.md) →
   [`report/technical_report.md`](report/technical_report.md)

# Research novelty & open questions — what's unexplored, and what *this repo* can test

A literature scan (2024–2026) of small/sub-1B document VLMs, their evaluation, adaptation, and
tokenizer/OOV behaviour. The recurring finding: **almost every conclusion was established at
2B–72B and changes one knob confounded with others** (compression *and* size *and* data *and*
resolution move together). The capacity-constrained **≤1B regime may invert those conclusions**,
and our **controlled synthetic probes + matched single-factor ablations** (already built in this
repo) are unusually well-suited to isolate them. Below, each direction lists *what's known
(cited)* → *the gap* → *our testable hypothesis* → *the repo asset that tests it*.

> Verification note: claims carry arXiv IDs + dates from the survey; 2026-dated IDs (`2601.*`…)
> are lightly verified — treat as weak evidence. "No study found" = targeted search returned only
> adjacent work.

## The biggest void: module-placement × capability × scale at ≤1B

- **Known.** LLM-side LoRA (frozen vision+connector) is the default; "the LLM unlocks multimodal
  skill" is folklore from ≥7B (emergentmind survey). Depth/module asymmetry exists for *text*
  LLMs only — MoLA "Higher Layers Need More LoRA Experts" (arXiv:2402.08562, 2024-02): attention
  concentrates on few experts, MLP uses them evenly. LangVision-LoRA-NAS (arXiv:2508.12512) finds
  non-uniform optimal rank across VLM layers.
- **Gap.** *No* paper crosses {vision enc, connector, LLM-attn, LLM-MLP} × {grounding, integrative
  reasoning, language/CJK} with a controlled per-module ablation on a **sub-1B document** model.
- **Hypothesis (ours) — capability is *module-localised*: fine-tune the module whose job the
  capability most resembles.** The intuition is mechanistic — *where* a capability physically lives
  in the encoder→connector→LLM stack tells you *where* to spend adapter capacity. So the best place
  to fine-tune **differs by capability**; "adapt the LLM" is not a universal answer.

  | Capability (probe axis) | Adapt this module | Why — the intuition | Ablation |
  | --- | --- | --- | --- |
  | spatial grounding, region & relative position (L1–L4) | **vision encoder** (+ connector) | "where" is a geometric/perceptual property formed in pixel space *before* language; boxes are a vision-tower output | A1, A5(vision) |
  | small-text / dense recognition (T1) | **vision encoder** + input resolution | legibility is an encoder + resolution property, not a language one | A7, A5(vision) |
  | reading order / layout serialization | **connector / projector** | the projector decides how 2-D visual tokens are flattened into the 1-D LM sequence — it *is* the ordering step | A5(connector) |
  | language diversity / OOV / unseen script | **LLM** (MLP + token embeddings) | genuinely new vocabulary must be *stored* in the LM and *emitted* by the decoder — the encoder can see a glyph it can't name | A4, A5(llm-mlp) |
  | integrative & numeric reasoning (H1–H3) | **LLM** (attention + MLP) | multi-region binding (attn) + arithmetic/compute (MLP) is a language-model computation over already-read values | A2, A5(llm-attn/mlp) |
  | structured emission / abstention (output side) | **LLM** (decoder) | when the model must *produce* structure (DocTags/JSON) or say "not present", that is a decoder generation behaviour | A2, A5(llm) |

  Corollary: at 1B the **connector may be the bottleneck** (too little capacity to both align and
  serialise), so adapting it could matter *more* than at ≥7B — **inverting** the LLM-centric folklore.
  These per-capability bets are exactly the arms of the [**Ablation plan**](ablation_plan.md):
  **A5** places LoRA per module, while **A1/A2/A4/A7** supply the matching supervision and inputs.
- **Repo asset.** Ablation **A5** (`configs/ablations.yaml`) × the capability probe + custom_eval
  per-axis. Single-module LoRA at matched trainable-param budget → `plot_ablation.py` placement bars.

## Grounding/spotting supervision: does "where" improve "whether"? (contested)

- **Known.** BBox-DocVQA (arXiv:2511.15090, 2025-11) — GT evidence regions raise answer accuracy
  1.7–12.4 pts, **smaller models gain more**; "correct answers don't guarantee correct reasoning
  paths". *But* "Does Object Grounding Really Reduce Hallucination?" (arXiv:2406.14492, 2024-06)
  finds grounding objectives have **near-zero** hallucination effect (natural images). SMuDGE /
  "Where is this coming from?" (arXiv:2503.19120, NAACL'25): ANLS/EM score grounded and
  hallucinated answers *identically*.
- **Gap.** Whether adding **box targets during fine-tuning** causally improves *non-localization*
  extraction at **sub-1B**, and whether the benefit is IID or OOD, is untested; the grounding→
  honesty link is unresolved in documents.
- **Hypothesis (ours).** Auxiliary "where" targets improve value-F1 with **no box at test time**,
  the gain grows with evidence-region count/dispersion, and concentrates in **OOD robustness**
  (relocated evidence / distractor injection) — reconciling BBox-DocVQA vs 2406.14492.
- **Repo asset.** Ablation **A1/A3** + `metrics/grounding.py` (IoU) + capability `cap_ground` +
  a *distractor-evidence* probe (answer string planted in a wrong region → measure
  P(correct answer ∧ wrong box)) — extends the spatial/context probe.

## "Does it read, or guess — and does it know when it can't?" (our strongest cluster)

Four under-measured reliability axes our probes already prototype:
- **OOV / invented-script fallback (largely unstudied).** PIXEL (arXiv:2207.06991) reads *real*
  unseen scripts via visual similarity, but byte/Unicode models *structurally can't* encode
  non-Unicode glyphs (ByT5 arXiv:2105.13626). VLMs lean on memorized priors over glyph texture
  ("Lost in Font Recognition", arXiv:2503.23768). **No study characterizes the fallback
  distribution** (visual-nearest-neighbour vs UNK vs confident hallucination vs refusal) on truly
  invented glyphs. → our **`oov_probe`** (invented glyphs ± legend, runic, 7-seg) + the fallback
  classifier in `build_insights.py` directly fill this; *novel hypothesis*: pixel input degrades
  smoothly with visual distance while subword collapses to UNK/hallucination; a legend enables
  in-context decoding (visual symbol reasoning) where tokenization can't.
- **DocVQA calibration (gap).** KIE-HVQA (arXiv:2506.20168, NeurIPS'25) covers KIE only; **full-page
  DocVQA ECE/selective-risk is unmeasured.** → our ECE (`metrics/calibration.py`) on every run.
- **Counterfactual pixel-edit sensitivity (no doc benchmark).** Builds on VQA shortcut/counterfactual
  work (arXiv:2210.04692). → our spatial/context **counterfactual controls** (total-at-top,
  inconsistent-total, name-flip) already implement this for documents; *novel metric*: counterfactual
  answer-flip rate as a "reads-pixels-not-priors" score.
- **Reading-direction as an isolated axis (gap).** Order is studied (LayoutReader arXiv:2108.11591),
  *direction* (LTR/RTL/vertical) is not. → our **custom_eval `reading_direction`** + vertical-CJK
  samples; hypothesis: direction-detection error predicts reading-order error.

## Grounded-CoT vs free-CoT at ≤1B (visual drift / capacity floor)

- **Known.** Distill-Step-by-Step (arXiv:2305.02301) — rationale supervision lets 770M beat 540B
  with <80% data (text). Multimodal CoT "drifts from image-grounded evidence, amplifies
  hallucinated steps" (arXiv:2506.17088). Granularity > teacher accuracy (arXiv:2502.18001).
- **Gap.** Transfer to **document** integrative/numeric reasoning at ≤1B is untested; is there a
  **capacity floor** below which free-CoT *net-hurts* (fluent-but-wrong chains)?
- **Hypothesis (ours).** Crossover: answer-only > free-CoT > but **grounded-CoT** (rationale with
  interleaved cell coords/values) > both; ungrounded CoT *increases* confident wrong answers at 1B.
- **Repo asset.** Ablation **A2/A3** on synthetic multi-cell numeric docs + `cap_integ_*` probes.

## Multilingual transfer is script-graph-structured (not isolated for OCR)

- **Known.** Multilingual VLT (arXiv:2506.11820, 2025-06): single-pair FT causes negative transfer;
  balanced mix recovers (+3.46 at 1000/dir). Text-LLM "curse of multilinguality" + curriculum
  effects (arXiv:2510.25947).
- **Gap.** *Which language pairs* help vs interfere for **OCR/extraction**, and whether interference
  tracks **glyph overlap** more than linguistic relatedness, is uncharacterized.
- **Hypothesis (ours).** Transfer is script-graph-structured: en↔es (shared Latin) positive;
  en↔ja (script-distant) interferes at fixed capacity; CJK siblings transfer *asymmetrically*
  (zh→ja/ko helps Han, hurts kana/hangul); interference correlates with glyph-set overlap, testable
  by holding language fixed and swapping script.
- **Repo asset.** Ablation **A4** language-transfer matrix (`plot_ablation.py` heatmap) +
  custom_eval per-language NED.

## Smaller, self-contained ablations worth a figure each

| Direction                                           | Known                                                                      | Gap → our hypothesis                                                                       | Asset                                                   |
| --------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------- |
| **Compression × LLM-size × font cliff**             | SmolVLM: smaller VLMs want *more* token compression (arXiv:2504.05299)     | not isolated; does the OCR-fidelity cliff move with LLM size & font size?                  | new compression-ratio ablation + custom_eval small-text |
| **End-to-end vs layout-first at matched ≤1B**       | GOT/DocTags (end-to-end) vs PaddleOCR-VL (layout-then-recognize)           | never matched at sub-1B; decomposition advantage may *grow* as LLM shrinks                 | two configs, same data → OmniDocBench-style             |
| **Resolution sweet-spot / token-budget knee at 1B** | NativeRes-LLaVA: native-res ≫ tiling at 7B (arXiv:2506.12776)              | at 1B the token budget may invert it; native helps small-text, overlap-tiling helps layout | ablation **A7** sweep                                   |
| **Training-time abstention vs post-hoc conformal**  | conformal/selective OCR is inference-time (arXiv:2502.06884)               | learned ⟨unknown⟩ supervision at 1B (where calibration is worst) untested                  | abstention probe + ECE                                  |
| **Staircase vs leave-one-out order-dependence**     | staged removal used as a reporting device (SpatialLadder arXiv:2510.08531) | not formalized; is forward-add == backward-remove?                                         | run `plot_ablation.py` staircase in *both* orders       |

## Two highest-leverage, genuinely-unpublished targets (proposed thesis)
1. **A document "reads-or-guesses" reliability suite** — fuse our OOV-fallback + counterfactual +
   calibration + reading-direction probes into one score that asks *does a small doc-VLM read the
   pixels, and abstain when it can't?* (no benchmark measures this jointly today).
2. **A causal sub-1B grounding-supervision study** — does forcing "where" improve "whether", and is
   the benefit IID or OOD — settling BBox-DocVQA vs 2406.14492 in the document regime.

Both are directly runnable with assets already in this repo (probes + ablation framework +
metrics), which is the point: the novelty is **isolating, at ≤1B, what the field only measured
confounded at scale.**

### Key references
Distill-step-by-step arXiv:2305.02301 · MoLA arXiv:2402.08562 · BBox-DocVQA arXiv:2511.15090 ·
"Object grounding & hallucination" arXiv:2406.14492 · groundedness in DocVQA arXiv:2503.19120 ·
multilingual VLT arXiv:2506.11820 · NativeRes-LLaVA arXiv:2506.12776 · SmolVLM arXiv:2504.05299 ·
SmolDocling arXiv:2503.11576 · OCRBench v2 arXiv:2501.00321 · OmniDocBench arXiv:2412.07626 ·
KIE-HVQA arXiv:2506.20168 · PIXEL arXiv:2207.06991 · ByT5 arXiv:2105.13626 · conformal abstention
arXiv:2502.06884 · SpatialLadder arXiv:2510.08531.

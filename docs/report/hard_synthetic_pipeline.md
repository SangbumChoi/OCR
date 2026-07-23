# Executable hard-document generation

The synthetic pipeline now has a semantic layer between random content and rendered pixels. It
targets the document skills that are difficult to supervise from public OCR data alone:
multi-cell tables, exact chart arithmetic, multi-hop investment ownership, and quantitative
scientific-paper verification.

## Contract

`docvlm_eval.synth.latent.LatentDocumentGraph` is the source of truth for hard reasoning:

1. `GraphNode` stores rendered and latent facts, including the `field_key` used to recover visual
   evidence after rendering.
2. `GraphEdge` stores typed relations and optional weights, such as direct ownership fractions.
3. `GraphQuery` declares a deterministic operation, input IDs, answer format, metric, task slice,
   and evidence nodes.
4. `resolve()` recomputes the answer and concise rationale from the graph. Validation fails on
   duplicate IDs, dangling relations, unsupported operations, invalid numeric values, or a stale
   authored `expected` answer.
5. `DocBuilder` renders the same values and resolves every evidence key to an exact PDF text box.

The operation registry currently covers direct lookup, sum, mean, difference, ratio, percent
change, relative reduction, extrema, weighted sum, path products, and sums of independent path
products. This is deliberately small and auditable: no teacher model invents hard labels.

Each `gt.json` records:

- the complete `semantic_graph`;
- resolved query answers, rationales, and evidence keys;
- content and template fingerprints;
- a machine-readable `difficulty` profile;
- explicit split provenance and a deterministic suggested split.

## Hard families

| Family | Rendered evidence | Gold program | Level-5 capability |
| --- | --- | --- | --- |
| `hard_table` | dense regional operating table plus an external budget field | lookup, sums, profit, argmax, cross-region budget subtraction | eleven-box evidence for a table-to-summary calculation |
| `hard_chart` | labelled temporal bar chart | lookup, percent change, argmax, multi-year mean | exact temporal chart aggregation |
| `hard_investment` | direct beneficial-ownership schedule | relation lookup, path product, sum of path products | effective ownership across two independent paths |
| `hard_science` | paper title, abstract, equation, result table, caption | lookup, argmin, control-relative reduction, treatment comparison | quantitative claim verification against the stated equation |

The `difficulty_level` knob is an integer from 1 to 5. Level 1 emits direct visual lookup. Higher
levels add aggregation or relational paths. Level 5 enables multi-path or cross-region reasoning
and the largest distractor budget. The profile records reasoning hops, distractor count, visual
density, cross-region status, and required skills, so curriculum sampling does not infer
difficulty from task names.

## Grounded post-training

Hard questions can cite multiple evidence cells. `QAItem.evidence_keys` is resolved after rendering
to `evidence_bboxes`, then `case_to_samples()` places all boxes in `Sample.meta["boxes"]`.
`RewardContext.from_sample()` normalizes every box, and `StructuredPostTrainingDataset` emits the
strict target:

```json
{"answer":"...","evidence":[[0.1,0.2,0.3,0.4]],"rationale":"..."}
```

This means answer correctness, evidence IoU, and grounded-rationale reward components operate on
the same authored graph query. Evidence is removed automatically in spotting-off ablations because
the boxes are resolved only from emitted spotting ground truth.

Control arms are fail-closed. Spotting-off removes grounding QAs, coordinate answers, and evidence
links, not just the top-level box dictionary. Rationale-off removes the executable node/edge/query
payload and retains only graph fingerprints and family provenance, so the operation program cannot
become a hidden reasoning target.

## Generation and leakage gates

```bash
# Curriculum endpoints
python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science \
  --difficulty-level 1 --split-name train --seed 7 --count 100 \
  --out data/generated/hard_train_l1

python scripts/make_realistic_cases.py \
  --only hard_table hard_chart hard_investment hard_science \
  --difficulty-level 5 --split-name heldout --seed 7007 --count 100 \
  --out data/generated/hard_heldout_l5

# Mandatory exact-content leakage check
python scripts/validate_synth_splits.py \
  --split train=data/generated/hard_train_l1 \
  --split heldout=data/generated/hard_heldout_l5 \
  --output docs/results/hard_split_audit.json
```

The default validator rejects the same semantic content fingerprint across splits while reporting
template overlap. `--require-template-isolation` additionally rejects the same graph program
topology across splits, which is useful for a strict template-generalization evaluation. Template
overlap is otherwise allowed so the standard heldout set can measure new values under known
programs, separately from the stricter topology holdout.

## Verified smoke path

The implementation is covered by graph, curriculum, fingerprint, split-leakage, DTO, sample bridge,
and reward-context tests. A real four-family render at difficulty 5 produced twelve hard reasoning
samples; every sample retained a non-empty rationale and between two and eleven normalized evidence
boxes in the structured post-training target. A second-seed heldout render produced eight unique
semantic contents across train and heldout, with template overlap reported separately.

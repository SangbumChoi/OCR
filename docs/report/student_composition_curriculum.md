# Composition curriculum for multi-page and cross-document reasoning

## Purpose

The hard synthetic pipeline already emits exact single-page documents, three-page packets, and
three-source dossiers. A static mixture, however, exposes the hardest source-attribution problems
from the first optimizer update and lets the more numerous QA rows determine their sampling mass.
That is not a controlled curriculum.

The native student now treats composition complexity as a secondary sampling axis while preserving
the primary task/source/language/component balance. The implementation is shared by
[`docvlm_eval.unified.hf`](../../src/docvlm_eval/unified/hf.py),
[`docvlm_eval.student.data`](../../src/docvlm_eval/student/data.py), and
[`docvlm_eval.student.curriculum`](../../src/docvlm_eval/student/curriculum.py).

## Exact metadata path

Synthetic `render.rendered_page_count` and `render.document_count` survive the complete path:

```text
gt.json
  -> UnifiedSample.meta
  -> UDD page_count/document_count columns
  -> normalized weighted mixture
  -> every expanded QA and grounding example
```

Public UDD components that predate these columns receive the conservative default `(1, 1)` during
mixture normalization. The tier function is deterministic:

| Tier | Condition |
| --- | --- |
| `cross_document` | `document_count > 1` |
| `multi_page` | `document_count == 1` and `page_count > 1` |
| `single_page` | otherwise |

Document count takes precedence because a dossier may also contain several rendered pages.

## Two-stage sampler

The existing `balance_by` key remains the primary distribution. For the default `task` setting, the
sampler first draws a task using its configured group weight. It then draws an available
composition tier inside that task using the active curriculum weights, followed by a uniform
example inside the selected task-tier cell.

For primary group \(g\), tier \(c\), and \(n_{gc}\) examples in the cell, each example has
conditional mass \(w_c / n_{gc}\). A tier therefore receives \(w_c\) mass rather than mass
proportional to its QA count. Missing task-tier cells are renormalized only within that task; they
do not silently change the requested primary task probability.

The same normalization is used with dense aspect-ratio buckets. Bucket mass is computed from the
sum of these per-example masses, so marginalizing over buckets recovers both the primary group
distribution and the requested composition distribution.

## Exact schedule

`training.pretraining.input_pipeline.composition_curriculum` uses absolute optimizer-step
boundaries. This is deliberate: a fraction driven by runtime token counts can be prefetched before
the trainer observes those counts, making its transition dependent on worker timing. Absolute
steps are known when the batch sampler emits each batch and therefore reproduce exactly after
checkpoint resume.

The default schedule is:

| Stage | Optimizer steps | Single page | Multi-page | Cross-document |
| --- | ---: | ---: | ---: | ---: |
| `single_page_bootstrap` | `[0, 100000)` | 1.00 | 0.10 | 0.00 |
| `multi_page_bridge` | `[100000, 600000)` | 0.65 | 0.70 | 0.25 |
| `cross_document_refinement` | `[600000, end)` | 0.45 | 0.80 | 1.00 |

Every stage must define all three finite, non-negative weights and retain at least one positive
weight. Boundaries must be strictly increasing, IDs unique, and the final `until_step` must be
`null`. Blueprint validation fails closed on a malformed schedule.

## Provenance and observability

The schedule fingerprint is included in the pretraining supervision contract and every checkpoint.
Exact resume rejects a changed composition schedule. W&B/JSONL logging records:

- `train/composition_curriculum_stage`;
- `train/composition_weight/single_page`;
- `train/composition_weight/multi_page`;
- `train/composition_weight/cross_document`.

The full experiment now generates `hard_table`, `hard_chart`, `hard_investment`, `hard_science`,
`hard_diagram`, `audit_packet`, and `investment_dossier`, so all three tiers are present in the
production synthetic component. The tiny smoke experiment remains valid with only single-page
examples because unavailable tiers are renormalized within its sole task.

## Falsifiable ablation

Compare the staged schedule with a static control that assigns the final-stage weights from step
zero. Hold generated rows, task weights, tokenizer, initialization, teacher targets, total
effective tokens, and seeds fixed. Promotion requires:

1. improved heldout `document_count=3` and `page_count=3` slices;
2. no regression on single-page grounding, OCR, or multilingual slices;
3. no larger train-minus-heldout gap;
4. identical realized token and student-FLOP budgets within the existing overshoot gate.

This tests curriculum ordering, not the presence of composed documents.

The paired three-seed design is executable:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_composition_curriculum_sweep.yaml \
  --dry-run
```

Remove `--dry-run` only after the resolved six-run plan and target-GPU feasibility gates have been
reviewed.

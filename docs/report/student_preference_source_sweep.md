# Gold-anchored preference bootstrapping

## Question

[`configs/sub1b_preference_source_sweep.yaml`](../../configs/sub1b_preference_source_sweep.yaml)
tests whether one exact structured target in each verifier-ranked DPO candidate set improves
learning from a weak SFT checkpoint without harming heldout document understanding.

This is an executable six-run design with two arms and three paired stochastic replicates. It is
not evidence that gold anchoring improves quality until the GPU runs and paired evaluation
artifacts exist.

## Failure mode

A near-random or lightly supervised document VLM can emit only malformed JSON. When all eight
reference-policy candidates receive the same malformed-response reward, best-versus-worst
selection has no verifier margin. The rollout consumes compute, but DPO performs no optimizer
update. A completed stage can therefore contain many candidate attempts and zero accepted pairs.

The production default, `gold_anchored_verifier_ranked`, keeps the frozen SFT rollout but replaces
one sampled candidate with the exact answer/evidence/rationale token sequence produced by the
training collator. It then scores every candidate with the same structured verifier and applies
the normal minimum-margin gate. The anchor is not automatically chosen: verifier ranking remains
authoritative. If token truncation makes the collated anchor malformed, the stage fails instead of
training on a claimed gold response.

The control, `reference_verifier_ranked`, uses only frozen-SFT samples. It remains useful after SFT
is strong enough to produce diverse valid candidates and avoids repeatedly presenting the authored
target as the preferred response.

## Matched design

| Arm | Candidate set | Intended estimand |
| --- | --- | --- |
| `reference_only` | eight frozen-SFT samples | original model-candidate preference learning |
| `gold_anchored` | seven effective samples plus one exact collated target | weak-policy bootstrap |

Both arms execute an eight-member frozen-reference rollout. The anchored arm samples all eight
members and then replaces one row, so rollout RNG consumption and visual-prefix compute remain
matched. Pair scoring uses the realized padded completion length; algorithmic student FLOPs
therefore include a longer gold response when applicable. Both arms stop at the same `192e15`
student-FLOP budget rather than assuming equal candidate-group counts.

All other controls are paired: document generation, public-data selection, initialization,
pretraining, evidence-linked SFT, DPO objective and beta, frozen reference, verifier, reward
weights, malformed reward, margin, group size, rollout controls, optimizer, activation
checkpointing, and evaluation.

Inspect or execute the sweep:

```bash
python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_source_sweep.yaml \
  --dry-run

python scripts/run_student_sweep.py \
  --sweep configs/sub1b_preference_source_sweep.yaml
```

## Evidence contract

`preference/gold_anchor_applied` distinguishes the candidate source in local metrics and W&B.
`preference/accepted_pairs`, `preference/skipped_pairs`,
`preference/verifier_reward_margin`, actual optimizer steps, and student FLOPs must accompany
heldout quality. `preference/sampled_reward_mean`,
`preference/sampled_reward_std`, and
`preference/sampled_valid_structure_fraction` exclude the authored anchor and prevent its guaranteed
validity from being reported as model-generated quality. The preference source is part of the
checkpointed objective contract, so resume rejects switching candidate construction mid-run.

The primary comparison is paired heldout score. Promotion additionally requires non-regressive
grounding, multilingual, reliability, reasoning, parameter-budget, and train-minus-heldout gates.
The gold-anchored arm should not be promoted merely because it performs more optimizer updates.
It must improve heldout capability without increasing unsupported rationales, answer leakage,
memorization, or malformed responses.

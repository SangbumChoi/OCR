# Canonical robustness-slice evaluation

Headline accuracy can hide a gain confined to one template, language, easy single-region
question, clean rendering, or an authored document mark. Native-student evaluation therefore
assigns every sample five
canonical labels:

| Axis | Canonical source |
| --- | --- |
| `document_family` | semantic template family, then document type, then dataset source |
| `language` | question language, then document language |
| `evidence_count` | number of gold evidence boxes required by the question |
| `degradation` | `clean` or the rendered degradation preset |
| `overlay` | `none`, one mark type, or a sorted `+`-joined handwriting/stamp/seal mixture |

The labels are stored in each evaluation row under `robustness_slices`. Synthetic conversion also
stores them in `Sample.meta`, so the labels survive JSONL serialization and can be audited before
model execution. Missing public-dataset metadata is represented as `unknown` or `und`; it is never
silently treated as clean or English.

## Evaluation artifacts

Each split summary contains:

- `by_robustness_axis/<axis>/<value>` summaries with sample count, benchmark score, structured
  reward, valid-structure fraction, and answer rate;
- `robustness_coverage` with exact label counts and unknown counts for every required axis;
- the legacy answer-type, source, and language summaries for compatibility.

The train-versus-heldout comparison includes only values present in both splits. This avoids
presenting unmatched families or degradation presets as a generalization gap.

W&B uses an axis-first hierarchy:

```text
eval_by_slice/document_family/<value>/train
eval_by_slice/document_family/<value>/heldout
eval_by_slice/language/<value>/train
eval_by_slice/language/<value>/heldout
eval_by_slice/evidence_count/<value>/train
eval_by_slice/evidence_count/<value>/heldout
eval_by_slice/degradation/<value>/train
eval_by_slice/degradation/<value>/heldout
eval_by_slice/overlay/<value>/train
eval_by_slice/overlay/<value>/heldout
```

Selecting the two metrics with the same `<axis>/<value>` suffix produces a directly matched panel.
Values are URL-encoded as one W&B path segment, so a family such as `invoice/receipt` appears as
`invoice%2Freceipt` rather than creating a second hierarchy level. The split name is `heldout`,
not `held`.

## Matched ablations

The sweep aggregator intersects slice labels across replicates and the matched baseline before
estimating effects. For every surviving `axis/value`, it reports:

- mean train and heldout score;
- heldout mean and standard deviation;
- paired delta against the replicate-matched baseline;
- a deterministic paired bootstrap 95% interval when multiple replicates exist.

The output is nested under `heldout_by_robustness_axis`,
`heldout_robustness_statistics`, `heldout_robustness_delta_vs_baseline`, and
`heldout_robustness_delta_statistics`.

Coverage is a prerequisite for interpretation. A populated axis is not necessarily a diverse
axis: a language slice containing only English or an evidence-count slice containing only zero
cannot support a multilingual or multi-evidence robustness claim. Use `robustness_coverage` and
the per-value `n` alongside every reported gain.

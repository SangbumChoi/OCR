# Generation and rendering safeguards

Long structured answers have two distinct failure surfaces: the language model can enter an exact
token cycle near its completion limit, and the synthetic renderer can silently rasterize only the
first page of a long HTML document. Treating both as a generic length problem hides data loss.

## Exact suffix-cycle guard

Student inference, preference rollout, RLVR rollout, and held-out evaluation share three controls:

| Control | Default | Meaning |
| --- | ---: | --- |
| `repetition_guard_min_tokens` | 24 | Do not intervene before this many completion tokens |
| `repetition_guard_max_period` | 16 | Inspect exact trailing cycles up to this period |
| `repetition_guard_repetitions` | 3 | Require this many consecutive copies of the cycle |

When appending a candidate would satisfy all three conditions, generation emits EOS instead. The
guard is intentionally narrower than no-repeat n-gram decoding. Repeated HTML tags, table columns,
punctuation, and labels remain legal when they do not form one consecutive trailing cycle.

Evaluation records `generated_tokens`, `reached_max_new_tokens`, and `degenerate_repetition` per
sample. Split summaries expose `max_token_rate` and `degenerate_repetition_rate`. Accuracy should
not be accepted as improved when either diagnostic regresses materially, especially on table,
full-page OCR, and long-context slices.

## Bounded task-aware token budgets

A single 128-token horizon is adequate for concise KIE but can truncate a valid table, full-page
transcription, reading-order sequence, or evidence-linked reasoning response. Evaluation,
preference sampling, and RLVR therefore share a task-label policy with three parts:

- `max_new_tokens` is the default budget;
- `max_new_tokens_by_answer_type` maps exact public task labels or trailing-wildcard prefixes to
  larger budgets;
- `max_new_tokens_hard_cap` bounds every override.

The production policy uses a 128-token default and a 512-token hard cap. Exact labels win over
prefix matches, and the longest matching prefix wins among wildcard rules. Resolution uses only
the public `answer_type`; it never examines gold text, target length, correctness, or hidden
annotations. The same resolved budget applies to every candidate in one preference or RLVR group.

Per-sample evaluation writes `generation_token_budget` and
`generation_token_budget_source`. Summaries report `mean_generation_token_budget` and
`budget_escalation_rate`; rollouts log the corresponding budget and escalation metrics. Actual
completion width continues to drive preference and RLVR FLOP accounting. The complete policy is
checkpointed as part of each rollout contract, so changing it invalidates resume.

## Tokenized target-fit audit

The production experiment runs `audit_generation_budgets` after the tokenizer and structured
samples exist but before student initialization. It serializes each target through the same SFT
dataset contract, counts tokenizer pieces plus EOS, and evaluates the exact evaluation,
preference, and RLVR policies. The gate requires policy identity and the configured coverage on
train, validation, and heldout. A continuation run audits its active replay mixture with the
attested parent tokenizer before SFT.

Budget recommendations use train and validation only. Heldout targets test coverage but never
alter a recommendation. The JSON artifact contains aggregate length quantiles, coverage,
near-budget counts, and a bounded list of overflow sample IDs. It deliberately excludes target
text, repeated completion tokens, and rendered HTML bodies. Complex tables and full pages remain
inspectable through the rendering audit, while the budget audit stays compact enough to compare
without duplicating a long context.

## HTML and full-page integrity

The renderer records total PDF pages separately from rasterized pages. Table and full-text targets
cannot use a first-page image when the HTML overflows:

1. Render the requested mode and inspect the PDF page count.
2. Automatically replace an overflowing first-page render with one vertical all-page canvas.
3. Audit rendered-page coverage, blank pages, and the survival of every non-empty table cell in the
   PDF text layer.
4. Fail before allocating a canvas above `max_canvas_pixels`.

The resulting sample metadata includes `render.auto_expanded_from_first` and
`render.layout_audit`. The audit reports page counts, omitted pages, blank pages, table/row/cell
counts, missing cells, canvas pixels, and failures. Complex layouts should be visualized as an
all-page canvas only while they fit the explicit pixel budget; beyond that budget they must be
split into page-level or tiled samples with shared document identity rather than downscaled into
unreadable pixels.

## Release gates

- Compare `max_token_rate` and `degenerate_repetition_rate` by answer type and context length.
- Compare score at a fixed policy and report mean budget and escalation rate; do not call a larger
  horizon a free quality gain.
- Require zero omitted pages and zero missing cells for table and full-text synthetic examples.
- Inspect representative all-page canvases at both native and model-input resolution.
- Keep token-cycle controls identical across training rollout and evaluation checkpoints.
- Require the tokenized target-fit audit to pass before allocating the student model.
- Treat a lower token count as a correction only when task score and structural validity do not
  regress.

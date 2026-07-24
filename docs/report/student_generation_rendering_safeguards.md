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
- Require zero omitted pages and zero missing cells for table and full-text synthetic examples.
- Inspect representative all-page canvases at both native and model-input resolution.
- Keep token-cycle controls identical across training rollout and evaluation checkpoints.
- Treat a lower token count as a correction only when task score and structural validity do not
  regress.

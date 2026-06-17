# Probing spatial & context understanding — falsifiable hypotheses and signal criteria

Asking "does the VLM understand *space* and *context*?" is easy to fake: a model can score
well by exploiting **shortcuts** — language priors ("totals are at the bottom"), position bias,
binary guessing, rubber-stamping ("yes it's consistent"), or hallucinating a plausible value.
So every hypothesis below is paired with a **control** that a shortcut would *fail*, and a
**signal criterion** that only true understanding satisfies. Probe:
[`data/benchmarks/spatial_context_probe`](../data/benchmarks/spatial_context_probe) (rendered,
exact GT); generator `scripts/make_spatial_context_probe.py`.

## Methodology: signal, not score

For each capability we report a **signal** = (does it pass the test) **AND** (does it pass the
control). Plain accuracy is reported too, but the *criterion for "understands"* is the
shortcut-robust signal. We design controls so that the **null hypothesis = "uses a shortcut"**
is falsifiable.

---

## Spatial understanding

| # | Hypothesis | Test | Shortcut it must beat | Signal / criterion |
|---|---|---|---|---|
| **S1** | Knows absolute region | "Which quadrant has ZEBRA?" ×4 (one per quadrant) | position bias (always answer one quadrant) | accuracy **> 25%** chance **and** not all-same-answer; criterion ≥ 3/4 |
| **S2** | Knows relative position from *perception* | "Is TOTAL above or below the items?" on **normal** (total bottom) **and counterfactual** (total top) | language prior "total = bottom" | **must be correct on BOTH**. Define **prior-reliance gap = acc(normal) − acc(counterfactual)**; understanding ⇒ gap ≈ 0 with both = 1 |
| **S3** | Box tracks the element | return bbox of ANCHOR at top / mid / bottom | constant-box prior (emit same box always) | **center-tracking**: predicted box center-y must increase with true position (correlation **r > 0.8**) **and** mean IoU > 0.3 |

**Why these are critical, not generic:** S2's counterfactual is the key idea — a model that
answers "below" on the *top* invoice is pattern-matching, not seeing. S3 turns grounding into a
*relational* test: even if absolute IoU is low, a box that *moves with* the element shows real
spatial tracking; a constant prior box scores r≈0.

## Context understanding

| # | Hypothesis | Test | Shortcut it must beat | Signal / criterion |
|---|---|---|---|---|
| **C1** | Cross-region numeric consistency | "Do the items add up to the TOTAL?" on **consistent** (yes) **and inconsistent** (no) | rubber-stamp "yes" | **must catch the inconsistent case** (n→no); criterion: both correct, esp. control=no |
| **C2** | Absence / anti-hallucination | "What is the discount?" when **no discount field exists** | invent a number | answers **"none"**; any numeric answer = hallucination (fail) |
| **C3** | Field disambiguation by context | "What is the TOTAL (not subtotal)?" with a look-alike Subtotal present | grab the nearest/first number | returns the **Total**, not the Subtotal |
| **C4** | Cross-reference + counterfactual sensitivity | "How much does the 'Bill to' person owe?" with header **Bob** vs **Alice** (table maps names→amounts) | answer invariant to header (ignore context) | **answer must flip** Bob→45, Alice→80; criterion: both correct (= sensitivity to the relevant context) |

**Why these are critical:** C1's inconsistent variant defeats a model that always says
"consistent"; C2 measures *honesty under absence* (the OCR-hallucination failure mode); C4's
two-variant design measures **counterfactual sensitivity** — if changing the only relevant
token (the name) doesn't change the answer, the model isn't using context, it's guessing.

---

## Signal-criteria summary (what counts as "has the capability")

| Capability | PASS criterion (shortcut-robust) |
|---|---|
| Absolute spatial (S1) | ≥ 3/4 quadrants correct, answers not constant |
| Relative spatial (S2) | correct on **both** normal & counterfactual (prior-reliance gap ≈ 0) |
| Spatial tracking (S3) | center-y correlation r > 0.8 **and** mean IoU > 0.3 |
| Consistency (C1) | catches the inconsistent case (control=no correct) |
| Anti-hallucination (C2) | "none" on the absent field |
| Disambiguation (C3) | returns Total, not Subtotal |
| Context sensitivity (C4) | answer flips correctly with the header (both variants) |

A model "understands space/context" only if it clears the **control**, not just the easy case.
This converts a vague question into measurable, falsifiable signals.

## Results (SmolVLM, CPU) — interpretation

Signal table (✅ = clears the shortcut control), from `scripts/analyze_probe_signals.py`:

| model | S1 absolute | S2 relative | S3 tracking | C1 consistency | C2 anti-halluc | C3 disambig | C4 context-sens |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| SmolVLM-256M | ❌ (acc .25, bias) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| SmolVLM-500M | ✅ (acc .75) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

**What the *controls* exposed (the whole point):**

- **S2 — the constant-answer trap.** Both models answer *"above"* for **every** layout, so they
  score **correct on the counterfactual** (total-at-top) and **wrong on the normal** one →
  `prior_reliance_gap = −1.0`. A naive test using only the counterfactual would have *falsely*
  certified "relative-position understanding". The control proves the model isn't perceiving
  position at all — it emits a fixed token. **This single result justifies the whole
  control-paired methodology.**
- **S1 — absolute spatial emerges with size.** 256M answers *"top-left"* for all four quadrants
  (position bias, acc .25 = chance); **500M reaches acc .75 with 3 distinct answers → PASS.**
  (A trailing-period scoring bug initially masked 500M's success — fixed; see results_analysis.)
- **C1 — rubber-stamping caught.** Both say *"Yes, it's consistent"* even when the total is
  deliberately wrong → the **inconsistent control** flips it to FAIL. No real cross-region
  arithmetic verification.
- **C2 — hallucination caught.** Asked for a non-existent discount, 256M invents *"$20.50"*,
  500M invents *"0"* — neither abstains with *"none"* → both FAIL (honesty-under-absence gap).
- **C4 — no context sensitivity.** Each model anchors to one table row regardless of the
  *"Bill to"* name (256M always Bob's $45; 500M doesn't flip correctly either) → the answer
  does **not** track the only relevant token → not using context.
- **C3 — the one real pass.** Both correctly return the **Total** (not the look-alike
  Subtotal): local field disambiguation works.

**Bottom line:** at ≤0.5B, document VLMs are **literal readers with local disambiguation** but
lack **spatial reasoning (relative position, grounding), cross-region verification, honesty
under absence, and context sensitivity**. Crucially, **most of these gaps are invisible to
single-example accuracy** and only surface under the paired controls — which is exactly the
signal framework this probe provides. The same probe + analyzer will rank the 1B / spotting
models on a GPU run with no code change.

## Extending to all models
Run any model with `python scripts/run_matrix.py --models <m> --benchmark
data/benchmarks/spatial_context_probe/probe.jsonl`. The control-aware signals above are then
computed by `scripts/analyze_probe_signals.py`, which prints the PASS/FAIL per criterion and
the prior-reliance gap / tracking correlation per model.

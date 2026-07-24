# End-to-end student experiment runner

[`scripts/run_student_experiment.py`](../../scripts/run_student_experiment.py) compiles one validated
YAML experiment into a resumable stage DAG. It connects hard-document synthesis, semantic split
validation, UDD conversion, weighted data mixing, quality-gated cross-tokenizer distillation,
tokenizer training, student initialization, pretraining, grounded SFT, RLVR, and train/heldout
generation evaluation. The full configuration first benchmarks packed visual attention on the
active runtime and stores the requested/resolved backend, numerical parity, latency, throughput,
and peak memory as a checked run artifact.

Random initialization is reproducible: `initialization.seed` is validated, passed to model
construction before parameter allocation, included in the stage signature, and written into the
initial checkpoint metadata.

External generation inputs are content-addressed. The plan records the byte count and SHA-256 of
`synthetic.config`; changing that YAML invalidates the experiment fingerprint and every dependent
stage instead of incorrectly resuming old documents. Configured sequence-teacher prediction files,
local initialization checkpoints, and initialization token maps receive the same treatment. Pinned
Hub initialization sources receive dedicated acquisition stages whose manifests validate the
resolved revision and cached weight files. A combined SHA-256 over the
`docvlm_eval` Python source tree and every compiled script entrypoint also invalidates resume after
generator, model, loss, reward, or runner implementation changes.

RLVR uses periodic supervised replay from its active samples by default. Set
`posttraining.rlvr.replay_samples` to an external benchmark JSONL to anchor broader multimodal
capabilities; the compiler content-addresses that file and passes it to the RLVR stage.
Set `posttraining.rlvr.enabled: false` with all RLVR runtime overrides null to compile an SFT-only
DAG. In that mode evaluation loads `@student:sft` directly. Non-null disabled-stage overrides fail
before data generation so an intended RLVR treatment cannot disappear silently.

The final evaluation also writes `gates.json`. Gate outcomes are `pass`, `fail`, or
`insufficient_evidence`; missing comparisons never count as success. The parameter gate uses the
actual loaded model count. Generalization, grounding, counterfactual reasoning, and reliability
require a matched reference-checkpoint evaluation, while multilingual retention requires a
per-language monolingual-control evaluation. The visual-efficiency gate consumes the preflight JSON
and requires matched loop/candidate measurements from the exact resolved student configuration.
Configured evaluation roots are content-addressed so changing a baseline invalidates the
evaluation stage.

For matched multi-run ablations, use
[`student_sweep_runner.md`](student_sweep_runner.md). It compiles RFC 6902 experiment/blueprint
patches, rejects changes to declared fixed controls, reuses this runner for every variant, and
aggregates baseline deltas from the final train/heldout comparisons.

## Configurations

The full approximately 800M experiment is
[`configs/sub1b_experiment.yaml`](../../configs/sub1b_experiment.yaml). Inspect its commands and
dependencies without creating files:

```bash
python scripts/run_student_experiment.py --dry-run
```

The full plan has 18 stages, including the target-device visual backend preflight. The CPU contract
test disables that performance preflight and omits the full plan's public-Hub acquisition, leaving
16 stages with a dummy cross-tokenizer teacher, one 587k-parameter student, and one optimizer step
per training phase:

```bash
python scripts/run_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml
```

The resolved full blueprint uses a hard 20B effective-token pretraining budget. It repeats
deterministic sampler epochs until that counter is reached; `pretraining.max_steps` is a deliberate
smoke or ablation ceiling. Checkpoints preserve supervised, text, and effective counters so a
resumed run continues the same learning-rate and curriculum position.

The experiment YAML controls synthetic families, difficulty, independent split seeds and counts,
degradation, data components and weights, tokenizer size, initialization arm and transfer sources,
training limits, evaluation settings, and W&B metadata. `synthetic.train_count` and
`synthetic.heldout_count` may override the legacy shared `synthetic.count`, allowing training-scale
experiments to retain one fixed benchmark. The runner writes a resolved architecture blueprint whose
`data_mix`, sampler groups, and tokenizer/model dimensions match the experiment.

`runtime.visual_backend_benchmark` compiles
[`benchmark_student_visual_backend.py`](../../scripts/benchmark_student_visual_backend.py) as the
first artifact-checked stage. Its sequence lengths, backends, warmup, measured iterations, mode,
precision, seed, parity tolerance, Flex requirement, and W&B destination are all explicit in the
experiment fingerprint. `initialize_student` depends on the resulting JSON. Set `require_flex:
true` on the target CUDA image to stop before model initialization when `auto` or explicit `flex`
falls back or fails. Keep it false when the portable loop backend is an accepted treatment; the
resolved fallback remains visible in the artifact and W&B.

The final evaluator receives that artifact through `--visual-backend-benchmark`. The default
`visual_efficiency` gate requires schema-v2 CUDA training-mode evidence over at least 4,096 visual
tokens, batch size two, three warmup iterations, ten measured iterations, and three order-rotated
rounds. The `auto` candidate must resolve to `flex`, be at least 1.05 times as fast as `loop` at
paired median latency, never fall below 1.0 in an individual round, and stay within 1.05
worst-round loop peak allocated memory. The same speed and memory rules apply against the matched
`dense_adaptive` control. It must also remain within 0.02 maximum absolute output delta. A CPU
report, legacy or short benchmark, or missing dense control is `insufficient_evidence`; a
mismatched architecture, fallback, execution error, numerical violation, or runtime regression
is `fail`.

Run only the authoritative target-GPU preflight before committing to a full training job:

```bash
python scripts/run_student_experiment.py \
  --experiment configs/sub1b_experiment.yaml \
  --no-resume \
  --to-stage visual_backend_benchmark
```

The production experiment enables `runtime.visual_backend_benchmark.require_deployment_gate`.
This first stage therefore writes the benchmark, embedded gate decision, experiment state, logs,
and optional W&B evidence, then stops nonzero unless the complete CUDA parity, backend, dose,
paired-speed, and worst-memory contract passes. `--no-resume` forces a fresh measurement instead
of accepting an older successful state. Blocking mode rejects a non-CUDA runtime before allocating
the benchmark model.

Initialization sources may be local paths or immutable Hub mappings. Hub snapshots remain in the
shared Hugging Face cache while each run stores a content manifest, avoiding checkpoint duplication
across paired sweeps. See
[`student_initialization_runner.md`](student_initialization_runner.md) for the source schema,
compatibility analyzer, and five-arm initialization suite.

Set `evaluation.baseline_evaluation` to an evaluation root produced by
`scripts/eval_student.py`, and set `evaluation.monolingual_control_evaluation` to the corresponding
control root. Both roots must contain `comparison.json` and split-level `per_sample.jsonl` files.
The native evaluator records each sample's source metadata and geometric-mean generated-token
confidence, enabling matched counterfactual and fixed-coverage reliability checks.

Public components may use a local `path` or a pinned Hugging Face `hub` specification. The full
configuration acquires the public UDD train fold at an immutable commit, validates its schema,
payload, duplicate identities, and image dimensions, then mixes it at 55% with 45% authored hard
documents. See
[`student_data_acquisition.md`](student_data_acquisition.md).

The full configuration uses `lfm2_5-vl-1.6b` as an offline sequence teacher. It exports 4,096
deterministic fingerprinted image-question requests, resumes generation from one pinned Hub
revision, applies each sample's native metric as a quality gate, and retains exactly 400 accepted
targets. Model/revision mismatches fail during apply. The student tokenizer is fit without teacher
answers, preventing teacher-specific vocabulary drift. The tiny configuration uses `dummy-echo`;
its deliberately incorrect outputs prove that all teacher responses can be rejected while the
pipeline safely trains on gold.

## Data mixture

Each `data.components` entry names an on-disk or pinned-Hub UDD dataset, sampling weight, and
optional source fold. `path: "@synthetic"` refers to the UDD produced by the experiment. The mixer
normalizes every component to the canonical UDD superset, marks selected rows as training data, and
records their component identity without duplicating rows.

`mixture_manifest.json` records row counts, fingerprints, paths, and normalized weights. The
balanced batch sampler applies those weights at runtime with `balance_by: component`. This keeps
the physical corpus stable while allowing mixture probabilities to change explicitly.

## Resume and provenance

Every stage has a command signature, dependencies, and required artifacts. Successful stages are
skipped only when their state signature still matches and every artifact remains valid. Interrupted
pretraining, SFT, and RLVR stages automatically pass `--resume latest` only when the interrupted
state has the same signature and a checkpoint pointer exists. A changed upstream checkpoint starts
the dependent stage fresh.

When a signature changes, the runner removes only that stage's declared outputs inside the
experiment root before rebuilding them. It applies the same cleanup to interrupted
non-checkpoint stages and completed stages with invalid artifacts. Checkpoints remain intact for
an interrupted training stage with the same signature, preserving exact resume.

Each run root contains:

- `resolved_blueprint.yaml`, `experiment_spec.json`, and `experiment_plan.json`;
- `state/stages/<stage>.json` with status, command, timing, return code, and signature;
- `logs/<stage>.log` with combined process output;
- `artifacts/` with immutable stage handoffs and final split comparison;
- `run_summary.json` with completed or skipped outcomes.

Run a bounded section only after its external dependencies have completed:

```bash
python scripts/run_student_experiment.py \
  --from-stage pretrain \
  --to-stage evaluate
```

The runner fails closed when a command exits successfully but its declared artifacts are absent.
Train and heldout synthesis must use different seeds, and split validation runs before either split
can enter training or evaluation.

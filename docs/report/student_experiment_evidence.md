# End-to-end experiment evidence attestation

An executable training DAG is not evidence that the DAG ran, and a completed smoke run is not
evidence that the model improved. The native-student pipeline keeps those claims separate through
[`scripts/attest_student_experiment.py`](../../scripts/attest_student_experiment.py).

## Two independent conclusions

Every attestation reports:

- `contract_status`: whether the current code and experiment configuration match every completed
  stage signature, all declared artifacts still exist, every evidence file matches its digest, the
  model is below one billion parameters, optimization advanced in pretraining and each configured
  post-training stage, evaluation used the final checkpoint, and both train and heldout generation
  produced samples;
- `capability_status`: the independent result from `artifacts/evaluation/gates.json`, including
  matched-reference quality, multilingual retention, reliability, target-device visual efficiency,
  and full-student training feasibility;
- `claim_scope`: `execution_contract_only` unless both conclusions pass. Only then does the
  attestation authorize `deployment_capability`.

This means the one-step CPU experiment can prove orchestration, checkpoint handoffs, gradient
updates, RLVR replay, and heldout evaluation without making a quality claim. Missing baselines or
target-GPU measurements remain `insufficient_evidence`, never an implicit pass.

## Create and verify

Run the experiment without resume when fresh execution evidence is required:

```bash
python scripts/run_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml \
  --no-resume
```

Create the deterministic attestation:

```bash
python scripts/attest_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml
```

The default `full` hash mode hashes every declared artifact, log, and resolved latest checkpoint,
including model and optimizer files. This is appropriate for a release or promotion decision. The
`metadata` mode still hashes files up to 1 MiB but records larger files only by path and byte count;
it is faster but does not authorize a tamper-resistant checkpoint claim.

Recompute the complete evidence view from the current worktree and compare it with the stored
attestation:

```bash
python scripts/attest_student_experiment.py \
  --experiment configs/sub1b_experiment_tiny.yaml \
  --verify outputs/docvlm-tiny-smoke/evidence_attestation.json
```

Verification exits nonzero after any relevant plan, source, stage signature, log, artifact,
checkpoint, trainer-state, comparison, or gate change. The attestation hash excludes its own output
file and is therefore deterministic.

## Resume-safe summary

`run_summary.json` schema version 2 distinguishes the latest invocation from authoritative stage
state:

- `outcomes` and `stages[].invocation_status` say whether the latest invocation completed, skipped,
  or did not select a stage;
- `stages[].state_status`, `signature_matches`, and `artifacts_valid` preserve whether the stage was
  actually completed under the current experiment;
- original stage timing, return code, and log path remain visible after a resume-only invocation;
- `pipeline_complete` is true only when every compiled stage is completed with a current signature
  and valid declared artifacts.

Consequently, a successful resume no longer replaces evidence of the original execution with a
list containing only `skipped`.

## Evidence boundary

An attestation proves the integrity and execution state of one experiment root. It does not turn a
one-step smoke run into statistical evidence. Model promotion still requires the matched,
multi-seed suites, heldout confidence intervals, deployment preflights, and fail-closed promotion
rules defined by the sweep runner.

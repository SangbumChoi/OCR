# Frontier-method implementation evidence

The 100-method catalog separates research selection from implementation evidence. A benefit,
limitation, and `adopt` label do not prove that a method exists in the training system.
[`configs/frontier_method_evidence.yaml`](../../configs/frontier_method_evidence.yaml) therefore
binds every adopted method to:

- live implementation files and exact text anchors;
- the complete set of adjustable knobs declared by its catalog row;
- live test files and exact verification anchors; and
- a concise claim limited to what those anchors establish.

[`scripts/audit_method_evidence.py`](../../scripts/audit_method_evidence.py) checks repository-local
paths, rejects stale anchors, requires exact knob coverage, and forbids implementation evidence
from documentation files. It writes a fingerprinted JSON report with file hashes but does not copy
source bodies into the artifact. Identical evidence is referenced once per method rather than
expanded into a large HTML or Markdown table.

## Production gate

The production experiment enables `method_evidence` and runs `audit_method_evidence` as stage zero.
The visual-backend benchmark, full training-feasibility benchmark, and student initialization all
depend on it. Starting from a later stage requires a completed audit with the current catalog,
evidence manifest, source anchors, tests, and experiment signature.

The CPU smoke configuration disables this repository-level preflight because its purpose is
training orchestration, not research-selection attestation. LFM transfer pilot and confirmatory
sweeps inherit the production gate.

## QLoRA correction

The first audit exposed a false-positive adoption claim: the external LFM LoRA path implemented
LoRA placement but still loaded dense base weights. It now defaults to NF4 four-bit base weights,
double quantization, and k-bit training preparation. `--quantization-bits 16` remains an explicit
dense control. Four-bit loading fails before allocation without CUDA, and the adapter artifact
records the realized base-weight quantization contract.

## Claim boundary

This audit proves implementation traceability, not empirical superiority. Methods marked
`ablate` still require matched multi-seed evidence before promotion. An adopted method can remain
disabled by a zero loss weight in a particular control arm; its evidence claim is that the
mechanism and controlled knob exist and are verified, not that every experiment activates it.

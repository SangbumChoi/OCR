# Local Examples

This directory contains small, inspectable examples used by benchmark tooling and probe
definitions. It is not the full training corpus.

- `benchmarks/` contains source examples downloaded from public datasets. Each benchmark keeps
  images in `download/` and one `annotations.jsonl` file with relative image paths and source
  annotations.
- `probes/` contains probe annotations and generator metadata. Probe images are intentionally
  omitted while the next differential probe design is being defined. Recreate them with the
  corresponding scripts in `scripts/` when needed.

Large generated training data and model outputs belong in ignored runtime directories or on the
dataset hub, not in this checked-in examples tree.

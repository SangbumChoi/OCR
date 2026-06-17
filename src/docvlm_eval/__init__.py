"""docvlm_eval: a reproducible evaluation harness for small (<1B) document-understanding VLMs.

The package is organised around three pluggable pieces:

* ``models``      - thin :class:`ModelAdapter` wrappers + a registry, so adding a new
                    candidate model means writing one small adapter and registering it.
* ``benchmarks``  - loaders that normalise every dataset into a single sample schema, and
                    a builder for the custom robustness probe.
* ``metrics``     - scoring functions that go *beyond accuracy*: ANLS, relaxed accuracy,
                    OCRBench-style scoring, calibration (ECE) and robustness deltas.

The :mod:`docvlm_eval.pipeline` ties them together; ``scripts/evaluate.py`` is the single
entrypoint required by the task ("loads any candidate model, runs it on the benchmark,
and outputs per-model scores").
"""

__version__ = "0.1.0"

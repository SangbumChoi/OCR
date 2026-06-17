"""Evaluation pipeline: model + benchmark -> per-sample predictions + summary.

This is the engine behind ``scripts/evaluate.py``. It is deliberately model-agnostic: it
only talks to a :class:`~docvlm_eval.models.base.ModelAdapter` and a list of
:class:`~docvlm_eval.schema.Sample`, so the exact same loop evaluates every candidate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .metrics import aggregate
from .models import build_model
from .schema import Prediction, Sample


def run_evaluation(
    model_key: str,
    samples: list[Sample],
    out_dir: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_new_tokens: int = 256,
    limit: int | None = None,
    benchmark_name: str = "benchmark",
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate one model on one benchmark and write artefacts to ``out_dir``.

    Writes:
      * ``predictions.jsonl`` - one line per sample (id, prediction, confidence). Enables
        resume and post-hoc re-scoring without re-running the model.
      * ``summary.json``      - headline + sliced metrics + run metadata.
    """
    if limit:
        samples = samples[:limit]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"

    # resume: skip ids already predicted
    done: dict[str, Prediction] = {}
    if resume and pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["sample_id"]] = Prediction(**d)

    from .models.base import GenConfig

    model = build_model(
        model_key, device=device, dtype=dtype, gen=GenConfig(max_new_tokens=max_new_tokens)
    )
    t_load = time.time()
    model.load()
    load_s = time.time() - t_load

    latencies: list[float] = []
    todo = [s for s in samples if s.sample_id not in done]
    with open(pred_path, "a", encoding="utf-8") as fout:
        for s in tqdm(todo, desc=f"{model_key}/{benchmark_name}"):
            t0 = time.time()
            try:
                text, conf = model.generate(s.image_path, s.question)
            except Exception as exc:  # one bad sample must not kill the whole run
                text, conf = "", None
                print(f"[warn] {s.sample_id}: {type(exc).__name__}: {exc}")
            latencies.append(time.time() - t0)
            pred = Prediction(sample_id=s.sample_id, prediction=text, confidence=conf, raw=text)
            done[s.sample_id] = pred
            fout.write(json.dumps(pred.__dict__, ensure_ascii=False) + "\n")
            fout.flush()

    result = aggregate(samples, done)
    result["summary"].update(
        {
            "model": model_key,
            "hf_id": model.hf_id,
            "param_count_m": model.param_count_m,
            "benchmark": benchmark_name,
            "device": device,
            "dtype": dtype,
            "load_seconds": round(load_s, 2),
            "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        }
    )

    (out / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "per_sample.json").write_text(
        json.dumps(result["per_sample"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result["summary"]

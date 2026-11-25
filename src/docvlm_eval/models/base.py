"""Model adapter interface.

Every candidate VLM is wrapped in a :class:`ModelAdapter`. The pipeline only ever sees
``load()`` and ``generate()``, so models with wildly different interfaces (HF ``generate``,
GOT-OCR's ``model.chat``, Florence-2's task tokens, PaddleOCR-VL's pipeline) all look the
same to the evaluation loop.

Adding a new model = subclass this, implement ``generate``, and register it (one line).
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenConfig:
    """Decoding settings held constant across models for a fair comparison."""

    max_new_tokens: int = 256
    do_sample: bool = False  # greedy by default -> deterministic, reproducible
    temperature: float = 0.0
    num_beams: int = 1


@dataclass
class ModelAdapter(abc.ABC):
    """Base class for a candidate model.

    Subclasses set ``hf_id``/``param_count_m`` as class attributes and implement
    :meth:`load` and :meth:`generate`.
    """

    device: str = "cuda"
    dtype: str = "bfloat16"
    gen: GenConfig = field(default_factory=GenConfig)
    revision: str | None = None
    # attention backend. Default "auto" -> **eager** (the only backend every model here supports;
    # see resolve_attn). flash_attention_2 is NOT used by default — it needs an Ampere+ GPU and
    # gives no measurable win on a T4 (see notebooks/flash_attention_benchmark.ipynb, reference
    # only). Opt in explicitly with "sdpa"/"flash_attention_2" if you want to benchmark them.
    attn: str = "auto"
    _loaded: bool = field(default=False, init=False, repr=False)

    # ---- descriptive metadata (filled by subclasses; surfaced in the report) ----
    key: str = "base"
    hf_id: str = ""
    param_count_m: float = 0.0  # millions of parameters
    family: str = ""

    def __post_init__(self) -> None:
        # ``register("key")`` stores the registry key on the class; copy it onto the instance
        # so it isn't shadowed by the dataclass field default ("base").
        rk = getattr(type(self), "_registry_key", None)
        if rk:
            self.key = rk

    @abc.abstractmethod
    def load(self) -> None:
        """Download/instantiate weights. Heavy imports happen here, not at import time."""

    @abc.abstractmethod
    def generate(self, image_path: str, question: str) -> tuple[str, float | None]:
        """Return ``(answer_text, confidence)`` for one image+question.

        ``confidence`` is a sequence-level probability in [0, 1] (mean token prob), or
        ``None`` if the backend cannot expose token logprobs. It feeds calibration (ECE).
        """

    # -- shared helper: turn HF ``generate`` logprobs into one confidence scalar --
    @staticmethod
    def _confidence_from_scores(scores: Any, sequences: Any, input_len: int) -> float | None:
        """Mean token probability of the generated continuation.

        ``scores`` is the tuple returned by HF generate(..., output_scores=True,
        return_dict_in_generate=True); we softmax each step's logits and read off the
        probability of the chosen token, then average and exp back from log-space.
        """
        try:
            import torch

            gen_ids = sequences[0][input_len:]
            logprobs = []
            for step, logits in enumerate(scores):
                if step >= len(gen_ids):
                    break
                lp = torch.log_softmax(logits[0].float(), dim=-1)
                logprobs.append(lp[gen_ids[step]].item())
            if not logprobs:
                return None
            return float(math.exp(sum(logprobs) / len(logprobs)))
        except Exception:
            return None

    def resolve_attn(self) -> str:
        """Resolve ``attn='auto'`` to a concrete backend.

        Default is **eager** — the only backend every model here supports (InternVL's remote code
        rejects ``sdpa`` with a ValueError; flash_attention_2 needs the flash-attn package AND an
        Ampere+ GPU, so it is unavailable on a T4/Turing). Pass ``attn='sdpa'`` or
        ``'flash_attention_2'`` explicitly (e.g. the flash-attn benchmark) to opt in. The
        benchmark records a per-(model, backend) status so unsupported combinations are visible."""
        if self.attn != "auto":
            return self.attn
        return "eager"

    def profile(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "hf_id": self.hf_id,
            "revision": self.revision,
            "param_count_m": self.param_count_m,
            "family": self.family,
            "attn": self.resolve_attn(),
        }

    # ---- resource measurement (declared on the wrapper so every model is measured the same) --
    def reset_peak_memory(self) -> None:
        """Reset the GPU peak-memory counter before a run (no-op on CPU)."""
        try:
            import torch

            if "cuda" in str(self.device) and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def peak_gpu_mb(self) -> float | None:
        """Peak GPU memory (MB) since the last reset, or None on CPU/unavailable."""
        try:
            import torch

            if "cuda" in str(self.device) and torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 1e6, 1)
        except Exception:
            pass
        return None

    @staticmethod
    def peak_cpu_mb() -> float | None:
        """Peak resident set size (MB) of this process. With one-model-per-subprocess runs this
        is effectively the model's peak CPU memory."""
        try:
            import resource

            # ru_maxrss is KB on Linux, bytes on macOS
            import sys

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return round(rss / (1024 if sys.platform != "darwin" else 1024 * 1024), 1)
        except Exception:
            return None

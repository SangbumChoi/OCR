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
    _loaded: bool = field(default=False, init=False, repr=False)

    # ---- descriptive metadata (filled by subclasses; surfaced in the report) ----
    key: str = "base"
    hf_id: str = ""
    param_count_m: float = 0.0  # millions of parameters
    family: str = ""

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

    def profile(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "hf_id": self.hf_id,
            "param_count_m": self.param_count_m,
            "family": self.family,
        }

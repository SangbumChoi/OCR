from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _levenshtein(a: List[str], b: List[str]) -> int:
    # 메모리 절약형 DP
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    """
    Character Error Rate = edit_distance(chars) / len(ref_chars)
    ref가 빈 문자열이면 0/1 정의가 애매하므로:
      - ref=="" and pred=="" => 0
      - ref=="" and pred!="" => 1
    """
    if ref == "":
        return 0.0 if pred == "" else 1.0
    dist = _levenshtein(list(pred), list(ref))
    return dist / max(1, len(ref))


def wer(pred: str, ref: str) -> float:
    """
    Word Error Rate = edit_distance(words) / len(ref_words)
    """
    ref_words = ref.split()
    pred_words = pred.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    dist = _levenshtein(pred_words, ref_words)
    return dist / max(1, len(ref_words))


@dataclass
class MetricAverages:
    cer: float
    wer: float


def compute_ocr_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    if len(preds) != len(refs):
        raise ValueError(f"preds/refs 길이가 다릅니다: {len(preds)} vs {len(refs)}")
    if len(preds) == 0:
        return {"cer": 0.0, "wer": 0.0}

    cer_sum = 0.0
    wer_sum = 0.0
    for p, r in zip(preds, refs):
        cer_sum += cer(p, r)
        wer_sum += wer(p, r)
    return {"cer": cer_sum / len(preds), "wer": wer_sum / len(preds)}



"""Spatial-grounding metric: does the model put the box in the right place?

For the *location-understanding* axis we ask a model to return the bounding box of a named
element. Models express boxes very differently (``[x1,y1,x2,y2]``, ``(x1,y1),(x2,y2)``,
Florence-2 ``loc_<n>`` tokens, normalised 0-1 or 0-1000, pixel coords), so the parser is
permissive: it extracts the first four numbers it can find and, if they look normalised
(<=1.0 or <=1000), rescales to pixels using the image size carried in the gold.

Gold format (in ``Sample.answers[0]``): ``"x1,y1,x2,y2;W,H"`` (pixel box + image size).
Score = IoU in [0,1]; the pipeline thresholds it (>=0.5) for "correct".

Coordinate-frame caveat (Qwen-VL family, incl. Qwen3.5-VL): these models **smart-resize** the input
(preprocessor: total pixels in [shortest_edge, longest_edge], dims rounded to patch*merge) and emit
boxes in **absolute pixels of that resized image**, NOT the original. So a Qwen box is offset/scaled
vs. our original-pixel gold. To score it, rescale the predicted box from the model's processed size
back to the original with :func:`rescale_box` (the processed H/W come from the processor's
``image_grid_thw`` * patch_size). ``parse_pred_box(text, size, model_size=...)`` does this when given
the model frame.
"""

from __future__ import annotations

import re


def rescale_box(box: list[float], from_wh: tuple[int, int], to_wh: tuple[int, int]) -> list[float]:
    """Linearly map a box from one image frame to another (e.g. Qwen's smart-resized frame -> original)."""
    fw, fh = from_wh
    tw, th = to_wh
    sx, sy = (tw / fw if fw else 1.0), (th / fh if fh else 1.0)
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def parse_gold_box(gold: str) -> tuple[list[float], tuple[int, int]] | None:
    """Parse 'x1,y1,x2,y2;W,H' -> ([x1,y1,x2,y2], (W,H))."""
    try:
        box_s, size_s = gold.split(";")
        box = [float(x) for x in box_s.split(",")]
        w, h = (int(float(x)) for x in size_s.split(","))
        if len(box) == 4:
            return box, (w, h)
    except Exception:
        return None
    return None


def parse_pred_box(text: str, size: tuple[int, int],
                   model_size: tuple[int, int] | None = None) -> list[float] | None:
    """Extract a 4-number box from arbitrary model text and return pixel coords in ``size`` (the
    original-image frame the gold uses).

    ``model_size`` (the model's processed/smart-resized W,H) handles the Qwen-VL convention: the box
    is in absolute pixels of the resized image, so we rescale it from ``model_size`` to ``size``."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace("loc_", " "))
    if len(nums) < 4:
        return None
    vals = [float(n) for n in nums[:4]]
    w, h = size
    mx = max(vals)
    if mx <= 1.0:                      # normalised 0-1
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
    elif mx <= 1000 and (w > 1000 or h > 1000 or mx <= 1.0):
        vals = [vals[0] / 1000 * w, vals[1] / 1000 * h, vals[2] / 1000 * w, vals[3] / 1000 * h]
    elif model_size and model_size != size:   # absolute pixels in the model's resized frame (Qwen-VL)
        vals = rescale_box(vals, model_size, size)
    # else: assume already pixels in the original frame
    x1, y1, x2, y2 = vals
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def grounding_score(pred: str, golds: list[str]) -> float:
    """Best IoU between the predicted box and any gold box. 0 if no box is parseable."""
    best = 0.0
    for g in golds:
        parsed = parse_gold_box(g)
        if not parsed:
            continue
        gbox, size = parsed
        pbox = parse_pred_box(pred, size)
        if pbox is None:
            continue
        best = max(best, iou(pbox, gbox))
    return best

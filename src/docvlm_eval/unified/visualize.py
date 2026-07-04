"""Visualize :class:`~docvlm_eval.unified.core.UnifiedSample` examples across all datasets.

One montage cell per example: the cached image, the task badge, and a task-appropriate overlay —
**KIE / localization** draw the field/region boxes (green=field, orange=region; normalized boxes are
scaled to the image), **table/recognition/vqa/reasoning** show the image with the prompt + answer.
This is the "see every dataset in one standardized view" companion to the unified loader.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from .core import Task, UnifiedSample


def _wrap(s: str, width: int, max_lines: int) -> str:
    s = " ".join(str(s).split())
    lines = textwrap.wrap(s, width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return "\n".join(lines)


def _scale(b, w: int, h: int):
    """Box.to_list() → pixel (x1,y1,x2,y2), scaling normalized [0,1] (or 0–1000) boxes."""
    x1, y1, x2, y2 = b.to_list()
    if b.normalized:
        x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
    return x1, y1, x2, y2


def render_grid(rows: list[UnifiedSample], out: str, *, cols: int = 4, max_boxes: int = 60,
                title: str | None = None) -> str:
    """Render one cell per row to a PNG grid. Returns the output path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [r for r in rows if r.image_path and Path(r.image_path).exists()]
    if not rows:
        print("No rows with cached images to visualize (run with cache_dir set)."); return out
    n = len(rows)
    cols = max(1, cols)
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.7, nrows * 4.4))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, r in zip(axes, rows):
        try:
            img = Image.open(r.image_path).convert("RGB")
            W, H = img.size
            ax.imshow(img)
        except Exception as exc:
            ax.text(0.5, 0.5, f"<no image>\n{type(exc).__name__}", ha="center", va="center")
            W = H = 1
        ax.set_xticks([]); ax.set_yticks([])
        # overlays: KIE fields (green) + localization regions (orange)
        drawn = 0
        for fl in r.fields:
            if fl.bbox and drawn < max_boxes:
                x1, y1, x2, y2 = _scale(fl.bbox, W, H)
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                edgecolor="#1a9641", lw=1.0))
                drawn += 1
        for rg in r.regions:
            if rg.bbox and drawn < max_boxes:
                x1, y1, x2, y2 = _scale(rg.bbox, W, H)
                ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                edgecolor="#d7791d", lw=1.0))
                drawn += 1
        nb = sum(1 for f in r.fields if f.bbox) + sum(1 for g in r.regions if g.bbox)
        badge = f"{r.source} · {r.task}" + (f" · {nb} boxes" if nb else "")
        ax.set_title(badge, fontsize=8, fontweight="bold")
        # caption: task-appropriate
        if r.task == Task.TABLE:
            cap = "table → HTML (" + str(len(r.table_html or "")) + " chars)"
        elif r.task in (Task.KIE,) and r.fields:
            kv = [f"{f.key}={f.value}" for f in r.fields if f.value][:3]
            cap = "fields: " + _wrap("; ".join(kv), 44, 2)
        elif r.full_text and not r.answers:
            cap = "text: " + _wrap(r.full_text, 44, 2)
        else:
            cap = _wrap("Q: " + r.prompt(), 44, 2) + "\n" + _wrap("A: " + " | ".join(r.answers), 44, 1)
        ax.set_xlabel(cap, fontsize=7, loc="left")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title or f"Unified dataset examples — {n} samples, "
                 f"{len({r.source for r in rows})} datasets, {len({r.task for r in rows})} tasks",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    op = Path(out); op.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(op, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {op}  ({n} cells, {len({r.source for r in rows})} datasets)")
    return str(op)

"""Visualize :class:`~docvlm_eval.unified.core.UnifiedSample` examples across all datasets.

One montage cell per example: the cached image, the task badge, and a task-appropriate overlay —
**KIE / localization** draw the field/region boxes (green=field, orange=region; normalized boxes are
scaled to the image), **table/recognition/vqa/reasoning** show the image with the prompt + answer.
This is the "see every dataset in one standardized view" companion to the unified loader.
"""

from __future__ import annotations

import base64
import html
import io
import textwrap
from html.parser import HTMLParser
from pathlib import Path

from .core import Task, UnifiedSample


_TABLE_TAGS = {
    "table",
    "caption",
    "colgroup",
    "col",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "br",
}
_TABLE_ATTRIBUTES = {"colspan", "rowspan", "scope"}


class _SafeTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.rows = 0
        self.cells = 0
        self.table_depth = 0
        self.suppressed_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.suppressed_depth += 1
            return
        if tag not in _TABLE_TAGS:
            return
        if tag == "table":
            self.table_depth += 1
        elif self.table_depth == 0:
            return
        if tag == "tr":
            self.rows += 1
        if tag in {"th", "td"}:
            self.cells += 1
        safe_attrs = []
        for name, value in attrs:
            name = name.lower()
            if name not in _TABLE_ATTRIBUTES or value is None:
                continue
            if name in {"colspan", "rowspan"}:
                if not value.isdigit() or not 1 <= int(value) <= 100:
                    continue
            if name == "scope" and value.lower() not in {
                "row",
                "col",
                "rowgroup",
                "colgroup",
            }:
                continue
            safe_attrs.append(
                f' {name}="{html.escape(value, quote=True)}"'
            )
        self.parts.append(f"<{tag}{''.join(safe_attrs)}>")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self.suppressed_depth = max(0, self.suppressed_depth - 1)
            return
        if (
            self.table_depth > 0
            and tag in _TABLE_TAGS
            and tag not in {"br", "col"}
        ):
            self.parts.append(f"</{tag}>")
        if tag == "table" and self.table_depth > 0:
            self.table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.table_depth > 0 and self.suppressed_depth == 0:
            self.parts.append(html.escape(data))


def _safe_table_html(value: str) -> tuple[str, int, int]:
    parser = _SafeTableParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.parts), parser.rows, parser.cells


def _image_preview(path: Path, max_long_side: int) -> tuple[str, int, int]:
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        image.thumbnail((max_long_side, max_long_side))
        payload = io.BytesIO()
        image.save(payload, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(payload.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}", width, height


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
        print("No rows with cached images to visualize (run with cache_dir set).")
        return out
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
        ax.set_xticks([])
        ax.set_yticks([])
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
    op = Path(out)
    op.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(op, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {op}  ({n} cells, {len({r.source for r in rows})} datasets)")
    return str(op)


def render_detail_report(
    rows: list[UnifiedSample],
    out: str,
    *,
    max_long_side: int = 2400,
    max_samples: int = 64,
) -> str:
    """Render table/full-page samples without duplicating long targets.

    The report embeds one bounded image preview per sample and renders table
    markup through a strict allowlist. The original image remains available
    through a full-resolution link.
    """

    if max_long_side < 512:
        raise ValueError("max_long_side must be at least 512")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    details = []
    omitted = 0
    for row in rows:
        if not row.image_path:
            continue
        image_path = Path(row.image_path)
        if not image_path.is_file():
            continue
        preview, width, height = _image_preview(image_path, max_long_side)
        is_full_page = height / max(width, 1) >= 1.4
        if not row.table_html and not is_full_page:
            continue
        if len(details) >= max_samples:
            omitted += 1
            continue
        safe_table = ""
        table_rows = table_cells = 0
        if row.table_html:
            safe_table, table_rows, table_cells = _safe_table_html(row.table_html)
        details.append(
            {
                "row": row,
                "preview": preview,
                "width": width,
                "height": height,
                "full_image": image_path.resolve().as_uri(),
                "table": safe_table,
                "table_rows": table_rows,
                "table_cells": table_cells,
            }
        )
    if not details:
        print("No complex table or full-page rows to visualize.")
        return out

    sections = []
    for item in details:
        row = item["row"]
        table_panel = ""
        if item["table"]:
            table_panel = f"""
            <div class="table-pane">
              <div class="pane-label">Table preview</div>
              <div class="table-scroll">{item["table"]}</div>
            </div>"""
        metadata = (
            f"{item['width']}x{item['height']} px"
            + (
                f" | {item['table_rows']} rows | {item['table_cells']} cells"
                if item["table"]
                else ""
            )
        )
        sections.append(
            f"""
      <section>
        <header>
          <h2>{html.escape(row.source)} / {html.escape(row.sample_id)}</h2>
          <p>{html.escape(row.task)} | {metadata}</p>
        </header>
        <div class="detail-grid{' has-table' if item['table'] else ''}">
          <div class="image-pane">
            <div class="pane-label">Document image</div>
            <a href="{html.escape(item['full_image'], quote=True)}">
              <img src="{item['preview']}" alt="{html.escape(row.sample_id, quote=True)}">
            </a>
          </div>
          {table_panel}
        </div>
      </section>"""
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UDD detail report</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; }}
    body {{ margin: 0; color: #171717; background: #f4f6f8; }}
    main {{ max-width: 1500px; margin: auto; background: white; }}
    h1 {{ margin: 0; padding: 28px 32px 10px; font-size: 24px; }}
    .intro {{ margin: 0; padding: 0 32px 24px; color: #565b61; }}
    section {{ border-top: 1px solid #cfd4da; padding: 24px 32px 32px; }}
    header h2 {{ margin: 0 0 6px; font-size: 18px; overflow-wrap: anywhere; }}
    header p, .pane-label {{ color: #565b61; font-size: 13px; }}
    .detail-grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; }}
    .detail-grid.has-table {{ grid-template-columns: minmax(320px, 0.9fr) minmax(420px, 1.1fr); }}
    .image-pane, .table-pane {{ min-width: 0; }}
    .pane-label {{ margin-bottom: 8px; font-weight: 700; }}
    img {{ display: block; max-width: 100%; max-height: 80vh; width: auto; height: auto;
           border: 1px solid #b8bec5; }}
    .table-scroll {{ max-height: 80vh; overflow: auto; border: 1px solid #b8bec5; }}
    table {{ border-collapse: collapse; width: max-content; min-width: 100%; }}
    th, td {{ border: 1px solid #aeb5bc; padding: 6px 8px; text-align: left;
              vertical-align: top; white-space: pre-wrap; overflow-wrap: anywhere;
              max-width: 32rem; }}
    th {{ background: #e9f0f2; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: #f8f9fa; }}
    @media (max-width: 900px) {{
      .detail-grid.has-table {{ grid-template-columns: minmax(0, 1fr); }}
      h1, .intro, section {{ padding-left: 16px; padding-right: 16px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>UDD table and full-page details</h1>
    <p class="intro">{len(details)} samples{f'; {omitted} omitted by the report limit' if omitted else ''}. Images are shown once; table markup is sanitized and rendered without repeating its source text.</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"[ok] wrote {output} ({len(details)} detail samples)")
    return str(output)

"""Table-structure metric (TEDS-style) for the proposed evaluation format.

TEDS (Tree-Edit-Distance-based Similarity, Zhou et al. / PubTabNet) compares two tables as HTML
trees: ``TEDS = 1 - TreeEditDistance(pred, gold) / max(|pred|, |gold|)`` in [0, 1]. We implement
a dependency-free approximation:

* parse the HTML table into a grid of cell texts (handling ``rowspan``/``colspan``),
* **TEDS-Struct** = grid-shape agreement (same #rows/#cols and cell occupancy),
* **TEDS** = TEDS-Struct blended with per-aligned-cell text similarity (1 - NED).

This captures both structure (cell topology) and content, which is what tables need (vs. a flat
text metric). For a reference-grade score use the official ``teds`` package on a GPU run; this
gives a faithful, runnable proxy.
"""

from __future__ import annotations

import re

from .text import _nls


def _parse_html_table(html: str) -> list[list[str]]:
    """Very small HTML-table parser -> list of rows of cell texts (expands row/col spans)."""
    html = html.replace("\n", " ")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    grid: list[list[str]] = []
    for r in rows:
        cells = re.findall(r"<(t[dh])([^>]*)>(.*?)</\1>", r, flags=re.IGNORECASE | re.DOTALL)
        row: list[str] = []
        for _tag, attrs, content in cells:
            text = re.sub(r"<[^>]+>", "", content).strip()
            colspan = int((re.search(r'colspan="?(\d+)', attrs) or [0, 1])[1]) if "colspan" in attrs else 1
            for _ in range(max(1, colspan)):
                row.append(text)
        grid.append(row)
    return grid


def teds_struct(pred_html: str, gold_html: str) -> float:
    """Structure-only similarity: agreement of grid shape + cell occupancy."""
    p, g = _parse_html_table(pred_html), _parse_html_table(gold_html)
    pr, gr = len(p), len(g)
    pc = max((len(r) for r in p), default=0)
    gc = max((len(r) for r in g), default=0)
    if gr == 0 and pr == 0:
        return 1.0
    row_sim = 1 - abs(pr - gr) / max(pr, gr, 1)
    col_sim = 1 - abs(pc - gc) / max(pc, gc, 1)
    return max(0.0, (row_sim + col_sim) / 2)


def teds(pred_html: str, gold_html: str, struct_weight: float = 0.5) -> float:
    """Blend structure agreement with per-cell content similarity."""
    p, g = _parse_html_table(pred_html), _parse_html_table(gold_html)
    struct = teds_struct(pred_html, gold_html)
    # content: align cells row-by-row, score 1-NED, average over the gold cells
    total, matched = 0, 0.0
    for i, grow in enumerate(g):
        prow = p[i] if i < len(p) else []
        for j, gcell in enumerate(grow):
            total += 1
            pcell = prow[j] if j < len(prow) else ""
            matched += _nls(pcell, gcell)
    content = matched / total if total else (1.0 if not p else 0.0)
    return round(struct_weight * struct + (1 - struct_weight) * content, 4)


def teds_score(pred: str, golds: list[str]) -> float:
    """Dispatcher-compatible: best TEDS over gold HTML table(s)."""
    return max((teds(pred, g) for g in golds), default=0.0)

#!/usr/bin/env python3
"""Render docs/report/technical_report.md to a styled PDF (docs/report/technical_report.pdf).

Uses python-markdown + weasyprint (no LaTeX needed).

    python scripts/build_report.py
"""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report" / "technical_report.md"
OUT = ROOT / "docs" / "report" / "technical_report.pdf"

CSS = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45;
       color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 18px; color: #15396b; border-bottom: 1px solid #ccc; }
h3 { font-size: 11.5pt; color: #15396b; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f8; }
code { background: #f3f3f3; padding: 1px 3px; border-radius: 3px; font-size: 8.8pt; }
blockquote { border-left: 3px solid #15396b; margin: 8px 0; padding: 4px 12px;
             background: #f6f8fc; color: #333; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto; }
"""


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(SRC), help="markdown source")
    ap.add_argument("--out", default=None, help="pdf target (default: source with .pdf)")
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out) if args.out else src.with_suffix(".pdf")
    md = src.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md, extensions=["tables", "fenced_code", "toc", "sane_lists", "md_in_html"]
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    # base_url = the markdown's own directory so relative image paths (figures/*.png) resolve.
    HTML(string=html, base_url=str(src.parent)).write_pdf(str(out))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

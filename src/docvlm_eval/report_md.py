"""Pretty-print GitHub-flavored markdown tables (align columns to equal width).

Used by the result writers so the generated `.md` is readable as plain text; `scripts/
prettify_md.py` applies it to existing files too.
"""

from __future__ import annotations

import re


def _is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.strip() or "") for c in cells if c != "")


def _split(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    padded = [c.ljust(widths[i]) for i, c in enumerate(cells)] + \
             ["".ljust(widths[i]) for i in range(len(cells), len(widths))]
    return "| " + " | ".join(padded) + " |"


def _fmt_sep(aligns: list[str], widths: list[int]) -> str:
    """Separator cells fill EXACTLY widths[i] chars so they align with data cells."""
    out = []
    for i, w in enumerate(widths):
        a = aligns[i] if i < len(aligns) else "-"
        if a == ":-:":
            out.append(":" + "-" * (w - 2) + ":")
        elif a == "-:":
            out.append("-" * (w - 1) + ":")
        elif a == ":-":
            out.append(":" + "-" * (w - 1))
        else:
            out.append("-" * w)
    return "| " + " | ".join(out) + " |"


def _align_of(cell: str) -> str:
    c = cell.strip()
    left, right = c.startswith(":"), c.endswith(":")
    return ":-:" if left and right else "-:" if right else ":-" if left else "-"


def prettify_tables(md: str) -> str:
    """Re-pad every markdown table block so columns align. Non-table text is untouched."""
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        # a table = header row, separator row, then body rows (all contain '|')
        if "|" in lines[i] and i + 1 < n and "|" in lines[i + 1] and _is_sep(_split(lines[i + 1])):
            block = [lines[i], lines[i + 1]]
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                block.append(lines[j])
                j += 1
            header = _split(block[0])
            aligns = [_align_of(c) for c in _split(block[1])]
            body = [_split(r) for r in block[2:]]
            ncol = max([len(header)] + [len(r) for r in body] + [len(aligns)])
            header += [""] * (ncol - len(header))
            aligns += ["-"] * (ncol - len(aligns))
            body = [r + [""] * (ncol - len(r)) for r in body]
            # min width 3 so the separator (>=3 dashes, with optional align colons) fits and
            # data/separator columns share the exact same width -> aligned plain text
            widths = [max(3, len(header[c]), *(len(r[c]) for r in body)) if body
                      else max(3, len(header[c])) for c in range(ncol)]
            out.append(_fmt_row(header, widths))
            out.append(_fmt_sep(aligns, widths))
            out.extend(_fmt_row(r, widths) for r in body)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)

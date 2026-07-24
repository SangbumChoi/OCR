"""Versioned visual layout families for executable hard documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

HARD_LAYOUT_FAMILIES = ("classic-v1", "compact-v1", "report-v1")


@dataclass(frozen=True)
class HardLayoutSpec:
    """Page and CSS contract for one hard-document visual template."""

    family: str
    page: str
    css: str


_LAYOUTS: dict[str, dict[str, HardLayoutSpec]] = {
    "hard_table": {
        "classic-v1": HardLayoutSpec(
            "classic-v1",
            "A4",
            ".summary{border:1px solid #666;padding:8px;margin-top:12px}.num{text-align:right}",
        ),
        "compact-v1": HardLayoutSpec(
            "compact-v1",
            "A4 landscape",
            ".hard-header{display:grid;grid-template-columns:2fr 1fr;gap:14px;align-items:end;"
            "border-bottom:2px solid #345;padding-bottom:8px;margin-bottom:10px}"
            ".hard-header .summary{margin:0;border:0;background:#e8eef3;padding:8px}"
            "table{font-size:9px}.num{text-align:right}",
        ),
        "report-v1": HardLayoutSpec(
            "report-v1",
            "A4",
            ".hard-header{border-left:6px solid #2f6c9e;padding:8px 12px;background:#f2f5f8;"
            "margin-bottom:12px}.data-section{margin:8px 0 14px}"
            ".summary{width:38%;margin-left:auto;border:2px solid #2f6c9e;padding:10px}"
            "th{background:#e8eef3}.num{text-align:right}",
        ),
    },
    "hard_chart": {
        "classic-v1": HardLayoutSpec(
            "classic-v1",
            "A5",
            ".chart{height:250px;display:flex;align-items:flex-end;gap:12px;border-left:2px "
            "solid #333;border-bottom:2px solid #333;padding:14px 10px 0}"
            ".col{flex:1;text-align:center}.bar{background:#2f6c9e;color:white;display:flex;"
            "align-items:flex-start;justify-content:center;padding-top:4px;font-weight:bold}"
            ".year{font-size:9px;margin-top:5px}.note{font-size:9px}",
        ),
        "compact-v1": HardLayoutSpec(
            "compact-v1",
            "A5 landscape",
            ".chart-shell{display:grid;grid-template-columns:1fr 3fr;gap:18px;align-items:stretch}"
            ".chart-copy{border-right:2px solid #234;padding-right:12px}"
            ".chart{height:180px;display:flex;align-items:flex-end;gap:8px;border-left:2px "
            "solid #333;border-bottom:2px solid #333;padding:8px 8px 0}"
            ".col{flex:1;text-align:center}.bar{background:#387c6d;color:white;display:flex;"
            "align-items:flex-start;justify-content:center;padding-top:4px;font-weight:bold}"
            ".year{font-size:8px;margin-top:4px}.note{font-size:9px}",
        ),
        "report-v1": HardLayoutSpec(
            "report-v1",
            "A4",
            ".chart-card{border:1px solid #8996a3;padding:14px;background:#f7f9fb}"
            ".chart{height:290px;display:flex;align-items:flex-end;gap:16px;border-left:2px "
            "solid #333;border-bottom:2px solid #333;padding:16px 12px 0}"
            ".col{flex:1;text-align:center}.bar{background:#80553f;color:white;display:flex;"
            "align-items:flex-start;justify-content:center;padding-top:5px;font-weight:bold}"
            ".year{font-size:9px;margin-top:5px}.note{font-size:9px;margin-top:12px}",
        ),
    },
    "hard_investment": {
        "classic-v1": HardLayoutSpec(
            "classic-v1",
            "A5",
            ".legal{font-size:9px;color:#444;border-top:1px solid #999;margin-top:14px;"
            "padding-top:8px}",
        ),
        "compact-v1": HardLayoutSpec(
            "compact-v1",
            "A5 landscape",
            ".disclosure-grid{display:grid;grid-template-columns:3fr 1fr;gap:16px;"
            "align-items:start}.legal{font-size:8px;color:#444;border-left:3px solid #607d8b;"
            "padding:10px;margin:0;background:#f2f5f6}th{background:#e3ebef}",
        ),
        "report-v1": HardLayoutSpec(
            "report-v1",
            "A4",
            ".legal{font-size:9px;color:#333;border:1px solid #9b875f;background:#faf6ea;"
            "padding:10px;margin-bottom:12px}.ownership-card{border-top:4px solid #9b875f;"
            "padding-top:12px}th{background:#f0eadc}",
        ),
    },
    "hard_science": {
        "classic-v1": HardLayoutSpec(
            "classic-v1",
            "A4",
            ".authors{font-size:9px;text-align:center}.abstract{columns:2;column-gap:18px;"
            "font-size:9px;text-align:justify}.equation{text-align:center;font-family:serif;"
            "margin:12px}.caption{font-size:8px;color:#444}",
        ),
        "compact-v1": HardLayoutSpec(
            "compact-v1",
            "A4 landscape",
            ".paper-grid{display:grid;grid-template-columns:1fr 1.35fr;gap:22px;"
            "align-items:start}.paper-intro{border-right:1px solid #999;padding-right:18px}"
            ".authors{font-size:9px}.abstract{font-size:9px;text-align:justify}"
            ".equation{text-align:center;font-family:serif;margin:8px}"
            ".caption{font-size:8px;color:#444}table{font-size:9px}",
        ),
        "report-v1": HardLayoutSpec(
            "report-v1",
            "A4",
            ".authors{font-size:9px;border-bottom:1px solid #777;padding-bottom:8px}"
            ".results-card{border:1px solid #8293a3;padding:12px;background:#f7f9fb}"
            ".abstract{font-size:9px;text-align:justify;margin-top:14px}"
            ".equation{text-align:center;font-family:serif;margin:10px}"
            ".caption{font-size:8px;color:#444}th{background:#e8eef3}",
        ),
    },
}


def hard_layout_spec(document_family: str, layout_family: str) -> HardLayoutSpec:
    """Resolve a supported versioned layout for a hard document family."""
    if layout_family not in HARD_LAYOUT_FAMILIES:
        raise ValueError(
            f"unknown hard layout {layout_family!r}; choose from {HARD_LAYOUT_FAMILIES}"
        )
    try:
        return _LAYOUTS[document_family][layout_family]
    except KeyError as error:
        raise ValueError(f"unknown hard document family {document_family!r}") from error


def layout_fingerprint(document_family: str, layout_family: str) -> str:
    """Hash the versioned visual template identity independently of document values."""
    if layout_family not in HARD_LAYOUT_FAMILIES:
        raise ValueError(
            f"unknown hard layout {layout_family!r}; choose from {HARD_LAYOUT_FAMILIES}"
        )
    material = f"hard-layout:{document_family}:{layout_family}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()

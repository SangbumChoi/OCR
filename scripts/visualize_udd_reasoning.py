#!/usr/bin/env python
"""Render example panels for the derived UDD rationales and instruction/answer variants.

Each panel shows the real UDD image with the rationale's evidence drawn on it (evidence box,
matched OCR words / UI elements, counted layout regions), and next to it the QA, the gold, the
rationale with its provenance tier, and the diversified (instruction -> target) variants.

    python scripts/derive_udd_reasoning.py --from-hub          # produces examples/udd_reasoning/
    python scripts/visualize_udd_reasoning.py                  # fetches only the needed images

Images are read from the Hub parquet shards row-group by row-group (only the groups holding the
chosen samples). Panels go to docs/report/figures/udd_reasoning/.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.unified.rationale import derive

HUB_FILES = [f"datasets/danelcsb/UDD/data/train-0000{i}-of-00005.parquet" for i in range(5)]
DEFAULT_PICKS = {
    "tatqa": "tatqa_0001_0", "cord": "cord_0004_0", "funsd": "funsd_0002_0",
    "ocrvqa": "ocrvqa_0000_0", "screenqa": "screenqa_0000_0", "pubtabnet": "pubtabnet_0001_0",
    "omnidocbench": "omnidocbench_0019_0", "doclaynet": "doclaynet_0000_0",
    "publaynet": "publaynet_0002_0", "hallusionbench_kept": "hallusionbench_0000_0",
    "hallusionbench_verdict_stripped": "hallusionbench_0021_0",
    "hallusionbench_rejected": "hallusionbench_0007_0", "dvqa": "dvqa_0000_0", "ai2d": "ai2d_0001_0",
}
TIER_COLOR = {"source": (124, 58, 237), "source_verified": (21, 128, 61), "derived_exact": (29, 78, 216),
              "evidence_match": (194, 65, 12), "rejected": (185, 28, 28), "repaired": (100, 100, 100)}
QA_COLORS = [(220, 38, 38), (37, 99, 235), (22, 163, 74), (217, 119, 6), (147, 51, 234), (8, 145, 178)]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_CJK = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]


def fetch(sample_ids: set[str]) -> dict[str, dict]:
    """sample_id -> {image (PIL), elements_json, full_annotation}, reading only needed row groups."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from PIL import Image
    fs, out = HfFileSystem(), {}
    for p in HUB_FILES:
        if len(out) == len(sample_ids):
            break
        with fs.open(p, "rb", block_size=8 * 2**20) as f:
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                ids = pf.read_row_group(g, columns=["sample_id"]).column("sample_id").to_pylist()
                hit = [i for i, s in enumerate(ids) if s in sample_ids and s not in out]
                if not hit:
                    continue
                t = pf.read_row_group(g, columns=["sample_id", "image", "elements_json", "full_annotation"])
                for i in hit:
                    rec = {c: t.column(c)[i].as_py() for c in t.column_names}
                    img = rec["image"]
                    rec["image"] = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB")
                    out[rec["sample_id"]] = rec
                print(f"[fetch] {p.split('/')[-1]} group {g}: {len(hit)} sample(s)")
    return out


def _font(size, bold=False, text=""):
    from PIL import ImageFont
    if any(ord(c) > 0x2E80 for c in text):
        for fp in FONT_CJK:
            if Path(fp).exists():
                return ImageFont.truetype(fp, size)
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def _wrap(text: str, width: int) -> list[str]:
    out = []
    for para in str(text).split("\n"):
        out += textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False) or [""]
    return out


def _evidence_boxes(full_row: dict, qa_index: int | None, rat) -> list[list[float]]:
    """Normalized boxes to draw for one rationale: its own box + elements it cites (text or class)."""
    boxes = [rat.box] if rat is not None and rat.box else []
    if rat is None:
        return boxes
    ev = {str(e).strip().lower() for e in rat.evidence if str(e).strip()}
    for e in json.loads(full_row.get("elements_json") or "[]"):
        b = e.get("bbox")
        if not b or len(b) < 5 or not b[4]:
            continue                              # pixel boxes are not trusted (see rationale._box_of)
        info, cls = (e.get("information") or "").strip().lower(), (e.get("class") or "").strip().lower()
        inside = rat.box is None or (b[0] >= rat.box[0] - 1e-3 and b[1] >= rat.box[1] - 1e-3 and
                                     b[2] <= rat.box[2] + 1e-3 and b[3] <= rat.box[3] + 1e-3)
        if (rat.op == "count" and cls in ev) or (rat.op != "count" and info and info in ev and inside):
            boxes.append([float(v) for v in b[:4]])
    return boxes


def render_panel(key: str, row: dict, full_row: dict, variants: list[dict], out_path: Path,
                 max_qas: int = 3) -> Path:
    from PIL import Image, ImageDraw
    img = full_row["image"]
    scale = min(620 / img.width, 760 / img.height, 2.0)
    im = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    draw = ImageDraw.Draw(im, "RGBA")
    rats = {}
    merged = {**row, "elements_json": full_row.get("elements_json"), "full_annotation": full_row.get("full_annotation")}
    # re-derive on the ORIGINAL row (answers before repair) to recover each rationale's evidence
    for r in derive({**merged, "answers": row.get("_orig_answers", row["answers"])}):
        idx = r.qa_index if not r.is_new else next(
            (i for i, q in enumerate(row["instructions"]) if q == r.question), None)
        if idx is not None and idx not in rats:
            rats[idx] = r
    shown = [i for i, p in enumerate(row["reasoning_provenance"]) if p and p != "repaired"][:max_qas] or [0]
    for k, i in enumerate(shown):
        col = QA_COLORS[k % len(QA_COLORS)]
        for b in _evidence_boxes(merged, i, rats.get(i)):
            x1, y1, x2, y2 = b[0] * im.width, b[1] * im.height, b[2] * im.width, b[3] * im.height
            draw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3, fill=col + (40,))
            draw.text((x1 + 2, max(0, y1 - 13)), f"Q{i}", fill=col + (255,), font=_font(12, True))

    W_TEXT = 860
    lines: list[tuple[str, tuple, int, bool]] = []      # (text, color, size, bold)

    def add(text, color=(30, 30, 30), size=15, bold=False, width=92):
        for ln in _wrap(text, width):
            lines.append((ln, color, size, bold))

    add(f"{key}  ·  {row['source']}  ·  {row['sample_id']}", (20, 20, 20), 19, True, 70)
    add(f"task: {row.get('task')}   rows QAs: {len(row['instructions'])}", (110, 110, 110), 13)
    lines.append(("", (0, 0, 0), 8, False))
    for k, i in enumerate(shown):
        col = QA_COLORS[k % len(QA_COLORS)]
        prov = row["reasoning_provenance"][i] or "answer-only"
        tier = prov.split(":", 1)[0]
        derived = (row.get("task_detail") or [""] * (i + 1))[i].startswith("derived:")
        q = row["instructions"][i]
        q = q if len(q) <= 420 else "…" + q[-420:]
        add(f"Q{i}{' (derived QA)' if derived else ''}: {q}", col, 15, True)
        orig = row.get("_orig_answers", row["answers"])
        if i < len(orig) and orig[i] != row["answers"][i]:
            add(f"gold (source, corrupted): {orig[i]!r}", (150, 150, 150), 14)
            add(f"gold (repaired): {row['answers'][i]}", (21, 128, 61), 15, True)
        else:
            add(f"gold: {row['answers'][i][:300]}", (20, 20, 20), 15, True)
        if prov.startswith("rejected:"):
            rej = (row.get("reasoning_rejected") or [""] * (i + 1))[i]
            add(f"rationale REJECTED by the gate ({prov.split(':', 1)[1]}): {rej}", TIER_COLOR["rejected"], 14)
        elif row["reasoning"][i]:
            add(f"rationale [{prov}]: {row['reasoning'][i]}", TIER_COLOR.get(tier, (60, 60, 60)), 14)
        vs = [v for v in variants if v["meta"]["qa_index"] == i and v["meta"]["format"] != "original"]
        for v in vs[:3]:
            ins = v["question"].replace("\n", " ⏎ ")
            ins = ins if len(ins) <= 160 else "…" + ins[-160:]
            tgt = v["answers"][0].replace("\n", " ⏎ ")
            add(f"   ▸ {v['meta']['format']:9s} {ins}", (70, 70, 70), 13, False, 104)
            add(f"       → {tgt[:260]}", (15, 15, 15), 13, True, 104)
        lines.append(("", (0, 0, 0), 10, False))

    text_h = sum(int(sz * 1.45) for _, _, sz, _ in lines) + 30
    H = max(im.height + 30, text_h)
    panel = Image.new("RGB", (im.width + W_TEXT + 40, H), (250, 250, 248))
    panel.paste(im, (15, 15))
    d = ImageDraw.Draw(panel)
    d.rectangle([14, 14, 15 + im.width, 15 + im.height], outline=(200, 200, 200))
    y = 15
    for text, color, size, bold in lines:
        d.text((im.width + 35, y), text, fill=color, font=_font(size, bold, text))
        y += int(size * 1.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=85)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoning-dir", default=str(ROOT / "examples" / "udd_reasoning"))
    ap.add_argument("--source-rows", help="UDD rows BEFORE derivation (json/jsonl) for the original "
                                          "golds; default: re-read from the Hub with the images")
    ap.add_argument("--out", default=str(ROOT / "docs" / "report" / "figures" / "udd_reasoning"))
    ap.add_argument("--pick", nargs="*", help="key=sample_id pairs (default: one per source)")
    args = ap.parse_args(argv)

    picks = dict(p.split("=", 1) for p in args.pick) if args.pick else DEFAULT_PICKS
    ids = set(picks.values())
    rows, variants = {}, defaultdict(list)
    with open(Path(args.reasoning_dir) / "rows.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] in ids:
                rows[r["sample_id"]] = r
    with open(Path(args.reasoning_dir) / "variants.jsonl", encoding="utf-8") as f:
        for line in f:
            v = json.loads(line)
            if v["udd_sample_id"] in ids:
                variants[v["udd_sample_id"]].append(v)
    if args.source_rows:
        p = Path(args.source_rows)
        src = json.loads(p.read_text()) if p.suffix == ".json" else [json.loads(x) for x in p.read_text().splitlines()]
        for s in src:
            if s["sample_id"] in rows:
                rows[s["sample_id"]]["_orig_answers"] = s["answers"]
    full = fetch(ids)
    written = []
    for key, sid in picks.items():
        if sid not in rows or sid not in full:
            print(f"[skip] {key}: {sid} not found")
            continue
        written.append(render_panel(key, rows[sid], full[sid], variants[sid], Path(args.out) / f"{key}.jpg"))
        print(f"[ok] {written[-1].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

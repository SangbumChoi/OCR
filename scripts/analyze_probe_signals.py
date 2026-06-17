#!/usr/bin/env python3
"""Compute the shortcut-robust SIGNAL criteria from spatial/context probe results.

Reads results/<model>/<probe>/per_sample.json and reports, per model, whether each capability
PASSES its *control-aware* criterion (see report/spatial_context_probes.md) — not just raw
accuracy. Prints a table and writes results/probe_signals.json.

    python scripts/analyze_probe_signals.py --probe spatial_context_probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docvlm_eval.metrics.grounding import parse_gold_box, parse_pred_box  # noqa: E402


def _by_id(rows):
    return {r["sample_id"]: r for r in rows}


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def analyze_model(rows: list[dict]) -> dict:
    r = _by_id(rows)
    out = {}

    # S1 quadrant: >=3/4 correct and not all-constant prediction
    quad = [v for k, v in r.items() if k.startswith("sp_quad_")]
    if quad:
        correct = sum(1 for v in quad if v["score"] >= 0.5)
        preds = {str(v["prediction"]).strip().lower()[:12] for v in quad}
        out["S1_absolute_spatial"] = {
            "acc": round(correct / len(quad), 2), "n": len(quad),
            "distinct_preds": len(preds),
            "PASS": correct >= 3 and len(preds) >= 2,
        }

    # S2 relative position with counterfactual control
    n_, c_ = r.get("sp_relpos_normal"), r.get("sp_relpos_counterfactual")
    if n_ and c_:
        an, ac = n_["score"] >= 0.5, c_["score"] >= 0.5
        out["S2_relative_spatial"] = {
            "normal_correct": an, "counterfactual_correct": ac,
            "prior_reliance_gap": round(float(an) - float(ac), 2),
            "PASS": an and ac,  # must beat the language prior
        }

    # S3 box tracking: center-y correlation across positions + mean IoU
    boxes = [r.get(f"sp_box_{t}") for t in ("top", "mid", "bot")]
    boxes = [b for b in boxes if b]
    if len(boxes) >= 2:
        true_cy, pred_cy, ious = [], [], []
        for b in boxes:
            gb = parse_gold_box(b["answers"][0])
            if not gb:
                continue
            gbox, size = gb
            true_cy.append((gbox[1] + gbox[3]) / 2)
            pbox = parse_pred_box(str(b["prediction"]), size)
            pred_cy.append((pbox[1] + pbox[3]) / 2 if pbox else 0.0)
            ious.append(b["score"])
        rcorr = _pearson(true_cy, pred_cy)
        out["S3_box_tracking"] = {
            "center_y_corr": round(rcorr, 2), "mean_iou": round(sum(ious) / len(ious), 2),
            "PASS": rcorr > 0.8 and (sum(ious) / len(ious)) > 0.3,
        }

    # C1 consistency: must catch the inconsistent case
    cc, ci = r.get("ctx_consistency_consistent"), r.get("ctx_consistency_inconsistent")
    if cc and ci:
        out["C1_consistency"] = {
            "consistent_correct": cc["score"] >= 0.5,
            "inconsistent_caught": ci["score"] >= 0.5,
            "PASS": ci["score"] >= 0.5,  # catching the error is the real test
        }

    # C2 anti-hallucination on absent field
    ab = r.get("ctx_absence")
    if ab:
        out["C2_anti_hallucination"] = {
            "answered_none": ab["score"] >= 0.5,
            "prediction": str(ab["prediction"])[:30], "PASS": ab["score"] >= 0.5,
        }

    # C3 distractor disambiguation
    di = r.get("ctx_distractor")
    if di:
        out["C3_disambiguation"] = {"score": di["score"], "PASS": di["score"] >= 0.5}

    # C4 cross-reference counterfactual sensitivity (both variants correct)
    xb, xa = r.get("ctx_xref_bob"), r.get("ctx_xref_alice")
    if xb and xa:
        out["C4_context_sensitivity"] = {
            "bob_correct": xb["score"] >= 0.5, "alice_correct": xa["score"] >= 0.5,
            "PASS": xb["score"] >= 0.5 and xa["score"] >= 0.5,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", default="spatial_context_probe")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out = {}
    for ps in sorted(results_dir.glob(f"*/{args.probe}/per_sample.json")):
        model = ps.parent.parent.name
        out[model] = analyze_model(json.loads(ps.read_text()))

    if not out:
        raise SystemExit(f"No results for probe '{args.probe}'. Run run_matrix first.")

    crits = ["S1_absolute_spatial", "S2_relative_spatial", "S3_box_tracking",
             "C1_consistency", "C2_anti_hallucination", "C3_disambiguation", "C4_context_sensitivity"]
    print(f"\nSignal criteria (PASS = clears the shortcut control) — probe: {args.probe}\n")
    hdr = ["model"] + [c.split("_", 1)[0] for c in crits]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for m, res in out.items():
        row = [m] + ["✅" if res.get(c, {}).get("PASS") else ("—" if c not in res else "❌") for c in crits]
        print("| " + " | ".join(row) + " |")

    (results_dir / "probe_signals.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[done] details -> {results_dir/'probe_signals.json'}")


if __name__ == "__main__":
    main()

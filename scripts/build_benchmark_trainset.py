#!/usr/bin/env python3
"""Build a small, uniform TRAINING set from every public benchmark in the catalog.

For each catalog entry with an ``hf_id`` we stream a *subset* (default 50, hard-capped < 200 images),
convert each record into our canonical :class:`~docvlm_eval.schema.Sample` DTO via
``docvlm_eval.benchmarks.trainset.extract_qa``, cache the (downscaled) image offline, and write:

  <out>/images/<key>/NNNN.jpg          cached images (downscaled, JPEG)
  <out>/per_bench/<key>.jsonl          per-benchmark Samples (absolute image_path)
  <out>/train.jsonl                    MERGED Samples, ready for run_ablation / lora_vlm
  <out>/metadata.jsonl                 HF *imagefolder* manifest (relative file_name) -> push_to_hub
  <out>/summary.json                   per-benchmark counts + failures

Why offline + HF packaging: streaming 20+ datasets every run is slow and flaky. Build ONCE, commit
the small artifact, or upload it as a brand-new HF dataset and thereafter ``load_dataset`` it in
seconds. The ``metadata.jsonl`` layout is exactly what ``load_dataset("imagefolder", data_dir=<out>)``
and ``huggingface-cli upload`` expect, so the user can publish it directly.

Examples
--------
    # build 50 images/benchmark from all streamable datasets
    python scripts/build_benchmark_trainset.py --per-bench 50

    # only a few, more per benchmark, then push to a new HF repo
    python scripts/build_benchmark_trainset.py --only docvqa,chartqa,cord --per-bench 150 \
        --push-to-hub <user>/docvlm-benchmark-trainset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docvlm_eval.benchmarks.catalog import find_image, json_safe, load_catalog  # noqa: E402
from docvlm_eval.benchmarks.loaders import save_jsonl  # noqa: E402
from docvlm_eval.benchmarks.trainset import extract_qa  # noqa: E402
from docvlm_eval.schema import Sample  # noqa: E402

HARD_CAP = 200  # "less than 200 images" per benchmark, by request

# Some catalog splits are eval-only; for *training* we can override to a split that carries answers
# (eval still uses the catalog split). Empty for now — ST-VQA's only HF split (test) has no GT, so it
# stays a documented skip. Add entries here if a benchmark exposes a labelled train split.
TRAIN_SPLIT_OVERRIDE: dict[str, str] = {}


def _downscale(img, max_px: int):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))))
    return img


def build_one(e: dict, out: Path, per_bench: int, max_px: int, quality: int,
              max_scan: int) -> tuple[list[Sample], str]:
    """Stream one benchmark -> list[Sample] (one per derived QA). Returns (samples, status)."""
    key = e["key"]
    if not e.get("hf_id"):
        return [], "no-hf-id"

    from datasets import load_dataset

    split = TRAIN_SPLIT_OVERRIDE.get(key, e["split"])
    try:
        ds = load_dataset(e["hf_id"], e.get("config"), split=split, streaming=True)
    except Exception as exc:
        print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:140]}")
        return [], "load-fail"

    img_dir = out / "images" / key
    samples: list[Sample] = []
    seen: set[str] = set()
    n_img = 0
    try:
        for scanned, ex in enumerate(ds):
            if n_img >= per_bench or scanned >= max_scan:
                break
            ex = dict(ex)
            img = find_image(ex)
            if img is None:
                continue
            qas = extract_qa(key, ex, e)
            if not qas:
                continue
            small = _downscale(img, max_px)
            h = hashlib.md5(small.tobytes()).hexdigest()
            if h in seen:                          # these sets often repeat one image per question
                continue
            seen.add(h)
            img_dir.mkdir(parents=True, exist_ok=True)
            fn = f"{n_img:04d}.jpg"
            small.save(img_dir / fn, quality=quality)
            rel = f"images/{key}/{fn}"
            for qi, qa in enumerate(qas):
                samples.append(Sample(
                    sample_id=f"{key}_{n_img:04d}_{qi}",
                    image_path=str(img_dir / fn),
                    question=qa["question"],
                    answers=qa["answers"],
                    answer_type=qa.get("answer_type", key),
                    metric=qa.get("metric", "anls"),
                    meta={"benchmark": key, "category": e.get("category", "-"),
                          "hf_id": e.get("hf_id"), "file_name": rel,
                          "raw": json_safe(ex)},
                ))
            n_img += 1
    except Exception as exc:
        print(f"[warn] {key}: stopped after {n_img} images ({type(exc).__name__}: {str(exc)[:80]})")

    if not samples:
        return [], "no-trainable"
    print(f"[ok]   {key}: {n_img} images -> {len(samples)} samples")
    return samples, "ok"


def write_hf_manifest(all_samples: list[Sample], out: Path) -> None:
    """Write an HF imagefolder ``metadata.jsonl`` (file_name relative to <out>).

    This lets the user upload the raw folder as-is with
    ``huggingface-cli upload --repo-type dataset <repo> <out> .`` (no datasets load needed)."""
    lines = []
    for s in all_samples:
        rel = s.meta.get("file_name")
        if not rel:
            continue
        lines.append(json.dumps({
            "file_name": rel, "sample_id": s.sample_id, "question": s.question,
            "answers": s.answers, "answer_type": s.answer_type, "metric": s.metric,
            "benchmark": s.meta.get("benchmark"), "category": s.meta.get("category"),
            "hf_id": s.meta.get("hf_id"),
        }, ensure_ascii=False))
    (out / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_hf_dataset(all_samples: list[Sample]):
    """Build an explicit ``datasets.Dataset`` (image column cast to the HF Image feature).

    Explicit construction is version-robust — unlike ``load_dataset('imagefolder', ...)`` auto-
    discovery, which is brittle with nested per-benchmark subfolders."""
    from datasets import Dataset, Features, Image as HFImage, Value

    recs = [{"image": s.image_path, "sample_id": s.sample_id, "question": s.question,
             "answers": s.answers, "answer_type": s.answer_type, "metric": s.metric,
             "benchmark": s.meta.get("benchmark"), "category": s.meta.get("category"),
             "hf_id": s.meta.get("hf_id")} for s in all_samples]
    feat = Features({"image": HFImage(), "sample_id": Value("string"), "question": Value("string"),
                     "answers": [Value("string")], "answer_type": Value("string"),
                     "metric": Value("string"), "benchmark": Value("string"),
                     "category": Value("string"), "hf_id": Value("string")})
    return Dataset.from_list(recs, features=feat)


def push_to_hub(all_samples: list[Sample], repo: str, token: str | None, private: bool) -> None:
    """Build the dataset explicitly and push it as a new HF dataset."""
    ds = build_hf_dataset(all_samples)
    print(f"[hub] pushing {len(ds)} rows -> {repo} (private={private})")
    ds.push_to_hub(repo, token=token, private=private)
    print(f"[hub] done: https://huggingface.co/datasets/{repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", default=None, help="benchmark_catalog.yaml (default: auto-locate)")
    p.add_argument("--out", default=str(ROOT / "data" / "benchmark_trainset"))
    p.add_argument("--per-bench", type=int, default=50,
                   help=f"images per benchmark (hard-capped at {HARD_CAP - 1})")
    p.add_argument("--only", default=None, help="comma-separated benchmark keys to include")
    p.add_argument("--skip", default=None, help="comma-separated benchmark keys to skip")
    p.add_argument("--max-px", type=int, default=1000, help="downscale longest side to this")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality for cached images")
    p.add_argument("--max-scan", type=int, default=3000,
                   help="max records streamed per benchmark (bounds cost of sparse-image sets)")
    p.add_argument("--no-hf-disk", action="store_true", help="skip writing the on-disk Arrow dataset")
    p.add_argument("--push-to-hub", default=None, help="HF repo id to publish the offline set to")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--private", action="store_true", help="push the HF dataset as private")
    args = p.parse_args()

    per_bench = max(1, min(args.per_bench, HARD_CAP - 1))
    if per_bench != args.per_bench:
        print(f"[note] clamped --per-bench {args.per_bench} -> {per_bench} (< {HARD_CAP})")

    catalog = load_catalog(args.catalog)
    only = {k.strip() for k in args.only.split(",")} if args.only else None
    skip = {k.strip() for k in args.skip.split(",")} if args.skip else set()
    entries = [e for e in catalog if e.get("hf_id")
               and (only is None or e["key"] in only) and e["key"] not in skip]
    if not entries:
        print("No streamable (hf_id) benchmarks matched the filter."); return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_bench").mkdir(exist_ok=True)

    all_samples: list[Sample] = []
    summary: dict[str, dict] = {}
    print(f"Building <{per_bench} images/benchmark from {len(entries)} datasets -> {out}\n")
    for e in entries:
        key = e["key"]
        samples, status = build_one(e, out, per_bench, args.max_px, args.quality, args.max_scan)
        summary[key] = {"status": status, "images": len({s.image_path for s in samples}),
                        "samples": len(samples), "hf_id": e.get("hf_id")}
        if samples:
            save_jsonl(samples, out / "per_bench" / f"{key}.jsonl")
            all_samples.extend(samples)

    save_jsonl(all_samples, out / "train.jsonl")
    write_hf_manifest(all_samples, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    # version-robust on-disk HF dataset (Arrow) the user can load_from_disk(...).push_to_hub(...)
    if not args.no_hf_disk:
        try:
            ds = build_hf_dataset(all_samples)
            ds.save_to_disk(str(out / "hf_dataset"))
        except Exception as exc:
            print(f"[hf] save_to_disk skipped ({type(exc).__name__}: {str(exc)[:100]})")

    ok = [k for k, v in summary.items() if v["status"] == "ok"]
    bad = {k: v["status"] for k, v in summary.items() if v["status"] != "ok"}
    print(f"\n=== built {len(all_samples)} samples from {len(ok)}/{len(entries)} benchmarks "
          f"({len({s.image_path for s in all_samples})} images) ===")
    print(f"  merged train : {out/'train.jsonl'}   (run_ablation / lora_vlm consume this directly)")
    print(f"  HF dataset   : {out/'hf_dataset'}     (load_from_disk(...).push_to_hub('<repo>'))")
    print(f"  HF raw folder: {out/'metadata.jsonl'} (huggingface-cli upload --repo-type dataset ...)")
    if bad:
        print(f"  unavailable  : {bad}")

    if args.push_to_hub:
        try:
            push_to_hub(all_samples, args.push_to_hub, args.hf_token, args.private)
        except Exception as exc:
            print(f"[hub] push failed ({type(exc).__name__}: {str(exc)[:140]}).")
            print(f"[hub] the offline set is ready at {out}; upload manually with either:")
            print(f"      huggingface-cli upload --repo-type dataset {args.push_to_hub} {out} .")
            print(f"      python -c \"from datasets import load_from_disk as L; "
                  f"L('{out/'hf_dataset'}').push_to_hub('{args.push_to_hub}')\"")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # avoid pyarrow/threading teardown abort

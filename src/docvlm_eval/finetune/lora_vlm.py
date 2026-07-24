"""Model-agnostic LoRA fine-tuning for the chat-template VLMs we adopt as Part-2 bases
(Qwen3.5-0.8B, LFM2.5-VL-1.6B). Unlike the DeepSeek-OCR scaffold next to it, this path loads any
``AutoModelForImageTextToText`` + ``AutoProcessor`` and trains on the synthetic
``realistic_cases`` DocSample data, with the **A5 LoRA placement** resolved by *introspecting the
loaded model* — so we never hardcode per-architecture module names.

The placement resolver is the piece worth unit-testing (it is pure); the training loop is GPU-only
and intentionally small (peft + a plain torch loop). Used by ``scripts/run_ablation.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

# capability -> module bucket (mirrors the hypotheses in research_novelty.md / ablation_plan.md)
PLACEMENT_GROUPS = (
    "vision",
    "connector",
    "vision_connector",
    "llm_attn",
    "llm_mlp",
    "all",
)

_VISION_PAT = re.compile(r"vis(ual|ion)|patch|siglip|navit|aimv2", re.I)
_CONNECTOR_PAT = re.compile(r"merger|projector|connector|mlp1|multi_modal|abstractor|resampler", re.I)
_ATTN_PATH = re.compile(r"(^|\.)(self_?attn|attention|attn)(\.|$)", re.I)
_ATTN_LEAF = re.compile(r"(^|\.)(q|k|v|o|qkv|out|wq|wk|wv|wo)_?proj$", re.I)
_MLP_PATH = re.compile(r"(^|\.)(mlp|feed_?forward|ffn)(\.|$)", re.I)
_MLP_LEAF = re.compile(
    r"(^|\.)(gate|up|down)_?proj$|(^|\.)(fc1|fc2|w1|w2|w3)$",
    re.I,
)


def _bucket(name: str) -> str:
    """Classify a module path into vision / connector / llm / other (by path, not leaf)."""
    if _CONNECTOR_PAT.search(name):
        return "connector"
    if _VISION_PAT.search(name):
        return "vision"
    return "llm"            # default: language-model side (language_model.* / model.layers.* / …)


def resolve_lora_targets(
    named_modules: Iterable[tuple[str, Any]],
    group: str,
    *,
    is_linear: Callable[[Any], bool] | None = None,
) -> list[str]:
    """Full module names of the Linear layers belonging to a placement ``group``.

    ``named_modules`` is ``model.named_modules()``; ``is_linear`` decides what counts as adaptable
    (defaults to ``isinstance(m, torch.nn.Linear)``). Returns **full paths** so peft adapts only the
    chosen group (e.g. LLM ``q_proj`` but not the vision tower's). This introspection is what makes
    A5 work across Qwen3.5-VL, LFM2-VL, … with no per-model name tables."""
    if group not in PLACEMENT_GROUPS:
        raise ValueError(f"unknown placement group {group!r}; choose from {PLACEMENT_GROUPS}")
    if is_linear is None:
        import torch
        is_linear = lambda m: isinstance(m, torch.nn.Linear)  # noqa: E731

    targets: list[str] = []
    for name, mod in named_modules:
        if not is_linear(mod) or not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        bkt = _bucket(name)
        if group == "all":
            targets.append(name)
        elif group == "vision" and bkt == "vision":
            targets.append(name)
        elif group == "connector" and bkt == "connector":
            targets.append(name)
        elif group == "vision_connector" and bkt in {"vision", "connector"}:
            targets.append(name)
        elif (
            group == "llm_attn"
            and bkt == "llm"
            and _ATTN_PATH.search(name)
            and _ATTN_LEAF.search(leaf)
        ):
            targets.append(name)
        elif (
            group == "llm_mlp"
            and bkt == "llm"
            and (_MLP_PATH.search(name) or _MLP_LEAF.search(leaf))
        ):
            targets.append(name)
    return sorted(set(targets))


def lora_parameter_coefficient(
    named_modules: Iterable[tuple[str, Any]],
    targets: Iterable[str],
) -> int:
    """Return LoRA trainable parameters per rank for selected Linear modules."""
    modules = dict(named_modules)
    coefficient = 0
    for name in targets:
        module = modules.get(name)
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if (
            not isinstance(in_features, int)
            or not isinstance(out_features, int)
            or in_features <= 0
            or out_features <= 0
        ):
            raise ValueError(
                f"LoRA target {name!r} does not expose positive "
                "in_features/out_features"
            )
        coefficient += in_features + out_features
    if coefficient <= 0:
        raise ValueError("LoRA parameter coefficient requires at least one target")
    return coefficient


def resolve_lora_budget(
    named_modules: Iterable[tuple[str, Any]],
    placement: str,
    *,
    requested_rank: int,
    requested_alpha: int,
    reference_placement: str | None = None,
    is_linear: Callable[[Any], bool] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Resolve targets and optionally match their adapter size to a reference placement."""
    if requested_rank <= 0 or requested_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    named_modules = list(named_modules)
    targets = resolve_lora_targets(
        named_modules,
        placement,
        is_linear=is_linear,
    )
    if not targets:
        return targets, {}
    coefficient = lora_parameter_coefficient(named_modules, targets)
    reference_targets = (
        targets
        if reference_placement is None
        else resolve_lora_targets(
            named_modules,
            reference_placement,
            is_linear=is_linear,
        )
    )
    if not reference_targets:
        raise ValueError(
            f"reference placement {reference_placement!r} has no LoRA targets"
        )
    reference_coefficient = lora_parameter_coefficient(
        named_modules,
        reference_targets,
    )
    target_budget = reference_coefficient * requested_rank
    effective_rank = max(1, round(target_budget / coefficient))
    alpha_ratio = requested_alpha / requested_rank
    effective_alpha = max(1, round(alpha_ratio * effective_rank))
    actual_parameters = coefficient * effective_rank
    relative_error = abs(actual_parameters - target_budget) / target_budget
    return targets, {
        "placement": placement,
        "reference_placement": reference_placement or placement,
        "target_count": len(targets),
        "reference_target_count": len(reference_targets),
        "requested_rank": requested_rank,
        "effective_rank": effective_rank,
        "requested_alpha": requested_alpha,
        "effective_alpha": effective_alpha,
        "parameters_per_rank": coefficient,
        "reference_parameters_per_rank": reference_coefficient,
        "target_trainable_parameters": target_budget,
        "actual_trainable_parameters": actual_parameters,
        "relative_budget_error": relative_error,
    }


def _linear_module_summary(named_modules: Iterable[tuple[str, Any]]) -> str:
    """Compact module-name evidence for actionable placement failures."""
    import torch

    names = [name for name, mod in named_modules if name and isinstance(mod, torch.nn.Linear)]
    leaves = sorted({name.rsplit(".", 1)[-1] for name in names})
    return f"{len(names)} linear modules; leaves={leaves[:32]}"


# ----------------------------------------------------------------------------- training (GPU)
@dataclass
class LoraVLMConfig:
    model_id: str
    train_jsonl: str                       # realistic_cases benchmark jsonl (image+question+answer)
    output_dir: str = "outputs/lora"
    placement: str = "all"                 # A5 group -> resolve_lora_targets
    lora_r: int = 16
    lora_alpha: int = 32
    lora_budget_reference_placement: str | None = None
    lora_budget_max_relative_error: float = 0.05
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 2                    # micro-batch (images/forward); effective batch = bs * grad_accum
    grad_accum: int = 8
    dtype: str = "bfloat16"
    seed: int = 7
    use_rationale: bool = True             # train target = rationale + answer (A2) when present
    grounding_repeat: int = 1              # A1 curriculum: repeat grounding rows so boxes are not diluted
    grounding_target: str = "pixel"        # "pixel" (legacy) or "norm" (normalized 0-1 boxes)
    # --- OOM / throughput controls (a single full-page doc -> ~1000+ vision tokens dominates memory) ---
    grad_checkpointing: bool = True        # trade compute for ~big activation-memory savings
    max_image_long_side: int | None = 768  # downscale the image's long side before the processor
                                           # (caps vision-token count; None = native resolution)
    # --- per-epoch logging / Weights & Biases (optional; no-op if unset or wandb missing) ---
    wandb_project: str | None = None       # set -> log loss + per-epoch eval metrics to W&B
    wandb_run: str | None = None           # W&B run name (e.g. "A0-qwen3_5-0.8b-n200")
    eval_max_new_tokens: int = 64          # generation length for the per-epoch eval
    eval_max_samples: int = 64             # cap eval samples/probe (arm regen makes realistic huge)
    log_every: int = 10                    # stdout + W&B train-loss cadence (micro-steps)


def _device_dtype(prefer_dtype: str):
    """Pick a device + a safe dtype: GPU keeps the requested dtype; CPU forces float32
    (bf16/fp16 matmul is slow/unsupported for some ops on CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda", getattr(torch, prefer_dtype)
    return "cpu", torch.float32


def _norm_box_answer(answer: str) -> str | None:
    """Convert the gold 'x1,y1,x2,y2;W,H' box into a stable normalized target string."""
    try:
        box_s, size_s = answer.split(";")
        x1, y1, x2, y2 = [float(x) for x in box_s.split(",")]
        w, h = [float(x) for x in size_s.split(",")]
        if not w or not h:
            return None
        return f"[{x1 / w:.4f}, {y1 / h:.4f}, {x2 / w:.4f}, {y2 / h:.4f}]"
    except Exception:
        return None


def _target_text(sample: dict, cfg: LoraVLMConfig | None = None) -> str:
    """Supervision target: rationale (if present, A2) then the gold answer.

    For A1 grounding, normalized boxes are easier for LFM to learn than document-pixel coordinates
    whose scale changes with every render. The existing grounding metric already accepts normalized
    predictions and rescales them to the original-pixel gold frame.
    """
    ans = sample["answers"][0]
    if cfg and cfg.grounding_target == "norm" and sample.get("metric") == "grounding":
        ans = _norm_box_answer(ans) or ans.split(";", 1)[0]
    rat = sample.get("rationale") or sample.get("meta", {}).get("rationale")
    return f"{rat}\nAnswer: {ans}" if rat else ans


def _training_question(sample: dict, cfg: LoraVLMConfig) -> str:
    """Use a matching prompt when the A1 curriculum trains normalized coordinates."""
    if cfg.grounding_target == "norm" and sample.get("metric") == "grounding":
        target = sample["question"].split(" as [x1", 1)[0]
        return (f"{target} as [x1, y1, x2, y2] normalized to 0-1 image coordinates. "
                "Answer with only the four numbers.")
    return sample["question"]


def _cap_image(img, long_side: int | None):
    """Downscale a PIL image so its longest side <= long_side (keeps aspect). The #vision tokens — and
    thus prefill/activation memory — scales with image area, so this is the main OOM lever for VLMs."""
    if not long_side:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= long_side:
        return img
    from PIL import Image as _Image
    s = long_side / m
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), _Image.LANCZOS)


def _to_model(inputs, device, dt):
    """Move processor outputs to the model's device, casting FLOATING tensors (pixel_values) to the
    model dtype. Without this, float32 pixel_values hit a bf16 vision tower -> 'expected BFloat16 but
    found Float'. int tensors (input_ids/attention_mask/image_grid_thw) keep their dtype."""
    import torch
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=dt) if torch.is_floating_point(v) else v.to(device)
        else:
            out[k] = v
    return out


def _auto_vlm():
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM
    return _AutoVLM


def _score(model, proc, device, jsonl: str, max_new_tokens: int = 64,
           max_image_long_side: int | None = 768, tag: str = "",
           max_samples: int | None = 64, save_preds: str | None = None) -> dict:
    """Generate on (a fixed subsample of) ``jsonl`` with the (already-loaded) model and return the
    project's aggregate summary. ``max_samples`` caps the eval set — the arm path regenerates
    realistic_cases at --count (thousands of samples), so without a cap the suite eval would score the
    whole TRAINING set (slow + memorization-tainted). None/0 = score all."""
    import sys
    import time
    from pathlib import Path

    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from docvlm_eval.benchmarks import load_jsonl
    from docvlm_eval.metrics import aggregate
    from docvlm_eval.schema import Prediction

    was_training = model.training
    model.eval()
    dt = next(model.parameters()).dtype                        # cast pixel_values to the model dtype
    samples = load_jsonl(jsonl)
    if max_samples and len(samples) > max_samples:             # fixed-seed subsample -> fast + comparable
        import random as _r
        samples = _r.Random(0).sample(samples, max_samples)
    every = max(1, len(samples) // 5)
    t0 = time.time()
    preds: dict[str, Prediction] = {}
    for i, s in enumerate(samples, 1):
        img = _cap_image(Image.open(s.image_path).convert("RGB"), max_image_long_side)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": s.question}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt")
        inputs = _to_model(inputs, device, dt)
        n = inputs["input_ids"].shape[1]
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()
        except torch.cuda.OutOfMemoryError:        # skip the offending sample, keep scoring the rest
            torch.cuda.empty_cache()
            text = ""
        preds[s.sample_id] = Prediction(sample_id=s.sample_id, prediction=text, raw=text)
        if i == 1 or i % every == 0 or i == len(samples):
            print(f"    [eval {tag}] {i}/{len(samples)} ({(time.time()-t0):.0f}s)", flush=True)
    if was_training:
        model.train()
    agg = aggregate(samples, preds)
    if save_preds:
        # per-sample predictions + scores -> jsonl, so before/after runs can be diffed at the
        # EXAMPLE level (which samples flipped) — the fixed-seed subsample keeps the sets aligned
        import json as _json
        by_id = {s.sample_id: s for s in samples}
        out_p = Path(save_preds)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            for row in agg["per_sample"]:
                s = by_id.get(row.get("sample_id"))
                fh.write(_json.dumps({**row,
                                      "question": s.question if s else "",
                                      "answers": s.answers if s else [],
                                      "image_path": s.image_path if s else ""},
                                     ensure_ascii=False) + "\n")
    return agg["summary"]


def _wandb_init(cfg: "LoraVLMConfig"):
    """Start a W&B run if cfg.wandb_project is set and wandb is importable+logged-in; else None."""
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed (pip install wandb) -> training without logging")
        return None
    try:
        if wandb.run is not None:           # close any run left open by a previous size, THEN init a
            wandb.finish()                  # fresh one — this replaces the now-deprecated reinit=True
    except Exception:
        pass
    try:
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run,
                         config={"model": cfg.model_id, "placement": cfg.placement,
                                 "epochs": cfg.epochs, "lr": cfg.learning_rate, "r": cfg.lora_r,
                                 "train_jsonl": cfg.train_jsonl})
        print(f"[wandb] logging to project={cfg.wandb_project} run={cfg.wandb_run}")
        return run
    except Exception as e:                  # not logged in / offline / no API key -> continue, no crash
        print(f"[wandb] init failed ({type(e).__name__}: {e}); set WANDB_API_KEY or run "
              f"`wandb login` -> training WITHOUT logging")
        return None


def train_lora_vlm(cfg: LoraVLMConfig,
                   eval_specs: list[tuple[str, str]] | None = None) -> tuple[str, dict]:
    """LoRA-fine-tune one VLM on the synthetic data. GPU-only. Returns (adapter_dir, last_eval).

    Builds (image, question) -> target with prompt-masked labels via the processor's chat template,
    applies LoRA on the A5-resolved modules, then runs an EPOCH loop. Each epoch logs the mean train
    loss and — for every (name, jsonl) in ``eval_specs`` (e.g. train + held-out) — an in-process eval
    summary, to stdout and to Weights & Biases (when cfg.wandb_project is set). ``last_eval`` maps
    each spec name to its final-epoch summary so callers can record it without reloading the model."""
    import json
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoProcessor
    from PIL import Image

    torch.manual_seed(cfg.seed)
    device, dt = _device_dtype(cfg.dtype)
    print(f"[lora_vlm.train] device={device} dtype={dt}"
          + ("" if device == "cuda" else "  (CPU — training will be slow; expected a GPU?)"))
    proc = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = _auto_vlm().from_pretrained(cfg.model_id, torch_dtype=dt,
                                        trust_remote_code=True).to(device)

    named_modules = list(model.named_modules())
    targets, budget_report = resolve_lora_budget(
        named_modules,
        cfg.placement,
        requested_rank=cfg.lora_r,
        requested_alpha=cfg.lora_alpha,
        reference_placement=cfg.lora_budget_reference_placement,
    )
    if not targets:
        summary = _linear_module_summary(named_modules)
        raise RuntimeError(
            f"no LoRA targets for placement={cfg.placement!r} on {cfg.model_id}; {summary}. "
            "Confirm that the Colab checkout is current and inspect the model's module names."
        )
    cfg = replace(
        cfg,
        lora_r=budget_report["effective_rank"],
        lora_alpha=budget_report["effective_alpha"],
    )
    if (
        budget_report["relative_budget_error"]
        > cfg.lora_budget_max_relative_error
    ):
        raise RuntimeError(
            "integer LoRA rank cannot satisfy the requested adapter budget: "
            f"relative error {budget_report['relative_budget_error']:.2%} > "
            f"{cfg.lora_budget_max_relative_error:.2%}; increase the reference "
            "rank or relax lora_budget_max_relative_error"
        )
    print(
        f"[lora_vlm.train] placement={cfg.placement} resolved {len(targets)} targets; "
        f"rank={cfg.lora_r} alpha={cfg.lora_alpha} "
        f"adapter_params={budget_report['actual_trainable_parameters']:,} "
        f"budget_error={budget_report['relative_budget_error']:.2%}; "
        f"sample={targets[:8]}",
        flush=True,
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", target_modules=targets, task_type="CAUSAL_LM"))
    budget_report["realized_trainable_parameters"] = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    budget_report["realized_relative_budget_error"] = (
        abs(
            budget_report["realized_trainable_parameters"]
            - budget_report["target_trainable_parameters"]
        )
        / budget_report["target_trainable_parameters"]
    )
    if (
        cfg.lora_budget_reference_placement is not None
        and budget_report["realized_relative_budget_error"]
        > cfg.lora_budget_max_relative_error
    ):
        raise RuntimeError(
            "realized PEFT trainable parameters violate the adapter budget: "
            f"relative error "
            f"{budget_report['realized_relative_budget_error']:.2%} > "
            f"{cfg.lora_budget_max_relative_error:.2%}"
        )
    model.print_trainable_parameters()
    if cfg.grad_checkpointing:                 # big activation-memory saver (the main OOM lever)
        if hasattr(model, "config"):
            model.config.use_cache = False     # incompatible with checkpointing
        model.enable_input_require_grads()     # needed so grads flow with a frozen embedding + PEFT
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = [
        json.loads(line)
        for line in Path(cfg.train_jsonl).read_text().splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("answers")]
    if not rows:
        raise RuntimeError(f"no trainable rows (with answers) in {cfg.train_jsonl}")
    if cfg.grounding_repeat > 1:
        grounding = [r for r in rows if r.get("metric") == "grounding"]
        if grounding:
            rows = rows + grounding * (cfg.grounding_repeat - 1)
            print(f"[lora_vlm.train] A1 grounding curriculum: repeated {len(grounding)} grounding "
                  f"rows x{cfg.grounding_repeat} -> {len(rows)} train rows; "
                  f"target={cfg.grounding_target}", flush=True)

    class _DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]

    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "right"   # so each sample's prompt span is at [0:prompt_len]

    def collate(batch):
        # batched VLM collate: one processor call over all conversations (pads input_ids/attention_mask
        # and stacks pixel_values), then mask each sample's PROMPT span and the padding in the labels.
        convs_full, prompt_lens = [], []
        for b in batch:
            img = _cap_image(Image.open(b["image_path"]).convert("RGB"), cfg.max_image_long_side)
            tgt = _target_text(b, cfg) if cfg.use_rationale else b["answers"][0]
            user = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": _training_question(b, cfg)}]}]
            convs_full.append(user + [{"role": "assistant", "content": [{"type": "text", "text": tgt}]}])
            p = proc.apply_chat_template(user, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt")
            prompt_lens.append(int(p["input_ids"].shape[1]))
        full = proc.apply_chat_template(convs_full, tokenize=True, return_dict=True,
                                        return_tensors="pt", padding=True)
        labels = full["input_ids"].clone()
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100                              # supervise only the answer span
        am = full.get("attention_mask")
        if am is not None:
            labels[am == 0] = -100                             # ignore right-padding
        full["labels"] = labels
        return _to_model(full, device, dt)                    # cast pixel_values -> model dtype

    dl = DataLoader(_DS(), batch_size=max(1, cfg.batch_size), shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cap = cfg.max_steps or float("inf")               # optional hard step cap (control factor)
    run = _wandb_init(cfg)
    if run:        # distinct namespaces so define_metric globs don't collide: loss vs step, eval vs epoch
        import wandb
        run.config.update({"lora_budget": budget_report}, allow_val_change=True)
        wandb.define_metric("train/global_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="train/global_step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
        wandb.define_metric("eval_by_axis/*", step_metric="epoch")
    import time
    steps_per_epoch = len(dl)
    total = int(min(cap, cfg.epochs * steps_per_epoch))
    print(f"[lora_vlm.train] {len(rows)} samples, bs={cfg.batch_size} x {cfg.epochs} epochs = {total} steps "
          f"(grad_accum={cfg.grad_accum} -> ~{max(total // cfg.grad_accum, 1)} optimizer steps) | "
          f"img_cap={cfg.max_image_long_side} grad_ckpt={cfg.grad_checkpointing}", flush=True)
    t0 = time.time()
    gstep = 0
    last_eval: dict = {}
    try:
        for epoch in range(cfg.epochs):
            model.train()
            losses = []
            for batch in dl:
                out = model(**batch)
                if out.loss is None:
                    raise RuntimeError("model returned loss=None — labels/inputs not accepted by this "
                                       "model's forward; check the processor chat-template / image keys")
                (out.loss / cfg.grad_accum).backward()
                if (gstep + 1) % cfg.grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                losses.append(float(out.loss.detach()))
                gstep += 1
                if gstep == 1 or gstep % cfg.log_every == 0 or gstep >= total:
                    el = time.time() - t0
                    rate = gstep / max(el, 1e-9)
                    eta = (total - gstep) / max(rate, 1e-9)
                    print(f"    step {gstep}/{total} (ep {epoch+1}/{cfg.epochs}) loss={losses[-1]:.3f} "
                          f"| {rate:.2f} it/s | elapsed {el/60:.1f}m | eta {eta/60:.1f}m", flush=True)
                    if run:
                        run.log({"train/loss": losses[-1], "train/global_step": gstep, "epoch": epoch})
                if gstep >= cap:
                    break
            log = {"epoch": epoch, "epoch/loss": sum(losses) / max(len(losses), 1)}
            for name, path in (eval_specs or []):          # per-epoch eval (loss AND metrics)
                summ = _score(model, proc, device, path, cfg.eval_max_new_tokens,
                              cfg.max_image_long_side, tag=name, max_samples=cfg.eval_max_samples)
                last_eval[name] = summ
                if summ.get("score") is not None:
                    log[f"eval/{name}_score"] = summ["score"]
                    log[f"eval_by_axis/score/{name}"] = summ["score"]
                for ax, info in (summ.get("by_answer_type") or {}).items():
                    if isinstance(info, dict) and info.get("score") is not None:
                        log[f"eval/{name}_{ax}"] = info["score"]
                        log[f"eval_by_axis/{ax}/{name}"] = info["score"]
            if run:
                run.log(log)
            print("  [epoch %d/%d done] " % (epoch + 1, cfg.epochs)
                  + " ".join(f"{k}={v:.3f}" for k, v in log.items() if isinstance(v, float)), flush=True)
            if gstep >= cap:
                break

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        proc.save_pretrained(out_dir)
        (out_dir / "lora_budget.json").write_text(
            json.dumps(budget_report, indent=2),
            encoding="utf-8",
        )
    finally:
        if run:                                            # always close, even on OOM/crash, so the
            run.finish()                                   # next sweep size's wandb.init() is clean
    return str(out_dir), last_eval


def _load_for_eval(model_id: str, adapter_path: str | None, dtype: str = "bfloat16"):
    """Load the (optionally LoRA-adapted) model + processor ONCE -> (model, proc, device)."""
    device, dt = _device_dtype(dtype)
    print(f"[lora_vlm.eval] loading {model_id}"
          + (" + adapter" if adapter_path else "") + f" on {device}/{dt}"
          + ("" if device == "cuda" else "  (CPU — eval will be slow)"), flush=True)
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(adapter_path or model_id, trust_remote_code=True)
    model = _auto_vlm().from_pretrained(model_id, torch_dtype=dt,
                                        trust_remote_code=True).to(device).eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to(device).eval()
    return model, proc, device


def score_suite(model_id: str, jsonls: dict, *, adapter_path: str | None = None,
                dtype: str = "bfloat16", max_new_tokens: int = 64,
                max_image_long_side: int | None = 768, max_samples: int | None = 64,
                save_preds_dir: str | None = None) -> dict:
    """Load the (optionally adapted) model ONCE and score several probe jsonls -> {name: summary}.
    Avoids reloading the full model per probe (the repeated 'Loading weights 589/589' stall after
    training that looked like a hang). ``max_samples`` caps each probe. ``save_preds_dir`` writes
    per-sample predictions+scores to ``<dir>/<probe>.jsonl`` (the before/after example-level diff
    needs them). Frees the model at the end."""
    import torch
    model, proc, device = _load_for_eval(model_id, adapter_path, dtype)
    try:
        return {name: _score(model, proc, device, jl, max_new_tokens, max_image_long_side,
                             tag=name, max_samples=max_samples,
                             save_preds=(f"{save_preds_dir}/{name}.jsonl"
                                         if save_preds_dir else None))
                for name, jl in jsonls.items()}
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def eval_vlm(model_id: str, jsonl: str, *, adapter_path: str | None = None,
             dtype: str = "bfloat16", max_new_tokens: int = 64,
             max_image_long_side: int | None = 768, max_samples: int | None = 64) -> dict:
    """Score a base (optionally LoRA-adapted) VLM on ONE probe jsonl -> aggregate summary + per-axis
    breakdown. GPU-only. (For several probes use score_suite, which loads the model only once.)"""
    model, proc, device = _load_for_eval(model_id, adapter_path, dtype)
    return _score(model, proc, device, jsonl, max_new_tokens, max_image_long_side,
                  tag="probe", max_samples=max_samples)

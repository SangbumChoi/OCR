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
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# capability -> module bucket (mirrors the hypotheses in research_novelty.md / ablation_plan.md)
PLACEMENT_GROUPS = ("vision", "connector", "llm_attn", "llm_mlp", "all")

_VISION_PAT = re.compile(r"vis(ual|ion)|patch|siglip|navit|aimv2", re.I)
_CONNECTOR_PAT = re.compile(r"merger|projector|connector|mlp1|multi_modal|abstractor|resampler", re.I)
_ATTN_LEAF = re.compile(r"(^|\.)(q|k|v|o|qkv|out|wq|wk|wv|wo)_?proj$", re.I)
_MLP_LEAF = re.compile(r"(^|\.)(gate|up|down|fc1|fc2|w1|w2|w3)_?proj$|(^|\.)(fc1|fc2)$", re.I)


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
        elif group == "llm_attn" and bkt == "llm" and _ATTN_LEAF.search(leaf):
            targets.append(name)
        elif group == "llm_mlp" and bkt == "llm" and _MLP_LEAF.search(leaf):
            targets.append(name)
    return sorted(set(targets))


# ----------------------------------------------------------------------------- training (GPU)
@dataclass
class LoraVLMConfig:
    model_id: str
    train_jsonl: str                       # realistic_cases benchmark jsonl (image+question+answer)
    output_dir: str = "outputs/lora"
    placement: str = "all"                 # A5 group -> resolve_lora_targets
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-4
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 1                    # micro-batch (one image/forward); effective batch = grad_accum
    grad_accum: int = 8
    dtype: str = "bfloat16"
    seed: int = 7
    use_rationale: bool = True             # train target = rationale + answer (A2) when present
    # --- OOM controls (a single full-page doc -> ~1000+ vision tokens dominates memory, not batch) ---
    grad_checkpointing: bool = True        # trade compute for ~big activation-memory savings
    max_image_long_side: int | None = 1024 # downscale the image's long side before the processor
                                           # (caps vision-token count; None = native resolution)
    # --- per-epoch logging / Weights & Biases (optional; no-op if unset or wandb missing) ---
    wandb_project: str | None = None       # set -> log loss + per-epoch eval metrics to W&B
    wandb_run: str | None = None           # W&B run name (e.g. "A0-qwen3_5-0.8b-n200")
    eval_max_new_tokens: int = 64          # generation length for the per-epoch eval
    log_every: int = 10                    # stdout + W&B train-loss cadence (micro-steps)


def _device_dtype(prefer_dtype: str):
    """Pick a device + a safe dtype: GPU keeps the requested dtype; CPU forces float32
    (bf16/fp16 matmul is slow/unsupported for some ops on CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda", getattr(torch, prefer_dtype)
    return "cpu", torch.float32


def _target_text(sample: dict) -> str:
    """Supervision target: rationale (if present, A2) then the gold answer."""
    ans = sample["answers"][0]
    rat = sample.get("rationale")
    return f"{rat}\nAnswer: {ans}" if rat else ans


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
           max_image_long_side: int | None = 1024) -> dict:
    """Generate on every sample of ``jsonl`` with the (already-loaded) model and return the project's
    aggregate summary. Shared by ``eval_vlm`` and the per-epoch eval inside ``train_lora_vlm`` so we
    never reload the model just to score it (the reload was a needless OOM risk)."""
    import sys
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
    preds: dict[str, Prediction] = {}
    for s in samples:
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
            torch.cuda.empty_cache(); text = ""
        preds[s.sample_id] = Prediction(sample_id=s.sample_id, prediction=text, raw=text)
    if was_training:
        model.train()
    return aggregate(samples, preds)["summary"]


def _wandb_init(cfg: "LoraVLMConfig"):
    """Start a W&B run if cfg.wandb_project is set and wandb is importable+logged-in; else None."""
    if not cfg.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] not installed (pip install wandb) -> training without logging"); return None
    try:
        if wandb.run is not None:           # a previous size that crashed left a run open -> close it
            wandb.finish()
    except Exception:
        pass
    try:
        run = wandb.init(project=cfg.wandb_project, name=cfg.wandb_run, reinit=True,
                         config={"model": cfg.model_id, "placement": cfg.placement,
                                 "epochs": cfg.epochs, "lr": cfg.learning_rate, "r": cfg.lora_r,
                                 "train_jsonl": cfg.train_jsonl})
        print(f"[wandb] logging to project={cfg.wandb_project} run={cfg.wandb_run}")
        return run
    except Exception as e:                  # not logged in / offline / no API key -> continue, no crash
        print(f"[wandb] init failed ({type(e).__name__}: {e}); set WANDB_API_KEY or run "
              f"`wandb login` -> training WITHOUT logging"); return None


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

    targets = resolve_lora_targets(model.named_modules(), cfg.placement)
    if not targets:
        raise RuntimeError(f"no LoRA targets for placement={cfg.placement!r} on {cfg.model_id}")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if cfg.grad_checkpointing:                 # big activation-memory saver (the main OOM lever)
        if hasattr(model, "config"):
            model.config.use_cache = False     # incompatible with checkpointing
        model.enable_input_require_grads()     # needed so grads flow with a frozen embedding + PEFT
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = [json.loads(l) for l in Path(cfg.train_jsonl).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("answers")]
    if not rows:
        raise RuntimeError(f"no trainable rows (with answers) in {cfg.train_jsonl}")

    class _DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]

    def collate(batch):
        b = batch[0]   # batch_size=1 keeps the variable image sizes simple
        img = _cap_image(Image.open(b["image_path"]).convert("RGB"), cfg.max_image_long_side)
        tgt = _target_text(b) if cfg.use_rationale else b["answers"][0]
        user = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": b["question"]}]}]
        full = proc.apply_chat_template(
            user + [{"role": "assistant", "content": [{"type": "text", "text": tgt}]}],
            tokenize=True, return_dict=True, return_tensors="pt")
        prompt = proc.apply_chat_template(user, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100      # supervise only the answer span
        full["labels"] = labels
        return _to_model(full, device, dt)                    # cast pixel_values -> model dtype

    dl = DataLoader(_DS(), batch_size=1, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    cap = cfg.max_steps or float("inf")               # optional hard step cap (control factor)
    run = _wandb_init(cfg)
    if run:        # distinct namespaces so define_metric globs don't collide: loss vs step, eval vs epoch
        import wandb
        wandb.define_metric("train/global_step"); wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="train/global_step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
    import time
    steps_per_epoch = len(dl)
    total = int(min(cap, cfg.epochs * steps_per_epoch))
    print(f"[lora_vlm.train] {len(rows)} samples x {cfg.epochs} epochs = {total} micro-steps "
          f"(grad_accum={cfg.grad_accum} -> ~{max(total // cfg.grad_accum, 1)} optimizer steps) | "
          f"img_cap={cfg.max_image_long_side} grad_ckpt={cfg.grad_checkpointing}", flush=True)
    t0 = time.time()
    gstep = 0; last_eval: dict = {}
    try:
        for epoch in range(cfg.epochs):
            model.train(); losses = []
            for batch in dl:
                out = model(**batch)
                if out.loss is None:
                    raise RuntimeError("model returned loss=None — labels/inputs not accepted by this "
                                       "model's forward; check the processor chat-template / image keys")
                (out.loss / cfg.grad_accum).backward()
                if (gstep + 1) % cfg.grad_accum == 0:
                    opt.step(); opt.zero_grad()
                losses.append(float(out.loss.detach()))
                gstep += 1
                if gstep == 1 or gstep % cfg.log_every == 0 or gstep >= total:
                    el = time.time() - t0; rate = gstep / max(el, 1e-9); eta = (total - gstep) / max(rate, 1e-9)
                    print(f"    step {gstep}/{total} (ep {epoch+1}/{cfg.epochs}) loss={losses[-1]:.3f} "
                          f"| {rate:.2f} it/s | elapsed {el/60:.1f}m | eta {eta/60:.1f}m", flush=True)
                    if run:
                        run.log({"train/loss": losses[-1], "train/global_step": gstep, "epoch": epoch})
                if gstep >= cap:
                    break
            log = {"epoch": epoch, "epoch/loss": sum(losses) / max(len(losses), 1)}
            for name, path in (eval_specs or []):          # per-epoch eval (loss AND metrics)
                summ = _score(model, proc, device, path, cfg.eval_max_new_tokens, cfg.max_image_long_side)
                last_eval[name] = summ
                if summ.get("score") is not None:
                    log[f"eval/{name}_score"] = summ["score"]
                for ax, info in (summ.get("by_answer_type") or {}).items():
                    if isinstance(info, dict) and info.get("score") is not None:
                        log[f"eval/{name}_{ax}"] = info["score"]
            if run:
                run.log(log)
            print("  [epoch %d/%d done] " % (epoch + 1, cfg.epochs)
                  + " ".join(f"{k}={v:.3f}" for k, v in log.items() if isinstance(v, float)), flush=True)
            if gstep >= cap:
                break

        out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir); proc.save_pretrained(out_dir)
    finally:
        if run:                                            # always close, even on OOM/crash, so the
            run.finish()                                   # next sweep size's wandb.init() is clean
    return str(out_dir), last_eval


def eval_vlm(model_id: str, jsonl: str, *, adapter_path: str | None = None,
             dtype: str = "bfloat16", max_new_tokens: int = 64,
             max_image_long_side: int | None = 1024) -> dict:
    """Score a base (optionally LoRA-adapted) VLM on a probe jsonl and return the aggregate summary
    + per-axis (by answer_type) breakdown — the before/after measurement for an ablation arm.

    GPU-only. Reuses the project's metrics so scores are comparable to the eval pipeline."""
    import torch  # noqa: F401  (ensures a clear error if torch is missing)

    device, dt = _device_dtype(dtype)
    print(f"[lora_vlm.eval] device={device} dtype={dt}"
          + ("" if device == "cuda" else "  (CPU — eval will be slow; expected a GPU?)"))
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(adapter_path or model_id, trust_remote_code=True)
    model = _auto_vlm().from_pretrained(model_id, torch_dtype=dt,
                                        trust_remote_code=True).to(device).eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).to(device).eval()
    return _score(model, proc, device, jsonl, max_new_tokens, max_image_long_side)

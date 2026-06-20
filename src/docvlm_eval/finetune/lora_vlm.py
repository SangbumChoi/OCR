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
    batch_size: int = 1
    grad_accum: int = 8
    dtype: str = "bfloat16"
    seed: int = 7
    use_rationale: bool = True             # train target = rationale + answer (A2) when present


def _target_text(sample: dict) -> str:
    """Supervision target: rationale (if present, A2) then the gold answer."""
    ans = sample["answers"][0]
    rat = sample.get("rationale")
    return f"{rat}\nAnswer: {ans}" if rat else ans


def train_lora_vlm(cfg: LoraVLMConfig) -> str:
    """LoRA-fine-tune one VLM on the synthetic data. GPU-only; returns the adapter dir.

    Kept deliberately small: builds (image, question) -> target with prompt-masked labels via the
    processor's chat template, applies LoRA on the A5-resolved modules, runs a short loop, saves
    the adapter. Heavy imports are local so importing this module stays cheap."""
    import json
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM
    from PIL import Image

    torch.manual_seed(cfg.seed)
    proc = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = _AutoVLM.from_pretrained(cfg.model_id, torch_dtype=getattr(torch, cfg.dtype),
                                     trust_remote_code=True).cuda()

    targets = resolve_lora_targets(model.named_modules(), cfg.placement)
    if not targets:
        raise RuntimeError(f"no LoRA targets for placement={cfg.placement!r} on {cfg.model_id}")
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in Path(cfg.train_jsonl).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("answers")]

    class _DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i): return rows[i]

    def collate(batch):
        b = batch[0]   # batch_size=1 keeps the variable image sizes simple
        img = Image.open(b["image_path"]).convert("RGB")
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
        return {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in full.items()}

    dl = DataLoader(_DS(), batch_size=1, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train()
    step, total = 0, cfg.max_steps or (cfg.epochs * len(dl))
    while step < total:
        for batch in dl:
            out = model(**batch)
            (out.loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                opt.step(); opt.zero_grad()
            step += 1
            if step >= total:
                break
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir); proc.save_pretrained(out_dir)
    return str(out_dir)


def eval_vlm(model_id: str, jsonl: str, *, adapter_path: str | None = None,
             dtype: str = "bfloat16", max_new_tokens: int = 64) -> dict:
    """Score a base (optionally LoRA-adapted) VLM on a probe jsonl and return the aggregate summary
    + per-axis (by answer_type) breakdown — the before/after measurement for an ablation arm.

    GPU-only. Reuses the project's metrics so scores are comparable to the eval pipeline."""
    import sys
    from pathlib import Path

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoVLM

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from docvlm_eval.benchmarks import load_jsonl
    from docvlm_eval.metrics import aggregate
    from docvlm_eval.schema import Prediction

    proc = AutoProcessor.from_pretrained(adapter_path or model_id, trust_remote_code=True)
    model = _AutoVLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype),
                                     trust_remote_code=True).cuda().eval()
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path).cuda().eval()

    samples = load_jsonl(jsonl)
    preds: dict[str, Prediction] = {}
    for s in samples:
        img = Image.open(s.image_path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": s.question}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to("cuda")
        n = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()
        preds[s.sample_id] = Prediction(sample_id=s.sample_id, prediction=text, raw=text)
    return aggregate(samples, preds)["summary"]

"""A native vision-resampler-decoder model for the approximately 800M student."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import ConnectorConfig, LanguageConfig, StudentConfig, VisionConfig
from .losses import box_iou_loss, decode_normalized_box


@dataclass
class _PackedAttentionPlan:
    requested_backend: str
    q_cu_seqlens: torch.Tensor
    kv_cu_seqlens: torch.Tensor
    use_flex: bool = False
    block_mask: Any = None


_COMPILED_FLEX_ATTENTION: Any = None
_FLEX_DISABLED_DEVICES: set[str] = set()
_SDPA_SUPPORTS_GQA = "enable_gqa" in (
    F.scaled_dot_product_attention.__doc__ or ""
)


def _scaled_dot_product_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_bias: torch.Tensor | None,
    dropout: float,
    is_causal: bool,
) -> torch.Tensor:
    if query.shape[1] == key.shape[1]:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=dropout,
            is_causal=is_causal,
        )
    if _SDPA_SUPPORTS_GQA and query.is_cuda:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=dropout,
            is_causal=is_causal,
            enable_gqa=True,
        )
    repeats = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(repeats, dim=1),
        value.repeat_interleave(repeats, dim=1),
        attn_mask=attention_bias,
        dropout_p=dropout,
        is_causal=is_causal,
    )


def _checkpointed(
    function,
    *args: torch.Tensor,
    enabled: bool,
    use_reentrant: bool,
) -> torch.Tensor:
    if enabled and torch.is_grad_enabled():
        return torch_checkpoint(
            function,
            *args,
            use_reentrant=use_reentrant,
        )
    return function(*args)


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION = torch.compile(
            flex_attention,
            dynamic=True,
        )
    return _COMPILED_FLEX_ATTENTION


def _create_document_block_mask(
    q_cu_seqlens: torch.Tensor,
    kv_cu_seqlens: torch.Tensor,
    *,
    q_length: int,
    kv_length: int,
    device: torch.device,
    compile_mask: bool,
):
    from torch.nn.attention.flex_attention import create_block_mask

    q_lengths = q_cu_seqlens[1:] - q_cu_seqlens[:-1]
    kv_lengths = kv_cu_seqlens[1:] - kv_cu_seqlens[:-1]
    q_documents = torch.repeat_interleave(
        torch.arange(q_lengths.numel(), device=device),
        q_lengths,
    )
    kv_documents = torch.repeat_interleave(
        torch.arange(kv_lengths.numel(), device=device),
        kv_lengths,
    )

    def same_document_mask(batch, head, q_index, kv_index):
        del batch, head
        return q_documents[q_index] == kv_documents[kv_index]

    return create_block_mask(
        same_document_mask,
        B=None,
        H=None,
        Q_LEN=q_length,
        KV_LEN=kv_length,
        device=device,
        BLOCK_SIZE=128,
        _compile=compile_mask,
    )


def _prepare_packed_attention(
    backend: str,
    q_cu_seqlens: torch.Tensor,
    kv_cu_seqlens: torch.Tensor,
    *,
    q_length: int,
    kv_length: int,
    device: torch.device,
    dropout: float = 0.0,
) -> _PackedAttentionPlan:
    plan = _PackedAttentionPlan(
        requested_backend=backend,
        q_cu_seqlens=q_cu_seqlens,
        kv_cu_seqlens=kv_cu_seqlens,
    )
    if backend not in {"auto", "flex", "loop"}:
        raise ValueError("packed attention backend must be auto, flex, or loop")
    if backend == "loop":
        return plan
    unsupported = (
        device.type != "cuda"
        or dropout != 0.0
        or str(device) in _FLEX_DISABLED_DEVICES
    )
    if unsupported:
        if backend == "flex":
            raise RuntimeError(
                "packed flex attention requires CUDA, zero vision dropout, "
                "and a working torch.compile FlexAttention backend"
            )
        return plan
    try:
        plan.block_mask = _create_document_block_mask(
            q_cu_seqlens,
            kv_cu_seqlens,
            q_length=q_length,
            kv_length=kv_length,
            device=device,
            compile_mask=True,
        )
        plan.use_flex = True
    except Exception as error:
        if backend == "flex":
            raise RuntimeError("failed to prepare packed FlexAttention") from error
        _FLEX_DISABLED_DEVICES.add(str(device))
    return plan


def _loop_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: _PackedAttentionPlan,
    *,
    dropout_p: float,
) -> torch.Tensor:
    q_boundaries = [int(value) for value in plan.q_cu_seqlens.tolist()]
    kv_boundaries = [int(value) for value in plan.kv_cu_seqlens.tolist()]
    if len(q_boundaries) != len(kv_boundaries):
        raise ValueError("packed query and source batches must have equal size")
    return torch.cat(
        [
            F.scaled_dot_product_attention(
                query[:, :, q_start:q_end],
                key[:, :, kv_start:kv_end],
                value[:, :, kv_start:kv_end],
                dropout_p=dropout_p,
            )
            for (q_start, q_end), (kv_start, kv_end) in zip(
                zip(q_boundaries, q_boundaries[1:]),
                zip(kv_boundaries, kv_boundaries[1:]),
            )
        ],
        dim=2,
    )


def _packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: _PackedAttentionPlan,
    *,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    if plan.use_flex:
        try:
            return _compiled_flex_attention()(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                block_mask=plan.block_mask,
            )
        except Exception as error:
            if plan.requested_backend == "flex":
                raise RuntimeError(
                    "compiled packed FlexAttention execution failed"
                ) from error
            _FLEX_DISABLED_DEVICES.add(str(query.device))
            plan.use_flex = False
    return _loop_packed_attention(
        query,
        key,
        value,
        plan,
        dropout_p=dropout_p,
    )


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(source_dtype)


class VisionAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        attention_bias = None
        if token_mask is not None:
            minimum = torch.finfo(q.dtype).min
            attention_bias = (
                1.0 - token_mask[:, None, None, :].to(q.dtype)
            ) * minimum
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, length, -1))

    def forward_packed(
        self,
        x: torch.Tensor,
        plan: _PackedAttentionPlan,
    ) -> torch.Tensor:
        length = x.shape[0]
        shape = (1, length, self.heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = _packed_attention(
            q,
            k,
            v,
            plan,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.o_proj(
            out.transpose(1, 2).contiguous().view(length, -1)
        )


class VisionMLP(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.dropout(F.gelu(self.fc1(x)), self.dropout, self.training))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attn = VisionAttention(config.width, config.attention_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = VisionMLP(
            config.width,
            int(config.width * config.mlp_ratio),
            config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_mask)
        return x + self.mlp(self.norm2(x))

    def forward_packed(
        self,
        x: torch.Tensor,
        plan: _PackedAttentionPlan,
    ) -> torch.Tensor:
        x = x + self.attn.forward_packed(self.norm1(x), plan)
        return x + self.mlp(self.norm2(x))


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(
            3,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(config.max_position_tokens, config.width)
        )
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.last_packed_attention_backend = "none"
        self.gradient_checkpointing = False
        self.checkpoint_use_reentrant = False

    def _position_ids(
        self,
        patch_height: int,
        patch_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        grid_side = int(self.position_embedding.shape[0] ** 0.5)
        if grid_side * grid_side != self.position_embedding.shape[0]:
            raise ValueError(
                "stable two-dimensional visual positions require a square "
                "max_position_tokens grid"
            )
        if patch_height > grid_side or patch_width > grid_side:
            raise ValueError(
                f"image patch grid {patch_height}x{patch_width} exceeds "
                f"the configured {grid_side}x{grid_side} position grid"
            )
        rows = torch.arange(patch_height, device=device)[:, None]
        columns = torch.arange(patch_width, device=device)[None, :]
        return (rows * grid_side + columns).flatten()

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
        capture_layers: set[int] | None = None,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]
    ):
        patch = self.config.patch_size
        height, width = pixel_values.shape[-2:]
        if pixel_mask is not None and pixel_mask.shape != (
            pixel_values.shape[0],
            height,
            width,
        ):
            raise ValueError(
                "pixel_mask must have shape [batch, image_height, image_width]"
            )
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))
            if pixel_mask is not None:
                pixel_mask = F.pad(pixel_mask, (0, pad_w, 0, pad_h))
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        if x.shape[1] > self.position_embedding.shape[0]:
            raise ValueError(
                f"image produced {x.shape[1]} visual tokens, above "
                f"max_position_tokens={self.position_embedding.shape[0]}"
            )
        if pixel_mask is None:
            patch_mask = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            patch_mask = (
                F.max_pool2d(
                    pixel_mask[:, None].to(dtype=torch.float32),
                    kernel_size=patch,
                    stride=patch,
                )
                .flatten(1)
                .bool()
            )
        patch_height = pixel_values.shape[-2] // patch
        patch_width = pixel_values.shape[-1] // patch
        position_ids = self._position_ids(
            patch_height,
            patch_width,
            x.device,
        )
        x = x + self.position_embedding[position_ids].unsqueeze(0).to(x.dtype)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = _checkpointed(
                block,
                x,
                patch_mask,
                enabled=self.gradient_checkpointing and self.training,
                use_reentrant=self.checkpoint_use_reentrant,
            )
            x = x * patch_mask.unsqueeze(-1)
            if capture_layers is not None and index in capture_layers:
                features[index] = x
        x = self.norm(x) * patch_mask.unsqueeze(-1)
        if capture_layers is not None and -1 in capture_layers:
            features[-1] = x
        if capture_layers is not None:
            missing = capture_layers - features.keys()
            if missing:
                raise ValueError(f"unknown vision feature layers: {sorted(missing)}")
            if not return_mask:
                raise ValueError("capturing vision layers requires return_mask=True")
            return x, patch_mask, features
        return (x, patch_mask) if return_mask else x

    def forward_packed(
        self,
        packed_pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        attention_backend: str = "auto",
        capture_layers: set[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Encode concatenated per-image patch sequences without batch padding."""

        patch = self.config.patch_size
        if packed_pixel_values.ndim != 4 or packed_pixel_values.shape[1:] != (
            3,
            patch,
            patch,
        ):
            raise ValueError(
                "packed_pixel_values must have shape [tokens, 3, patch, patch]"
            )
        token_count = int(packed_pixel_values.shape[0])
        if position_ids.shape != (token_count,):
            raise ValueError("packed_position_ids must have shape [tokens]")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("packed_cu_seqlens must have shape [batch + 1]")
        boundaries = [int(value) for value in cu_seqlens.tolist()]
        if (
            boundaries[0] != 0
            or boundaries[-1] != token_count
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(
                "packed_cu_seqlens must be strictly increasing from zero to tokens"
            )
        if token_count > 0:
            minimum = int(position_ids.min().item())
            maximum = int(position_ids.max().item())
            if minimum < 0 or maximum >= self.position_embedding.shape[0]:
                raise ValueError("packed_position_ids exceed the visual position grid")
        x = self.patch_embed(packed_pixel_values).flatten(1)
        x = x + self.position_embedding[position_ids].to(x.dtype)
        plan = _prepare_packed_attention(
            attention_backend,
            cu_seqlens,
            cu_seqlens,
            q_length=token_count,
            kv_length=token_count,
            device=x.device,
            dropout=self.config.dropout if self.training else 0.0,
        )
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = _checkpointed(
                lambda value, block=block: block.forward_packed(
                    value,
                    plan,
                ),
                x,
                enabled=self.gradient_checkpointing and self.training,
                use_reentrant=self.checkpoint_use_reentrant,
            )
            if capture_layers is not None and index in capture_layers:
                features[index] = x
        x = self.norm(x)
        if capture_layers is not None and -1 in capture_layers:
            features[-1] = x
        if capture_layers is not None:
            missing = capture_layers - features.keys()
            if missing:
                raise ValueError(f"unknown vision feature layers: {sorted(missing)}")
        self.last_packed_attention_backend = (
            "flex" if plan.use_flex else "loop"
        )
        return x, features


class CrossAttention(nn.Module):
    def __init__(self, query_width: int, source_width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = query_width // heads
        self.q_proj = nn.Linear(query_width, query_width)
        self.k_proj = nn.Linear(source_width, query_width)
        self.v_proj = nn.Linear(source_width, query_width)
        self.o_proj = nn.Linear(query_width, query_width)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, query_length, width = query.shape
        source_length = source.shape[1]
        q = self.q_proj(query).view(
            batch, query_length, self.heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(source).view(
            batch, source_length, self.heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(source).view(
            batch, source_length, self.heads, self.head_dim
        ).transpose(1, 2)
        attention_bias = None
        if source_mask is not None:
            minimum = torch.finfo(q.dtype).min
            attention_bias = (
                1.0 - source_mask[:, None, None, :].to(q.dtype)
            ) * minimum
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        return self.o_proj(
            out.transpose(1, 2).contiguous().view(batch, query_length, width)
        )

    def forward_packed(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        plan: _PackedAttentionPlan,
    ) -> torch.Tensor:
        batch, query_length, width = query.shape
        source_length = source.shape[0]
        q = self.q_proj(query).view(
            1,
            batch * query_length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(source).view(
            1,
            source_length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(source).view(
            1,
            source_length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)
        out = _packed_attention(q, k, v, plan)
        return self.o_proj(
            out.transpose(1, 2)
            .contiguous()
            .view(batch, query_length, width)
        )


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, hidden)
        self.up_proj = nn.Linear(width, hidden)
        self.down_proj = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ResamplerLayer(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.output_width)
        self.cross_attn = CrossAttention(
            config.output_width,
            config.input_width,
            config.attention_heads,
        )
        self.cross_gate = nn.Parameter(torch.tensor(float(config.gate_init)))
        self.norm2 = RMSNorm(config.output_width)
        self.mlp = SwiGLU(config.output_width, config.mlp_width)

    def forward(
        self,
        latents: torch.Tensor,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate = torch.tanh(self.cross_gate)
        latents = latents + gate * self.cross_attn(
            self.norm1(latents),
            vision_tokens,
            vision_mask,
        )
        return latents + self.mlp(self.norm2(latents))

    def forward_packed(
        self,
        latents: torch.Tensor,
        vision_tokens: torch.Tensor,
        plan: _PackedAttentionPlan,
    ) -> torch.Tensor:
        gate = torch.tanh(self.cross_gate)
        latents = latents + gate * self.cross_attn.forward_packed(
            self.norm1(latents),
            vision_tokens,
            plan,
        )
        return latents + self.mlp(self.norm2(latents))


class GatedResampler(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.latents = nn.Parameter(torch.empty(config.latent_tokens, config.output_width))
        self.layers = nn.ModuleList([ResamplerLayer(config) for _ in range(config.layers)])
        self.last_packed_attention_backend = "none"
        self.gradient_checkpointing = False
        self.checkpoint_use_reentrant = False

    @property
    def gradient_probe_anchor(self) -> nn.Parameter:
        return self.layers[-1].norm2.weight

    def forward(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = self.latents.unsqueeze(0).expand(vision_tokens.shape[0], -1, -1)
        for layer in self.layers:
            latents = _checkpointed(
                lambda latent_values, source, layer=layer: layer(
                    latent_values,
                    source,
                    vision_mask,
                ),
                latents,
                vision_tokens,
                enabled=self.gradient_checkpointing and self.training,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        return latents

    def forward_packed(
        self,
        vision_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        attention_backend: str = "auto",
    ) -> torch.Tensor:
        """Resample packed image sequences without materializing padded sources."""

        batch_size = int(cu_seqlens.numel() - 1)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        latent_cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * self.latents.shape[0],
            self.latents.shape[0],
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        plan = _prepare_packed_attention(
            attention_backend,
            latent_cu_seqlens,
            cu_seqlens,
            q_length=batch_size * self.latents.shape[0],
            kv_length=vision_tokens.shape[0],
            device=vision_tokens.device,
        )
        for layer in self.layers:
            latents = _checkpointed(
                lambda latent_values, source, layer=layer: (
                    layer.forward_packed(latent_values, source, plan)
                ),
                latents,
                vision_tokens,
                enabled=self.gradient_checkpointing and self.training,
                use_reentrant=self.checkpoint_use_reentrant,
            )
        self.last_packed_attention_backend = (
            "flex" if plan.use_flex else "loop"
        )
        return latents


class AveragePoolProjector(nn.Module):
    """Ordered adaptive pooling followed by one vision-to-language projection."""

    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.latent_tokens = config.latent_tokens
        self.projection = nn.Linear(config.input_width, config.output_width)
        self.last_packed_attention_backend = "pool"
        self.gradient_checkpointing = False
        self.checkpoint_use_reentrant = False

    @property
    def gradient_probe_anchor(self) -> nn.Parameter:
        return self.projection.weight

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[0] == 0:
            raise ValueError("average-pool connector requires non-empty token sequences")
        pooled = F.adaptive_avg_pool1d(
            tokens.T.unsqueeze(0),
            self.latent_tokens,
        )
        return pooled.squeeze(0).T

    def _project(self, pooled: torch.Tensor) -> torch.Tensor:
        return _checkpointed(
            self.projection,
            pooled,
            enabled=self.gradient_checkpointing and self.training,
            use_reentrant=self.checkpoint_use_reentrant,
        )

    def forward(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vision_tokens.ndim != 3:
            raise ValueError("vision_tokens must have shape [batch, tokens, width]")
        if vision_mask is not None and vision_mask.shape != vision_tokens.shape[:2]:
            raise ValueError("vision_mask must match vision token dimensions")
        pooled = []
        for index, tokens in enumerate(vision_tokens):
            valid = (
                tokens
                if vision_mask is None
                else tokens[vision_mask[index].to(dtype=torch.bool)]
            )
            pooled.append(self._pool(valid))
        return self._project(torch.stack(pooled))

    def forward_packed(
        self,
        vision_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        attention_backend: str = "auto",
    ) -> torch.Tensor:
        del attention_backend
        if vision_tokens.ndim != 2:
            raise ValueError("packed vision_tokens must have shape [tokens, width]")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least one sequence")
        pooled = [
            self._pool(
                vision_tokens[
                    int(cu_seqlens[index].item()) : int(
                        cu_seqlens[index + 1].item()
                    )
                ]
            )
            for index in range(cu_seqlens.numel() - 1)
        ]
        self.last_packed_attention_backend = "pool"
        return self._project(torch.stack(pooled))


def build_connector(config: ConnectorConfig) -> nn.Module:
    if config.family == "gated_resampler":
        return GatedResampler(config)
    if config.family == "average_pool_projector":
        return AveragePoolProjector(config)
    raise ValueError(f"unsupported connector family {config.family!r}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _multi_positive_contrastive_loss(
    similarities: torch.Tensor,
    query_ids: torch.Tensor,
    key_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    key_ids = query_ids if key_ids is None else key_ids
    positive = query_ids[:, None] == key_ids[None, :]
    if not torch.all(positive.any(dim=-1)):
        raise ValueError("every contrastive query must have at least one positive key")
    minimum = torch.finfo(similarities.dtype).min
    positive_score = torch.logsumexp(
        similarities.masked_fill(~positive, minimum),
        dim=-1,
    )
    all_score = torch.logsumexp(similarities, dim=-1)
    return (all_score - positive_score).mean()


def _multi_positive_siglip_loss(
    logits: torch.Tensor,
    query_ids: torch.Tensor,
    key_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    key_ids = query_ids if key_ids is None else key_ids
    positive = query_ids[:, None] == key_ids[None, :]
    if not torch.all(positive.any(dim=-1)):
        raise ValueError("every contrastive query must have at least one positive key")
    signs = torch.where(positive, 1.0, -1.0).to(dtype=logits.dtype)
    return -F.logsigmoid(signs * logits).sum() / logits.shape[0]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / float(head_dim)
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[int, torch.device, torch.dtype, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            cached_length, cached_device, cached_dtype, cos, sin = self._cache
            if cached_length >= length and cached_device == device and cached_dtype == dtype:
                return cos[..., :length, :], sin[..., :length, :]
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(device=device))
        cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype)[None, None]
        sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype)[None, None]
        self._cache = (length, device, dtype, cos, sin)
        return cos, sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.kv_heads
        self.head_dim = config.width // config.attention_heads
        self.kv_width = self.kv_heads * self.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.width, config.width)
        self.k_proj = nn.Linear(config.width, self.kv_width)
        self.v_proj = nn.Linear(config.width, self.kv_width)
        self.o_proj = nn.Linear(config.width, config.width)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        is_causal: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if past_key_value is not None:
            past_key, past_value = past_key_value
            expected = (batch, self.kv_heads, self.head_dim)
            if (
                past_key.ndim != 4
                or past_value.shape != past_key.shape
                or (
                    past_key.shape[0],
                    past_key.shape[1],
                    past_key.shape[3],
                )
                != expected
            ):
                raise ValueError("invalid grouped-query attention KV cache")
            k = torch.cat((past_key, k), dim=2)
            v = torch.cat((past_value, v), dim=2)
        present = (k, v)
        out = _scaled_dot_product_gqa(
            q,
            k,
            v,
            attention_bias=attention_bias,
            dropout=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        projected = self.o_proj(
            out.transpose(1, 2).contiguous().view(batch, length, width)
        )
        return (projected, present) if use_cache else projected

    def forward_static_cache(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_position: int,
    ) -> torch.Tensor:
        """Attend one query while updating a preallocated compact GQA cache."""

        batch, length, width = x.shape
        if length != 1:
            raise ValueError("static KV-cache attention requires one query")
        expected = (batch, self.kv_heads, self.head_dim)
        if (
            key_cache.ndim != 4
            or value_cache.shape != key_cache.shape
            or (
                key_cache.shape[0],
                key_cache.shape[1],
                key_cache.shape[3],
            )
            != expected
            or not 0 <= cache_position < key_cache.shape[2]
        ):
            raise ValueError("invalid static grouped-query attention KV cache")
        q = self.q_proj(x).view(
            batch,
            length,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch,
            length,
            self.kv_heads,
            self.head_dim,
        ).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        key_cache[:, :, cache_position : cache_position + 1].copy_(k)
        value_cache[:, :, cache_position : cache_position + 1].copy_(v)
        key = key_cache[:, :, : cache_position + 1]
        value = value_cache[:, :, : cache_position + 1]
        out = _scaled_dot_product_gqa(
            q,
            key,
            value,
            attention_bias=attention_bias,
            dropout=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.o_proj(
            out.transpose(1, 2).contiguous().view(batch, length, width)
        )


class GatedShortConvolution(nn.Module):
    """LFM-style gated causal depthwise convolution."""

    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.width = config.width
        self.kernel_size = config.conv_kernel_size
        self.in_proj = nn.Linear(
            config.width,
            3 * config.width,
            bias=config.conv_bias,
        )
        self.conv = nn.Conv1d(
            config.width,
            config.width,
            self.kernel_size,
            groups=config.width,
            bias=config.conv_bias,
            padding=self.kernel_size - 1,
        )
        self.out_proj = nn.Linear(
            config.width,
            config.width,
            bias=config.conv_bias,
        )

    def _project(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_mask is not None:
            if token_mask.shape != x.shape[:2]:
                raise ValueError(
                    "short-convolution token mask must match hidden states"
                )
            x = x * token_mask[:, :, None].to(x.dtype)
        gate_in, gate_out, value = self.in_proj(x).chunk(3, dim=-1)
        return gate_out, gate_in * value

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        gate, mixed = self._project(x, token_mask)
        length = x.shape[1]
        convolved = self.conv(mixed.transpose(1, 2))[..., :length]
        return self.out_proj(
            gate * convolved.transpose(1, 2).contiguous()
        )

    def prefill(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, mixed = self._project(x, token_mask)
        length = x.shape[1]
        mixed_channels = mixed.transpose(1, 2)
        convolved = self.conv(mixed_channels)[..., :length]
        history = F.pad(
            mixed_channels,
            (max(self.kernel_size - 1 - length, 0), 0),
        )[..., -(self.kernel_size - 1) :].contiguous()
        output = self.out_proj(
            gate * convolved.transpose(1, 2).contiguous()
        )
        return output, history

    def decode(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError(
                "cached short-convolution decode requires one token"
            )
        expected = (x.shape[0], self.width, self.kernel_size - 1)
        if state.shape != expected:
            raise ValueError("invalid short-convolution cache state")
        gate, mixed = self._project(x, token_mask)
        current = mixed.transpose(1, 2)
        window = torch.cat((state, current), dim=-1)
        weight = self.conv.weight[:, 0, :][None]
        convolved = (window * weight).sum(dim=-1, keepdim=True)
        if self.conv.bias is not None:
            convolved = convolved + self.conv.bias[None, :, None]
        state.copy_(window[..., 1:])
        return self.out_proj(
            gate * convolved.transpose(1, 2).contiguous()
        )


@dataclass(frozen=True)
class AttentionLayerCache:
    key: torch.Tensor
    value: torch.Tensor

    @property
    def tensor_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.key, self.value)
        )


@dataclass(frozen=True)
class ShortConvLayerCache:
    state: torch.Tensor

    @property
    def tensor_bytes(self) -> int:
        return self.state.numel() * self.state.element_size()


LanguageLayerCache = AttentionLayerCache | ShortConvLayerCache


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: LanguageConfig,
        layer_type: str,
    ):
        super().__init__()
        if layer_type not in {"attention", "short_conv"}:
            raise ValueError(f"unsupported language layer type {layer_type!r}")
        self.layer_type = layer_type
        self.norm1 = RMSNorm(config.width)
        if self.is_attention:
            self.attn = GroupedQueryAttention(config)
        else:
            self.conv = GatedShortConvolution(config)
        self.norm2 = RMSNorm(config.width)
        self.mlp = SwiGLU(config.width, config.mlp_width)

    @property
    def is_attention(self) -> bool:
        return self.layer_type == "attention"

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        is_causal: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        mixed = (
            self.attn(
                normalized,
                attention_bias,
                is_causal,
                cos,
                sin,
            )
            if self.is_attention
            else self.conv(normalized, token_mask)
        )
        x = x + mixed
        return x + self.mlp(self.norm2(x))

    def forward_cached(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        is_causal: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, LanguageLayerCache]:
        normalized = self.norm1(x)
        if self.is_attention:
            attention, present = self.attn(
                normalized,
                attention_bias,
                is_causal,
                cos,
                sin,
                use_cache=True,
            )
            key, value = present
            mixed = attention
            cache: LanguageLayerCache = AttentionLayerCache(key, value)
        else:
            mixed, state = self.conv.prefill(normalized, token_mask)
            cache = ShortConvLayerCache(state)
        x = x + mixed
        return x + self.mlp(self.norm2(x)), cache

    def forward_static_cache(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_cache: LanguageLayerCache,
        cache_position: int,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = self.norm1(x)
        if self.is_attention:
            if not isinstance(layer_cache, AttentionLayerCache):
                raise ValueError(
                    "attention layer received a convolution cache"
                )
            mixed = self.attn.forward_static_cache(
                normalized,
                attention_bias,
                cos,
                sin,
                layer_cache.key,
                layer_cache.value,
                cache_position,
            )
        else:
            if not isinstance(layer_cache, ShortConvLayerCache):
                raise ValueError(
                    "short-convolution layer received an attention cache"
                )
            mixed = self.conv.decode(
                normalized,
                layer_cache.state,
                token_mask,
            )
        x = x + mixed
        return x + self.mlp(self.norm2(x))


@dataclass(frozen=True)
class LanguageKVCache:
    layers: tuple[LanguageLayerCache, ...]
    sequence_length: int
    capacity: int

    @property
    def tensor_bytes(self) -> int:
        return sum(layer.tensor_bytes for layer in self.layers)


class LanguageDecoder(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config, layer_type)
                for layer_type in config.layer_types
            ]
        )
        self.norm = RMSNorm(config.width)
        self.rotary = RotaryEmbedding(
            config.width // config.attention_heads,
            config.rope_base,
        )
        self.gradient_checkpointing = False
        self.checkpoint_use_reentrant = False

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        capture_layers: set[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        x = embeddings
        attention_bias, is_causal = self._attention_plan(
            x,
            attention_mask,
        )
        cos, sin = self.rotary(x.shape[1], x.device, x.dtype)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = _checkpointed(
                lambda value, block=block: block(
                    value,
                    attention_bias,
                    is_causal,
                    cos,
                    sin,
                    attention_mask,
                ),
                x,
                enabled=self.gradient_checkpointing and self.training,
                use_reentrant=self.checkpoint_use_reentrant,
            )
            if capture_layers is not None and index in capture_layers:
                features[index] = x
        x = self.norm(x)
        if capture_layers is not None and -1 in capture_layers:
            features[-1] = x
        if capture_layers is not None:
            missing = capture_layers - features.keys()
            if missing:
                raise ValueError(f"unknown language feature layers: {sorted(missing)}")
            return x, features
        return x

    @staticmethod
    def _attention_plan(
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, bool]:
        if attention_mask is None:
            return None, True
        length = x.shape[1]
        if attention_mask.shape != (x.shape[0], length):
            raise ValueError("language attention_mask must match embeddings")
        minimum = torch.finfo(x.dtype).min
        causal = torch.full(
            (length, length),
            minimum,
            dtype=x.dtype,
            device=x.device,
        ).triu(1)
        padding = (
            1.0 - attention_mask[:, None, None, :].to(x.dtype)
        ) * minimum
        return causal[None, None] + padding, False

    def prefill(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_cache_length: int | None = None,
    ) -> tuple[torch.Tensor, LanguageKVCache]:
        """Prefill compact attention K/V and convolution recurrent state."""

        if torch.is_grad_enabled():
            raise RuntimeError("language KV cache is inference-only")
        x = embeddings
        attention_bias, is_causal = self._attention_plan(
            x,
            attention_mask,
        )
        cos, sin = self.rotary(x.shape[1], x.device, x.dtype)
        sequence_length = int(x.shape[1])
        capacity = (
            sequence_length
            if max_cache_length is None
            else int(max_cache_length)
        )
        if capacity < sequence_length:
            raise ValueError("language KV cache cannot be shorter than prefill")
        present: list[LanguageLayerCache] = []
        for block in self.blocks:
            x, layer_cache = block.forward_cached(
                x,
                attention_bias,
                is_causal,
                cos,
                sin,
                attention_mask,
            )
            if (
                isinstance(layer_cache, AttentionLayerCache)
                and capacity > sequence_length
            ):
                key_buffer = layer_cache.key.new_empty(
                    layer_cache.key.shape[0],
                    layer_cache.key.shape[1],
                    capacity,
                    layer_cache.key.shape[3],
                )
                value_buffer = layer_cache.value.new_empty(
                    layer_cache.value.shape[0],
                    layer_cache.value.shape[1],
                    capacity,
                    layer_cache.value.shape[3],
                )
                key_buffer[:, :, :sequence_length].copy_(
                    layer_cache.key
                )
                value_buffer[:, :, :sequence_length].copy_(
                    layer_cache.value
                )
                layer_cache = AttentionLayerCache(
                    key_buffer,
                    value_buffer,
                )
            present.append(layer_cache)
        return self.norm(x), LanguageKVCache(
            tuple(present),
            sequence_length=sequence_length,
            capacity=capacity,
        )

    def decode(
        self,
        embeddings: torch.Tensor,
        cache: LanguageKVCache,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LanguageKVCache]:
        """Decode one token against a prefilled hybrid language cache."""

        if torch.is_grad_enabled():
            raise RuntimeError("language KV cache is inference-only")
        if embeddings.ndim != 3 or embeddings.shape[1] != 1:
            raise ValueError("cached language decode requires one token")
        if len(cache.layers) != len(self.blocks):
            raise ValueError("language KV cache layer count mismatch")
        past_length = cache.sequence_length
        total_length = past_length + 1
        if total_length > cache.capacity:
            raise ValueError("language KV cache capacity exceeded")
        if attention_mask is not None and attention_mask.shape != (
            embeddings.shape[0],
            total_length,
        ):
            raise ValueError(
                "cached language attention_mask must cover past and current tokens"
            )
        attention_bias = None
        if attention_mask is not None:
            minimum = torch.finfo(embeddings.dtype).min
            attention_bias = (
                1.0
                - attention_mask[:, None, None, :].to(embeddings.dtype)
            ) * minimum
        cos, sin = self.rotary(
            total_length,
            embeddings.device,
            embeddings.dtype,
        )
        cos = cos[..., past_length:total_length, :]
        sin = sin[..., past_length:total_length, :]
        x = embeddings
        for block, layer_cache in zip(self.blocks, cache.layers):
            if (
                isinstance(layer_cache, AttentionLayerCache)
                and int(layer_cache.key.shape[2]) != cache.capacity
            ):
                raise ValueError(
                    "language attention-cache capacities disagree"
                )
            x = block.forward_static_cache(
                x,
                attention_bias,
                cos,
                sin,
                layer_cache,
                past_length,
                (
                    attention_mask[:, -1:]
                    if attention_mask is not None
                    else None
                ),
            )
        return self.norm(x), LanguageKVCache(
            cache.layers,
            sequence_length=total_length,
            capacity=cache.capacity,
        )


@dataclass
class StudentOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    losses: dict[str, torch.Tensor] = field(default_factory=dict)
    box_predictions: torch.Tensor | None = None
    orientation_logits: torch.Tensor | None = None
    vision_embeddings: torch.Tensor | None = None
    text_embeddings: torch.Tensor | None = None
    vision_features: dict[int, torch.Tensor] = field(default_factory=dict)
    language_features: dict[int, torch.Tensor] = field(default_factory=dict)
    vision_mask: torch.Tensor | None = None
    visual_attention_backend: str = "none"


@dataclass(frozen=True)
class GenerationState:
    cache: LanguageKVCache
    attention_mask: torch.Tensor | None


class DocumentVLMStudent(nn.Module):
    """End-to-end image-prefix causal LM with removable auxiliary pretraining heads."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        errors = config.validate()
        if errors:
            raise ValueError("invalid student config: " + "; ".join(errors))
        self.config = config
        self.vision = VisionTower(config.vision)
        self.connector = build_connector(config.connector)
        self.language = LanguageDecoder(config.language)
        self.lm_head = nn.Linear(config.language.width, config.language.vocab_size, bias=False)
        if config.language.tied_embeddings:
            self.lm_head.weight = self.language.token_embedding.weight

        heads = config.task_heads
        self.vision_projection = (
            nn.Linear(config.vision.width, heads.contrastive_width)
            if heads.region_text_contrastive
            else None
        )
        self.text_projection = (
            nn.Linear(config.language.width, heads.contrastive_width)
            if heads.region_text_contrastive
            else None
        )
        self.orientation_head = (
            nn.Linear(config.vision.width, 4) if heads.orientation else None
        )
        self.last_visual_attention_backend = "none"
        self._gradient_checkpointing_enabled = False
        self._gradient_checkpointing_components = (
            "vision",
            "connector",
            "language",
        )
        self._gradient_checkpointing_use_reentrant = False
        self.box_head = (
            nn.Linear(config.language.width, 4) if heads.box_regression else None
        )
        self.contrastive_objective = heads.contrastive_objective
        if heads.region_text_contrastive:
            self.contrastive_logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / heads.contrastive_temperature))
            )
            self.contrastive_logit_bias = nn.Parameter(
                torch.tensor(heads.contrastive_bias_init)
            )
        else:
            self.register_parameter("contrastive_logit_scale", None)
            self.register_parameter("contrastive_logit_bias", None)
        self.apply(self._init_weights)
        if isinstance(self.connector, GatedResampler):
            nn.init.normal_(self.connector.latents, std=0.02)
        nn.init.zeros_(self.vision.position_embedding)

    def configure_gradient_checkpointing(
        self,
        *,
        enabled: bool,
        components: tuple[str, ...] = (
            "vision",
            "connector",
            "language",
        ),
        use_reentrant: bool = False,
    ) -> None:
        """Configure component-level block recomputation for training."""

        supported = {"vision", "connector", "language"}
        requested = set(components)
        if len(requested) != len(components):
            raise ValueError(
                "gradient checkpointing components must be unique"
            )
        if not requested <= supported:
            raise ValueError(
                "unsupported gradient checkpointing components: "
                f"{sorted(requested - supported)}"
            )
        if enabled and not requested:
            raise ValueError(
                "enabled gradient checkpointing requires at least one component"
            )
        self._gradient_checkpointing_enabled = bool(enabled)
        self._gradient_checkpointing_components = tuple(components)
        self._gradient_checkpointing_use_reentrant = bool(
            use_reentrant
        )
        for name in supported:
            module = getattr(self, name)
            module.gradient_checkpointing = enabled and name in requested
            module.checkpoint_use_reentrant = bool(use_reentrant)

    @property
    def gradient_checkpointing_state(self) -> dict[str, Any]:
        return {
            "enabled": self._gradient_checkpointing_enabled,
            "components": list(
                self._gradient_checkpointing_components
            ),
            "use_reentrant": (
                self._gradient_checkpointing_use_reentrant
            ),
        }

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(
            module,
            (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d),
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _last_text_state(
        self,
        hidden: torch.Tensor,
        prefix_length: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        text = hidden[:, prefix_length:]
        if attention_mask is None:
            return text[:, -1]
        indices = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        return text[torch.arange(text.shape[0], device=text.device), indices]

    @staticmethod
    def _text_state_at(
        hidden: torch.Tensor,
        prefix_length: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        text = hidden[:, prefix_length:]
        if positions.shape != (text.shape[0],):
            raise ValueError("box_query_positions must have shape [batch]")
        if torch.any(positions < 0) or torch.any(positions >= text.shape[1]):
            raise ValueError("box_query_positions contains an out-of-range text position")
        return text[
            torch.arange(text.shape[0], device=text.device),
            positions.to(device=text.device, dtype=torch.long),
        ]

    def encode_images(
        self,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        *,
        packed_pixel_values: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_attention_backend: str = "auto",
    ) -> torch.Tensor:
        """Encode an image batch once into the fixed visual prefix."""

        packed_inputs = (
            packed_pixel_values,
            packed_position_ids,
            packed_cu_seqlens,
        )
        if any(value is not None for value in packed_inputs):
            if pixel_values is not None or pixel_mask is not None:
                raise ValueError("dense and packed visual inputs are mutually exclusive")
            if not all(value is not None for value in packed_inputs):
                raise ValueError("all packed visual inputs must be provided together")
            vision_tokens, _ = self.vision.forward_packed(
                packed_pixel_values,
                packed_position_ids,
                packed_cu_seqlens,
                attention_backend=packed_attention_backend,
            )
            prefix = self.connector.forward_packed(
                vision_tokens,
                packed_cu_seqlens,
                attention_backend=packed_attention_backend,
            )
            self.last_visual_attention_backend = (
                "flex"
                if self.vision.last_packed_attention_backend == "flex"
                and self.connector.last_packed_attention_backend in {"flex", "pool"}
                else "loop"
            )
            return prefix
        if pixel_values is None:
            raise ValueError("encode_images requires dense or packed visual inputs")
        vision_tokens, vision_mask = self.vision(
            pixel_values,
            pixel_mask,
            return_mask=True,
        )
        self.last_visual_attention_backend = "dense"
        return self.connector(vision_tokens, vision_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        packed_pixel_values: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_attention_backend: str = "auto",
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        box_targets: torch.Tensor | None = None,
        box_target_mask: torch.Tensor | None = None,
        box_query_positions: torch.Tensor | None = None,
        orientation_labels: torch.Tensor | None = None,
        contrastive: bool = False,
        contrastive_ids: torch.Tensor | None = None,
        contrastive_vision_keys: torch.Tensor | None = None,
        contrastive_text_keys: torch.Tensor | None = None,
        contrastive_key_ids: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
        box_iou_loss_kind: str = "giou",
        feature_layers: dict[str, list[int] | tuple[int, ...]] | None = None,
        visual_prefix: torch.Tensor | None = None,
    ) -> StudentOutput:
        weights = {
            "autoregressive": 1.0,
            "box_regression": 1.0,
            "orientation": 1.0,
            "region_text_contrastive": 1.0,
            **(loss_weights or {}),
        }
        vision_tokens = None
        vision_mask = None
        vision_features: dict[int, torch.Tensor] = {}
        language_features: dict[int, torch.Tensor] = {}
        requested_vision = set((feature_layers or {}).get("vision", ()))
        requested_language = set((feature_layers or {}).get("language", ()))
        packed_inputs = (
            packed_pixel_values,
            packed_position_ids,
            packed_cu_seqlens,
        )
        has_packed = any(value is not None for value in packed_inputs)
        if has_packed and not all(value is not None for value in packed_inputs):
            raise ValueError("all packed visual inputs must be provided together")
        if has_packed and (pixel_values is not None or pixel_mask is not None):
            raise ValueError("dense and packed visual inputs are mutually exclusive")
        if has_packed and packed_cu_seqlens.numel() != input_ids.shape[0] + 1:
            raise ValueError("packed visual batch dimension must match input_ids")
        if visual_prefix is not None:
            if pixel_values is not None or pixel_mask is not None or has_packed:
                raise ValueError(
                    "visual_prefix cannot be combined with dense or packed visual inputs"
                )
            if visual_prefix.ndim != 3:
                raise ValueError("visual_prefix must have shape [batch, tokens, width]")
            if visual_prefix.shape[0] != input_ids.shape[0]:
                raise ValueError("visual_prefix batch dimension must match input_ids")
            if visual_prefix.shape[-1] != self.config.language.width:
                raise ValueError("visual_prefix width must match the language width")
            if requested_vision or orientation_labels is not None or contrastive:
                raise ValueError(
                    "cached visual_prefix is inference-only for vision-side outputs"
                )
            prefix = visual_prefix
            self.last_visual_attention_backend = "cached"
        elif pixel_values is not None:
            vision_result = self.vision(
                pixel_values,
                pixel_mask,
                return_mask=True,
                capture_layers=requested_vision if requested_vision else None,
            )
            if requested_vision:
                vision_tokens, vision_mask, vision_features = vision_result
            else:
                vision_tokens, vision_mask = vision_result
            prefix = self.connector(vision_tokens, vision_mask)
            self.last_visual_attention_backend = "dense"
        elif has_packed:
            vision_tokens, vision_features = self.vision.forward_packed(
                packed_pixel_values,
                packed_position_ids,
                packed_cu_seqlens,
                attention_backend=packed_attention_backend,
                capture_layers=requested_vision if requested_vision else None,
            )
            vision_mask = torch.ones(
                vision_tokens.shape[0],
                dtype=torch.bool,
                device=vision_tokens.device,
            )
            prefix = self.connector.forward_packed(
                vision_tokens,
                packed_cu_seqlens,
                attention_backend=packed_attention_backend,
            )
            self.last_visual_attention_backend = (
                "flex"
                if self.vision.last_packed_attention_backend == "flex"
                and self.connector.last_packed_attention_backend in {"flex", "pool"}
                else "loop"
            )
        elif pixel_mask is not None:
            raise ValueError("pixel_mask requires pixel_values")
        else:
            prefix = None
            self.last_visual_attention_backend = "none"
        text_embeddings = self.language.token_embedding(input_ids)
        prefix_length = 0 if prefix is None else prefix.shape[1]
        embeddings = (
            text_embeddings if prefix is None else torch.cat((prefix, text_embeddings), dim=1)
        )
        full_mask = attention_mask
        if attention_mask is not None and prefix_length:
            prefix_mask = torch.ones(
                attention_mask.shape[0],
                prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        language_result = self.language(
            embeddings,
            full_mask,
            capture_layers=requested_language if requested_language else None,
        )
        if requested_language:
            hidden, language_features = language_result
        else:
            hidden = language_result
        logits = self.lm_head(hidden)
        losses: dict[str, torch.Tensor] = {}

        if labels is not None:
            if prefix_length:
                ignored = torch.full(
                    (labels.shape[0], prefix_length),
                    self.config.ignore_index,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                labels = torch.cat((ignored, labels), dim=1)
            losses["autoregressive"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=self.config.ignore_index,
            )

        pooled_text = self._last_text_state(hidden, prefix_length, attention_mask)
        pooled_box = (
            self._text_state_at(hidden, prefix_length, box_query_positions)
            if box_query_positions is not None
            else pooled_text
        )
        box_predictions = (
            decode_normalized_box(self.box_head(pooled_box))
            if self.box_head is not None
            else None
        )
        if box_targets is not None:
            if box_predictions is None:
                raise ValueError("box_targets were provided but the box head is disabled")
            valid_boxes = (
                torch.ones(
                    box_targets.shape[0],
                    dtype=torch.bool,
                    device=box_targets.device,
                )
                if box_target_mask is None
                else box_target_mask.to(device=box_targets.device, dtype=torch.bool)
            )
            if valid_boxes.shape != (box_targets.shape[0],):
                raise ValueError("box_target_mask must have shape [batch]")
            if torch.any(valid_boxes):
                losses["box_regression"] = (
                    F.smooth_l1_loss(
                        box_predictions[valid_boxes],
                        box_targets[valid_boxes],
                    )
                    + box_iou_loss(
                        box_predictions[valid_boxes],
                        box_targets[valid_boxes],
                        kind=box_iou_loss_kind,
                    )
                )

        orientation_logits = None
        vision_embeddings = None
        text_projected = None
        if vision_tokens is not None:
            if has_packed:
                boundaries = [
                    int(value) for value in packed_cu_seqlens.tolist()
                ]
                pooled_vision = torch.stack(
                    [
                        vision_tokens[start:end].mean(dim=0)
                        for start, end in zip(boundaries, boundaries[1:])
                    ]
                )
            elif vision_mask is None:
                pooled_vision = vision_tokens.mean(dim=1)
            else:
                denominator = vision_mask.sum(dim=1, keepdim=True).clamp_min(1)
                pooled_vision = (
                    vision_tokens * vision_mask.unsqueeze(-1)
                ).sum(dim=1) / denominator
            orientation_logits = (
                self.orientation_head(pooled_vision)
                if self.orientation_head is not None
                else None
            )
            if orientation_labels is not None:
                if orientation_logits is None:
                    raise ValueError(
                        "orientation_labels were provided but the orientation head is disabled"
                    )
                valid_orientation = orientation_labels != self.config.ignore_index
                if torch.any(valid_orientation):
                    losses["orientation"] = F.cross_entropy(
                        orientation_logits[valid_orientation],
                        orientation_labels[valid_orientation],
                    )
            if self.vision_projection is not None and self.text_projection is not None:
                vision_embeddings = F.normalize(
                    self.vision_projection(pooled_vision), dim=-1
                )
                text_projected = F.normalize(self.text_projection(pooled_text), dim=-1)
            if contrastive:
                if vision_embeddings is None or text_projected is None:
                    raise ValueError(
                        "contrastive loss was requested but contrastive heads are disabled"
                    )
                if (
                    self.contrastive_logit_scale is None
                    or self.contrastive_logit_bias is None
                ):
                    raise RuntimeError("contrastive parameters are unavailable")
                scale = self.contrastive_logit_scale.exp().clamp(max=100.0)
                similarities = (vision_embeddings @ text_projected.T) * scale
                group_ids = (
                    torch.arange(
                        similarities.shape[0],
                        device=similarities.device,
                    )
                    if contrastive_ids is None
                    else contrastive_ids.to(device=similarities.device)
                )
                if group_ids.shape != (similarities.shape[0],):
                    raise ValueError("contrastive_ids must have shape [batch]")
                memory_values = (
                    contrastive_vision_keys,
                    contrastive_text_keys,
                    contrastive_key_ids,
                )
                if any(value is not None for value in memory_values):
                    if any(value is None for value in memory_values):
                        raise ValueError(
                            "contrastive memory requires vision keys, text keys, and ids"
                        )
                    memory_vision = contrastive_vision_keys.to(
                        device=similarities.device,
                        dtype=vision_embeddings.dtype,
                    )
                    memory_text = contrastive_text_keys.to(
                        device=similarities.device,
                        dtype=text_projected.dtype,
                    )
                    memory_ids = contrastive_key_ids.to(
                        device=similarities.device,
                        dtype=group_ids.dtype,
                    )
                    expected_width = vision_embeddings.shape[1]
                    if (
                        memory_vision.ndim != 2
                        or memory_text.ndim != 2
                        or memory_vision.shape != memory_text.shape
                        or memory_vision.shape[1] != expected_width
                        or memory_ids.shape != (memory_vision.shape[0],)
                    ):
                        raise ValueError(
                            "contrastive memory tensors have incompatible shapes"
                        )
                    key_ids = torch.cat((group_ids, memory_ids), dim=0)
                    image_to_text = torch.cat(
                        (
                            similarities,
                            (vision_embeddings @ memory_text.T) * scale,
                        ),
                        dim=1,
                    )
                    text_to_image = torch.cat(
                        (
                            similarities.T,
                            (text_projected @ memory_vision.T) * scale,
                        ),
                        dim=1,
                    )
                else:
                    key_ids = group_ids
                    image_to_text = similarities
                    text_to_image = similarities.T
                if self.contrastive_objective == "siglip":
                    bias = self.contrastive_logit_bias
                    losses["region_text_contrastive"] = 0.5 * (
                        _multi_positive_siglip_loss(
                            image_to_text + bias,
                            group_ids,
                            key_ids,
                        )
                        + _multi_positive_siglip_loss(
                            text_to_image + bias,
                            group_ids,
                            key_ids,
                        )
                    )
                else:
                    losses["region_text_contrastive"] = 0.5 * (
                        _multi_positive_contrastive_loss(
                            image_to_text,
                            group_ids,
                            key_ids,
                        )
                        + _multi_positive_contrastive_loss(
                            text_to_image,
                            group_ids,
                            key_ids,
                        )
                    )

        total_loss = None
        for name, value in losses.items():
            weighted = value * float(weights.get(name, 1.0))
            total_loss = weighted if total_loss is None else total_loss + weighted
        return StudentOutput(
            logits=logits,
            loss=total_loss,
            losses=losses,
            box_predictions=box_predictions,
            orientation_logits=orientation_logits,
            vision_embeddings=vision_embeddings,
            text_embeddings=text_projected,
            vision_features=vision_features,
            language_features=language_features,
            vision_mask=vision_mask,
            visual_attention_backend=self.last_visual_attention_backend,
        )

    @torch.no_grad()
    def prefill_generation(
        self,
        input_ids: torch.Tensor,
        *,
        visual_prefix: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 1,
    ) -> tuple[torch.Tensor, GenerationState]:
        """Prefill prompt and visual tokens once for incremental decoding."""

        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError(
                "generation input_ids must have shape [batch, nonzero tokens]"
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("generation attention_mask must match input_ids")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if visual_prefix is not None:
            if (
                visual_prefix.ndim != 3
                or visual_prefix.shape[0] != input_ids.shape[0]
                or visual_prefix.shape[-1] != self.config.language.width
            ):
                raise ValueError(
                    "visual_prefix must have shape [batch, tokens, language width]"
                )
        text_embeddings = self.language.token_embedding(input_ids)
        embeddings = (
            text_embeddings
            if visual_prefix is None
            else torch.cat((visual_prefix, text_embeddings), dim=1)
        )
        full_mask = attention_mask
        if attention_mask is not None and visual_prefix is not None:
            prefix_mask = torch.ones(
                input_ids.shape[0],
                visual_prefix.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            full_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        hidden, cache = self.language.prefill(
            embeddings,
            full_mask,
            max_cache_length=(
                embeddings.shape[1] + max_new_tokens - 1
            ),
        )
        return self.lm_head(hidden[:, -1]).float(), GenerationState(
            cache=cache,
            attention_mask=full_mask,
        )

    @torch.no_grad()
    def decode_generation(
        self,
        token_ids: torch.Tensor,
        state: GenerationState,
    ) -> tuple[torch.Tensor, GenerationState]:
        """Advance an incremental generation state by exactly one token."""

        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError(
                "cached generation token_ids must have shape [batch, 1]"
            )
        attention_mask = state.attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        token_ids.shape[0],
                        1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
        hidden, cache = self.language.decode(
            self.language.token_embedding(token_ids),
            state.cache,
            attention_mask,
        )
        return self.lm_head(hidden[:, -1]).float(), GenerationState(
            cache=cache,
            attention_mask=attention_mask,
        )

    @torch.no_grad()
    def _generate_with_confidence(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        packed_pixel_values: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_attention_backend: str = "auto",
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        use_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        generated = input_ids
        generated_attention_mask = attention_mask
        has_packed = any(
            value is not None
            for value in (
                packed_pixel_values,
                packed_position_ids,
                packed_cu_seqlens,
            )
        )
        if pixel_values is None and pixel_mask is not None and not has_packed:
            raise ValueError("pixel_mask requires pixel_values")
        visual_prefix = (
            self.encode_images(
                pixel_values,
                pixel_mask,
                packed_pixel_values=packed_pixel_values,
                packed_position_ids=packed_position_ids,
                packed_cu_seqlens=packed_cu_seqlens,
                packed_attention_backend=packed_attention_backend,
            )
            if pixel_values is not None or has_packed
            else None
        )
        log_probability_sum = torch.zeros(
            input_ids.shape[0],
            dtype=torch.float32,
            device=input_ids.device,
        )
        token_count = torch.zeros_like(log_probability_sum)
        active = torch.ones(
            input_ids.shape[0],
            dtype=torch.bool,
            device=input_ids.device,
        )
        next_logits = None
        generation_state = None
        if use_kv_cache:
            next_logits, generation_state = self.prefill_generation(
                generated,
                visual_prefix=visual_prefix,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )
        for step in range(max_new_tokens):
            if not use_kv_cache:
                output = self(
                    generated,
                    attention_mask=generated_attention_mask,
                    visual_prefix=visual_prefix,
                )
                next_logits = output.logits[:, -1].float()
            next_probability, next_token_flat = torch.softmax(
                next_logits,
                dim=-1,
            ).max(dim=-1)
            if eos_token_id is not None:
                next_token_flat = torch.where(
                    active,
                    next_token_flat,
                    torch.full_like(next_token_flat, eos_token_id),
                )
            next_token = next_token_flat.unsqueeze(-1)
            log_probability_sum += torch.where(
                active,
                next_probability.clamp_min(1e-12).log(),
                torch.zeros_like(next_probability),
            )
            token_count += active.float()
            generated = torch.cat((generated, next_token), dim=1)
            if generated_attention_mask is not None:
                generated_attention_mask = torch.cat(
                    (
                        generated_attention_mask,
                        torch.ones(
                            generated.shape[0],
                            1,
                            dtype=generated_attention_mask.dtype,
                            device=generated_attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
            if eos_token_id is not None:
                active &= next_token_flat != eos_token_id
                if not torch.any(active):
                    break
            if use_kv_cache and step + 1 < max_new_tokens:
                next_logits, generation_state = self.decode_generation(
                    next_token,
                    generation_state,
                )
        confidence = torch.exp(
            log_probability_sum / token_count.clamp_min(1.0)
        )
        return generated, confidence

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        packed_pixel_values: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_attention_backend: str = "auto",
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        use_kv_cache: bool = True,
    ) -> torch.Tensor:
        generated, _ = self._generate_with_confidence(
            input_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            packed_pixel_values=packed_pixel_values,
            packed_position_ids=packed_position_ids,
            packed_cu_seqlens=packed_cu_seqlens,
            packed_attention_backend=packed_attention_backend,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_kv_cache=use_kv_cache,
        )
        return generated

    @torch.no_grad()
    def generate_with_confidence(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        packed_pixel_values: torch.Tensor | None = None,
        packed_position_ids: torch.Tensor | None = None,
        packed_cu_seqlens: torch.Tensor | None = None,
        packed_attention_backend: str = "auto",
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        use_kv_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return greedy sequences and geometric-mean generated-token probabilities."""

        return self._generate_with_confidence(
            input_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            packed_pixel_values=packed_pixel_values,
            packed_position_ids=packed_position_ids,
            packed_cu_seqlens=packed_cu_seqlens,
            packed_attention_backend=packed_attention_backend,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            use_kv_cache=use_kv_cache,
        )

    def save_pretrained(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_payload = dict(metadata or {})
        if "parameter_attestation" in metadata_payload:
            raise ValueError(
                "parameter_attestation is reserved for runtime measurement"
            )
        metadata_payload["parameter_attestation"] = parameter_attestation(self)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "student_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / "model.pt")
        (output / "metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "DocumentVLMStudent":
        checkpoint = Path(checkpoint_dir)
        config = StudentConfig.from_dict(
            json.loads((checkpoint / "student_config.json").read_text(encoding="utf-8"))
        )
        device = torch.device(map_location)
        with torch.device(device):
            model = cls(config)
        state = torch.load(
            checkpoint / "model.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        metadata_path = checkpoint / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            recorded = metadata.get("parameter_attestation")
            if recorded is not None:
                validate_parameter_attestation(model, recorded)
        return model


def count_unique_parameters(model: nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for component in ("vision", "connector", "language"):
        counts[component] = sum(
            parameter.numel() for parameter in getattr(model, component).parameters()
        )
    language_parameter_ids = {
        id(parameter) for parameter in model.language.parameters()
    }
    counts["language"] += sum(
        parameter.numel()
        for parameter in model.lm_head.parameters()
        if id(parameter) not in language_parameter_ids
    )
    head_parameters = {
        id(parameter): parameter
        for module in (
            model.vision_projection,
            model.text_projection,
            model.orientation_head,
            model.box_head,
        )
        if module is not None
        for parameter in module.parameters()
    }
    for parameter in (
        model.contrastive_logit_scale,
        model.contrastive_logit_bias,
    ):
        if parameter is not None:
            head_parameters[id(parameter)] = parameter
    counts["task_heads"] = sum(parameter.numel() for parameter in head_parameters.values())
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def _parameter_architecture_fingerprint(model: nn.Module) -> str:
    records = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in model.named_parameters()
    ]
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def parameter_attestation(
    model: nn.Module,
    *,
    max_parameters_exclusive: int = 1_000_000_000,
) -> dict[str, Any]:
    """Measure and fail closed on the native student's deployment budget."""

    maximum = int(max_parameters_exclusive)
    if maximum <= 0:
        raise ValueError("max_parameters_exclusive must be positive")
    counts = count_unique_parameters(model)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    frozen = counts["total"] - trainable
    within_budget = 0 < counts["total"] < maximum
    if not within_budget:
        raise ValueError(
            f"runtime model size {counts['total']:,} is not below "
            f"{maximum:,}"
        )
    return {
        "schema_version": 1,
        "source": "runtime_numel",
        "architecture_fingerprint": _parameter_architecture_fingerprint(model),
        "parameter_counts": counts,
        "trainability": {
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_fraction": trainable / counts["total"],
        },
        "deployment": {
            "parameters_including_task_heads": counts["total"],
            "temporary_task_head_parameters": counts["task_heads"],
            "parameters_without_task_heads": (
                counts["total"] - counts["task_heads"]
            ),
        },
        "budget": {
            "max_parameters_exclusive": maximum,
            "within_budget": within_budget,
        },
    }


def validate_parameter_attestation(
    model: nn.Module,
    recorded: Any,
) -> dict[str, Any]:
    """Recompute immutable checkpoint topology and reject stale attestations."""

    if not isinstance(recorded, dict):
        raise ValueError("checkpoint parameter_attestation must be a mapping")
    budget = recorded.get("budget")
    if not isinstance(budget, dict):
        raise ValueError("checkpoint parameter_attestation budget is missing")
    observed = parameter_attestation(
        model,
        max_parameters_exclusive=int(
            budget.get("max_parameters_exclusive", 0)
        ),
    )
    for attestation_field in (
        "source",
        "architecture_fingerprint",
        "parameter_counts",
        "deployment",
        "budget",
    ):
        if recorded.get(attestation_field) != observed[attestation_field]:
            raise ValueError(
                "checkpoint parameter_attestation does not match runtime "
                f"model field {attestation_field}"
            )
    return observed

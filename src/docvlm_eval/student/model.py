"""A native vision-resampler-decoder model for the approximately 800M student."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConnectorConfig, LanguageConfig, StudentConfig, VisionConfig
from .losses import decode_normalized_box, generalized_box_iou_loss


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
        x = x + self.position_embedding[: x.shape[1]].unsqueeze(0).to(x.dtype)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = block(x, patch_mask)
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


class GatedResampler(nn.Module):
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.latents = nn.Parameter(torch.empty(config.latent_tokens, config.output_width))
        self.layers = nn.ModuleList([ResamplerLayer(config) for _ in range(config.layers)])

    def forward(
        self,
        vision_tokens: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = self.latents.unsqueeze(0).expand(vision_tokens.shape[0], -1, -1)
        for layer in self.layers:
            latents = layer(latents, vision_tokens, vision_mask)
        return latents


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
    group_ids: torch.Tensor,
) -> torch.Tensor:
    positive = group_ids[:, None] == group_ids[None, :]
    minimum = torch.finfo(similarities.dtype).min
    positive_score = torch.logsumexp(
        similarities.masked_fill(~positive, minimum),
        dim=-1,
    )
    all_score = torch.logsumexp(similarities, dim=-1)
    return (all_score - positive_score).mean()


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
    ) -> torch.Tensor:
        batch, length, width = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        repeats = self.heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return self.o_proj(out.transpose(1, 2).contiguous().view(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.width)
        self.attn = GroupedQueryAttention(config)
        self.norm2 = RMSNorm(config.width)
        self.mlp = SwiGLU(config.width, config.mlp_width)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.Tensor | None,
        is_causal: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_bias, is_causal, cos, sin)
        return x + self.mlp(self.norm2(x))


class LanguageDecoder(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.norm = RMSNorm(config.width)
        self.rotary = RotaryEmbedding(
            config.width // config.attention_heads,
            config.rope_base,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        capture_layers: set[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        x = embeddings
        attention_bias = None
        is_causal = attention_mask is None
        if attention_mask is not None:
            length = x.shape[1]
            minimum = torch.finfo(x.dtype).min
            causal = torch.full(
                (length, length),
                minimum,
                dtype=x.dtype,
                device=x.device,
            ).triu(1)
            padding = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * minimum
            attention_bias = causal[None, None] + padding
        cos, sin = self.rotary(x.shape[1], x.device, x.dtype)
        features: dict[int, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            x = block(x, attention_bias, is_causal, cos, sin)
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


class DocumentVLMStudent(nn.Module):
    """End-to-end image-prefix causal LM with removable auxiliary pretraining heads."""

    def __init__(self, config: StudentConfig):
        super().__init__()
        errors = config.validate()
        if errors:
            raise ValueError("invalid student config: " + "; ".join(errors))
        self.config = config
        self.vision = VisionTower(config.vision)
        self.connector = GatedResampler(config.connector)
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
        self.box_head = (
            nn.Linear(config.language.width, 4) if heads.box_regression else None
        )
        self.contrastive_temperature = float(heads.contrastive_temperature)
        self.apply(self._init_weights)
        nn.init.normal_(self.connector.latents, std=0.02)
        nn.init.zeros_(self.vision.position_embedding)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
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

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        box_targets: torch.Tensor | None = None,
        box_target_mask: torch.Tensor | None = None,
        box_query_positions: torch.Tensor | None = None,
        orientation_labels: torch.Tensor | None = None,
        contrastive: bool = False,
        contrastive_ids: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
        feature_layers: dict[str, list[int] | tuple[int, ...]] | None = None,
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
        if pixel_values is not None:
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
        elif pixel_mask is not None:
            raise ValueError("pixel_mask requires pixel_values")
        prefix = (
            self.connector(vision_tokens, vision_mask)
            if vision_tokens is not None
            else None
        )
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
                    + generalized_box_iou_loss(
                        box_predictions[valid_boxes],
                        box_targets[valid_boxes],
                    )
                )

        orientation_logits = None
        vision_embeddings = None
        text_projected = None
        if vision_tokens is not None:
            if vision_mask is None:
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
                similarities = (
                    vision_embeddings @ text_projected.T
                ) / self.contrastive_temperature
                if contrastive_ids is None:
                    targets = torch.arange(
                        similarities.shape[0],
                        device=similarities.device,
                    )
                    losses["region_text_contrastive"] = 0.5 * (
                        F.cross_entropy(similarities, targets)
                        + F.cross_entropy(similarities.T, targets)
                    )
                else:
                    group_ids = contrastive_ids.to(device=similarities.device)
                    if group_ids.shape != (similarities.shape[0],):
                        raise ValueError("contrastive_ids must have shape [batch]")
                    losses["region_text_contrastive"] = 0.5 * (
                        _multi_positive_contrastive_loss(similarities, group_ids)
                        + _multi_positive_contrastive_loss(
                            similarities.T,
                            group_ids,
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
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        generated = input_ids
        for _ in range(max_new_tokens):
            output = self(
                generated,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
        return generated

    def save_pretrained(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "student_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output / "model.pt")
        if metadata is not None:
            (output / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
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
    counts["task_heads"] = sum(parameter.numel() for parameter in head_parameters.values())
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    return counts

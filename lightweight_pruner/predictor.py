"""TCLM-RankSwiGLU predictor for learnable visual-token pruning.

This module keeps the public class name, ``LearnablePrunePredictor``, and the
same forward interface as the training wrapper expects, but reduces the
architecture to the parts needed by the training target: prompt-conditioned
TopK seed prediction.

Design:
    1. Shared down projection from LVLM hidden size to predictor width.
    2. Parameter-free sinusoidal position encodings: true normalized 2D
       coordinates for visual tokens and 1D positions for prompt text tokens.
    3. One multi-head low-rank visual-to-text token alignment.
    4. Low-rank multiplicative visual-text matching.
    5. A tiny SwiGLU FFN in rank space, followed by one scalar score head.

No visual self-attention, no final visual full attention, no visual-only scoring
branch, and no diversity modeling are used here.  Visual diversity/coverage is
assumed to be handled downstream by SCOPE.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


_POSITION_EMBEDDING_CACHE: dict[tuple, torch.Tensor] = {}


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight.to(dtype=x.dtype) * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def _sinusoidal_position_embedding(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1, seq_len, dim] sinusoidal PE without trainable parameters."""
    if seq_len <= 0:
        return torch.empty((1, 0, dim), device=device, dtype=dtype)

    cache_key = ("1d", int(seq_len), int(dim), device.type, device.index, dtype)
    cached = _POSITION_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Build in fp32 for numerical stability, then cast back.
    position = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    half_dim = (dim + 1) // 2
    div_term = torch.exp(
        torch.arange(half_dim, device=device, dtype=torch.float32)
        * (-(math.log(10000.0) / max(1, half_dim - 1)))
    )
    pe = torch.zeros((seq_len, dim), device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    pe = pe.unsqueeze(0).to(dtype=dtype)
    _POSITION_EMBEDDING_CACHE[cache_key] = pe
    return pe


def _sinusoidal_2d_position_embedding(
    seq_len: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1, seq_len, dim] 2D sin-cos PE for square visual grids."""
    side = int(seq_len**0.5)
    if seq_len <= 0:
        return torch.empty((1, 0, dim), device=device, dtype=dtype)
    cache_key = ("2d", int(seq_len), int(dim), device.type, device.index, dtype)
    cached = _POSITION_EMBEDDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if side * side != seq_len:
        return _sinusoidal_position_embedding(seq_len, dim, device=device, dtype=dtype)

    row_dim = dim // 2
    col_dim = dim - row_dim
    row_pe = _sinusoidal_position_embedding(side, row_dim, device=device, dtype=dtype)
    col_pe = _sinusoidal_position_embedding(side, col_dim, device=device, dtype=dtype)
    rows = row_pe[:, :, None, :].expand(1, side, side, row_dim)
    cols = col_pe[:, None, :, :].expand(1, side, side, col_dim)
    pe = torch.cat([rows, cols], dim=-1).reshape(1, seq_len, dim)
    _POSITION_EMBEDDING_CACHE[cache_key] = pe
    return pe


def _coordinate_2d_features(coordinates: torch.Tensor) -> torch.Tensor:
    """Cheap polynomial basis for normalized ``[..., x, y]`` coordinates."""
    if coordinates.shape[-1] != 2:
        raise ValueError(f"visual_coordinates must end in size 2, got {tuple(coordinates.shape)}")
    x, y = coordinates.unbind(dim=-1)
    x2, y2 = x.square(), y.square()
    return torch.stack((x, y, x2, y2, x * y, x2 * y, x * y2, torch.ones_like(x)), dim=-1)


class RankSwiGLU(nn.Module):
    """Small gated FFN applied in low-rank matching space."""

    def __init__(self, rank: int, mlp_ratio: int = 2):
        super().__init__()
        inner = int(rank) * int(mlp_ratio)
        self.gate_proj = nn.Linear(rank, inner, bias=False)
        self.up_proj = nn.Linear(rank, inner, bias=False)
        self.down_proj = nn.Linear(inner, rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LearnablePrunePredictor(nn.Module):
    """Text-conditioned low-rank matcher for visual-token TopK scoring.

    Args:
        input_size: LVLM hidden size of input embeddings, e.g. 4096 for LLaVA-7B.
        hidden_size: Predictor width after down projection.
        rank: Low-rank interaction width for alignment and matching.
        num_heads: Number of fused cross-attention heads. ``rank`` is the
            total interaction width across all heads.
        rank_mlp_ratio: Expansion ratio of the Rank-SwiGLU FFN.
        use_visual_position: Encode supplied true visual-grid coordinates;
            retain a legacy sequence-derived fallback when unavailable.
        use_text_position: Add fixed sinusoidal 1D position encoding to text tokens.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        rank: int = 256,
        num_heads: int = 4,
        rank_mlp_ratio: int = 2,
        use_visual_position: bool = True,
        use_text_position: bool = True,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.rank_mlp_ratio = int(rank_mlp_ratio)
        self.use_visual_position = bool(use_visual_position)
        self.use_text_position = bool(use_text_position)

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.rank % self.num_heads != 0:
            raise ValueError("rank must be divisible by num_heads")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.rank_mlp_ratio <= 0:
            raise ValueError("rank_mlp_ratio must be positive")

        self.in_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.visual_position_proj = nn.Linear(8, self.hidden_size, bias=False)

        # Normalization for each stream / interaction space.
        self.v_norm = RMSNorm(self.hidden_size)
        self.t_norm = RMSNorm(self.hidden_size)
        self.c_norm = RMSNorm(self.rank)
        self.m_norm = RMSNorm(self.rank)

        # Multi-head low-rank visual-to-text alignment.
        self.q_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.text_value_proj = nn.Linear(self.hidden_size, self.rank, bias=False)

        # Low-rank multiplicative matching.
        self.v_match = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.c_match = nn.Linear(self.rank, self.rank, bias=False)

        # Nonlinear TopK decision boundary in matching space.
        self.rank_ffn = RankSwiGLU(self.rank, mlp_ratio=self.rank_mlp_ratio)
        self.score = nn.Linear(self.rank, 1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.visual_position_proj.weight, mean=0.0, std=0.02)
        for module in (self.q_proj, self.k_proj, self.text_value_proj, self.v_match, self.c_match):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
        for module in (self.rank_ffn.gate_proj, self.rank_ffn.up_proj, self.rank_ffn.down_proj):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
        nn.init.normal_(self.score.weight, mean=0.0, std=1.0 / math.sqrt(self.score.in_features))
        nn.init.zeros_(self.score.bias)

    def _gather_token_states(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        auxiliary_states: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pack tokens and optional small side features with one index pass."""
        bsz, _, hidden_size = hidden_states.shape
        token_mask = token_mask.bool()
        token_lengths = token_mask.long().sum(dim=1)
        max_tokens = int(token_lengths.max().item()) if token_lengths.numel() > 0 else 0
        token_states = hidden_states.new_zeros((bsz, max_tokens, hidden_size))
        packed_auxiliary = None
        if auxiliary_states is not None:
            if auxiliary_states.shape[:2] != hidden_states.shape[:2]:
                raise ValueError("auxiliary_states must share [B, S] with hidden_states")
            packed_auxiliary = auxiliary_states.new_zeros(
                (bsz, max_tokens, *auxiliary_states.shape[2:])
            )
        valid_mask = torch.zeros((bsz, max_tokens), device=hidden_states.device, dtype=torch.bool)
        if max_tokens > 0:
            batch_idx, seq_idx = token_mask.nonzero(as_tuple=True)
            slot_idx = token_mask.long().cumsum(dim=1)[batch_idx, seq_idx] - 1
            token_states[batch_idx, slot_idx] = hidden_states[batch_idx, seq_idx]
            if packed_auxiliary is not None:
                packed_auxiliary[batch_idx, slot_idx] = auxiliary_states[batch_idx, seq_idx]
            valid_mask[batch_idx, slot_idx] = True
        return token_states, valid_mask, packed_auxiliary

    def _maybe_add_text_position(self, text: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
        if not self.use_text_position or text.shape[1] <= 0:
            return text
        pe = _sinusoidal_position_embedding(
            text.shape[1],
            text.shape[-1],
            device=text.device,
            dtype=text.dtype,
        )
        text = text + pe
        # Keep padded slots exactly zero; this makes empty-row dummy tokens safe.
        return text * text_valid.to(dtype=text.dtype).unsqueeze(-1)

    def _maybe_add_visual_position(
        self,
        visual: torch.Tensor,
        visual_valid: torch.Tensor,
        visual_coordinates: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.use_visual_position or visual.shape[1] <= 0:
            return visual
        if visual_coordinates is None:
            pe = _sinusoidal_2d_position_embedding(
                visual.shape[1], visual.shape[-1], device=visual.device, dtype=visual.dtype
            )
        else:
            if visual_coordinates.shape != (*visual.shape[:2], 2):
                raise ValueError(
                    "visual_coordinates must have shape [B, Nv, 2], got "
                    f"{tuple(visual_coordinates.shape)} for visual shape {tuple(visual.shape)}"
                )
            pe = self.visual_position_proj(
                _coordinate_2d_features(
                    visual_coordinates.to(device=visual.device, dtype=visual.dtype)
                )
            )
        visual = visual + pe
        return visual * visual_valid.to(dtype=visual.dtype).unsqueeze(-1)

    def _score_visual_text(
        self,
        visual: torch.Tensor,
        visual_valid: torch.Tensor,
        text: torch.Tensor,
        text_valid: torch.Tensor,
        visual_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual.shape[1] == 0:
            return visual.new_zeros((visual.shape[0], 0))

        # ``nn.MultiheadAttention``-style fully masked rows are avoided by
        # inserting a zero dummy text token for samples without prompt text.
        if text.shape[1] == 0:
            text = visual.new_zeros((visual.shape[0], 1, visual.shape[-1]))
            text_valid = torch.ones((visual.shape[0], 1), device=visual.device, dtype=torch.bool)
        else:
            # Mark empty packed rows with an all-zero dummy key without a
            # host-synchronizing branch or an extra clone.
            empty_text = ~text_valid.any(dim=1)
            text_valid[:, 0] |= empty_text

        visual = self._maybe_add_visual_position(visual, visual_valid, visual_coordinates)
        text = self._maybe_add_text_position(text, text_valid)

        visual_norm = self.v_norm(visual)
        q = self.q_proj(visual_norm)
        text_norm = self.t_norm(text)
        k = self.k_proj(text_norm)
        value = self.text_value_proj(text_norm)

        batch_size, visual_len, _ = q.shape
        text_len = k.shape[1]
        head_rank = self.rank // self.num_heads
        q = q.view(batch_size, visual_len, self.num_heads, head_rank).transpose(1, 2)
        k = k.view(batch_size, text_len, self.num_heads, head_rank).transpose(1, 2)
        value = value.view(batch_size, text_len, self.num_heads, head_rank).transpose(1, 2)
        # Fused SDPA avoids materializing B x heads x Nv x Nt attention.
        ctx = F.scaled_dot_product_attention(
            q,
            k,
            value,
            attn_mask=text_valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2).reshape(batch_size, visual_len, self.rank)

        v_match = F.gelu(self.v_match(visual_norm))
        c_match = F.gelu(self.c_match(self.c_norm(ctx)))
        match = v_match * c_match
        match = match + self.rank_ffn(self.m_norm(match))
        logits = self.score(match).squeeze(-1)
        return logits.masked_fill(~visual_valid, 0.0)

    def forward(
        self,
        multimodal_hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        visual_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = multimodal_hidden_states.new_ones(multimodal_hidden_states.shape[:2], dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if visual_token_mask is None:
            visual_token_mask = attention_mask
        else:
            visual_token_mask = visual_token_mask.bool() & attention_mask
        if text_token_mask is None:
            text_token_mask = attention_mask & ~visual_token_mask
        else:
            text_token_mask = text_token_mask.bool() & attention_mask & ~visual_token_mask

        if visual_coordinates is not None:
            if visual_coordinates.shape[:2] != multimodal_hidden_states.shape[:2] or visual_coordinates.shape[-1] != 2:
                raise ValueError("visual_coordinates must have shape [B, S, 2]")
        visual, visual_valid, packed_coordinates = self._gather_token_states(
            multimodal_hidden_states, visual_token_mask, visual_coordinates
        )
        text, text_valid, _ = self._gather_token_states(multimodal_hidden_states, text_token_mask)
        visual = self.in_proj(visual)
        text = self.in_proj(text)
        return self._score_visual_text(visual, visual_valid, text, text_valid, packed_coordinates)

    def score_visual_text_by_positions(
        self,
        multimodal_hidden_states: torch.Tensor,
        visual_positions: torch.Tensor,
        text_positions: torch.Tensor,
        visual_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference helper used by the LLaVA pruning wrapper.

        Args:
            multimodal_hidden_states: [B, S, input_size].
            visual_positions: 1D positions of visual placeholder/image tokens.
            text_positions: 1D positions of prompt text tokens.
        Returns:
            [B, Nv] logits.
        """
        visual_positions = visual_positions.to(device=multimodal_hidden_states.device, dtype=torch.long)
        text_positions = text_positions.to(device=multimodal_hidden_states.device, dtype=torch.long)

        visual = self.in_proj(multimodal_hidden_states.index_select(1, visual_positions))
        visual_valid = torch.ones(visual.shape[:2], device=visual.device, dtype=torch.bool)
        if text_positions.numel() == 0:
            text = visual.new_zeros((visual.shape[0], 1, visual.shape[-1]))
            text_valid = torch.ones(text.shape[:2], device=visual.device, dtype=torch.bool)
        else:
            text = self.in_proj(multimodal_hidden_states.index_select(1, text_positions))
            text_valid = torch.ones(text.shape[:2], device=visual.device, dtype=torch.bool)
        return self._score_visual_text(visual, visual_valid, text, text_valid, visual_coordinates)

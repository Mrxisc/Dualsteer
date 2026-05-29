from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn


class _FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class _AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)


    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        query = self.norm(x)
        key_value = query if context is None else context
        attn_out, _ = self.attn(query, key_value, key_value, need_weights=False)
        return residual + attn_out


class SD35Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.self_attn = _AttentionBlock(dim, num_heads, dropout)
        self.cross_attn = _AttentionBlock(dim, num_heads, dropout)
        self.ff = _FeedForward(dim, mlp_ratio, dropout)


    def forward(self, latents: torch.Tensor, cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        latents = self.self_attn(latents)
        if cond_tokens is not None:
            latents = self.cross_attn(latents, cond_tokens)
        latents = self.ff(latents)
        return latents


class SD35Backbone(nn.Module):
    def __init__(
        self,
        *,
        sample_size: int = 128,
        in_channels: int = 16,
        latent_dim: int = 4096,
        cond_dim: int = 4096,
        out_dim: Optional[int] = None,
        num_blocks: int = 38,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim or in_channels
        self.token_in = nn.Linear(in_channels, latent_dim)
        self.token_out = nn.Linear(latent_dim, self.out_dim)
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.transformer_blocks = nn.ModuleList(
            [SD35Block(latent_dim, num_heads, dropout, mlp_ratio) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(latent_dim)


    @property
    def dtype(self) -> torch.dtype:
        first_param = next(self.parameters(), None)
        return first_param.dtype if first_param is not None else torch.get_default_dtype()


    def forward(
        self,
        latent_tokens: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        *,
        hook_layers: Optional[Iterable[int]] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        features: Dict[str, torch.Tensor] = {}
        hook_set = set(hook_layers or [])
        hidden = self.token_in(latent_tokens)
        hidden = hidden + self.pos_embed
        hidden = self.dropout(hidden)
        cond_proj: Optional[torch.Tensor] = None
        if cond_tokens is not None:
            if cond_tokens.ndim == 2:
                cond_tokens = cond_tokens.unsqueeze(1)
            cond_proj = self.cond_proj(cond_tokens)
            if cond_proj.shape[1] == 1 and hidden.shape[1] > 1:
                cond_proj = cond_proj.expand(-1, hidden.shape[1], -1)

        for idx, block in enumerate(self.transformer_blocks):
            hidden = block(hidden, cond_proj)
            if idx in hook_set:
                features[f"transformer_blocks.{idx}"] = hidden.detach()


        hidden = self.norm(hidden)
        hidden = self.token_out(hidden)

        if return_features:
            return hidden, features
        return hidden


    def enable_xformers_memory_efficient_attention(self):
        return self


__all__ = ["SD35Backbone", "SD35Block"]

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


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
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )


    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if context is None:
            attn_out, _ = self.attn(x, x, x, need_weights=False)
        else:
            attn_out, _ = self.attn(x, context, context, need_weights=False)
        return residual + attn_out


class FluxDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.self_attn = _AttentionBlock(dim, num_heads)
        self.cross_attn = _AttentionBlock(dim, num_heads)
        self.ff = _FeedForward(dim, mlp_ratio)
        self.cond_proj = nn.Linear(cond_dim, dim)


    def forward(
        self,
        latents: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latents = self.self_attn(latents)
        if cond_tokens is not None:
            cond_tokens = self.cond_proj(cond_tokens)
            latents = self.cross_attn(latents, cond_tokens)
        latents = self.ff(latents)
        return latents


class FluxSingleStreamBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = _AttentionBlock(dim, num_heads)
        self.ff = _FeedForward(dim, mlp_ratio)


    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.attn(tokens)
        tokens = self.ff(tokens)
        return tokens


class FluxBackbone(nn.Module):
    def __init__(
        self,
        *,
        sample_size: int = 128,
        in_channels: int = 16,
        latent_dim: int = 4096,
        cond_dim: int = 4096,
        out_dim: Optional[int] = None,
        double_blocks: int = 6,
        single_blocks: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim or in_channels


        self.token_in = nn.Linear(in_channels, latent_dim)
        self.token_out = nn.Linear(latent_dim, self.out_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))


        self.double_blocks = nn.ModuleList(
            [
                FluxDoubleStreamBlock(latent_dim, cond_dim, num_heads, mlp_ratio)
                for _ in range(double_blocks)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [FluxSingleStreamBlock(latent_dim, num_heads, mlp_ratio)
             for _ in range(single_blocks)]
        )
        self.norm = nn.LayerNorm(latent_dim)


    def forward(
        self,
        latent_tokens: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        *,
        hook_layers: Optional[Dict[str, List[int]]] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features: Dict[str, torch.Tensor] = {}
        hook_double = set((hook_layers or {}).get("double", []))
        hook_single = set((hook_layers or {}).get("single", []))


        latent_tokens = self.token_in(latent_tokens)
        latent_tokens = latent_tokens + self.pos_embed
        latent_tokens = self.dropout(latent_tokens)


        for idx, block in enumerate(self.double_blocks):
            latent_tokens = block(latent_tokens, cond_tokens)
            if idx in hook_double:
                features[f"double_blocks.{idx}"] = latent_tokens.detach()


        for idx, block in enumerate(self.single_blocks):
            latent_tokens = block(latent_tokens)
            if idx in hook_single:
                features[f"single_blocks.{idx}"] = latent_tokens.detach()


        latent_tokens = self.norm(latent_tokens)
        latent_tokens = self.token_out(latent_tokens)


        if return_features:
            return latent_tokens, features
        return latent_tokens


    def enable_xformers_memory_efficient_attention(self):

        return self


__all__ = [
    "FluxBackbone",
    "FluxDoubleStreamBlock",
    "FluxSingleStreamBlock",
]

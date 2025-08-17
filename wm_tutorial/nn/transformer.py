"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Folco Bertini Baldassini
Source: https://gitlab.com/folbaeni/linguistic-watermark
----------------------------------------------------------
Rotary multi-head attention mechanism a transformer block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

from .pos_enc import (
    build_sinusoidal_cache,
    apply_sinusoidal,
    build_rope_cache, 
    apply_rope,
)

class MultiheadAttention(nn.Module):
    def __init__(
        self, 
        model_dim: int,
        num_heads: int,
        bias: bool,
        dropout: float,
        max_len: int,
        pos_enc: str = "rope",
    ):
        super().__init__()

        assert pos_enc in ["rope", "absolute", "none", None]

        head_dim = model_dim // num_heads

        assert (
            head_dim * num_heads == model_dim
        ), f"`model_dim` must be divisible by `num_heads`"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(model_dim, model_dim, bias)
        self.k_proj = nn.Linear(model_dim, model_dim, bias)
        self.v_proj = nn.Linear(model_dim, model_dim, bias)
        
        self.out_proj = nn.Linear(model_dim, model_dim, bias)
        self.dropout = dropout

        self.pos_enc = pos_enc
        
        self.max_len = max_len
        self.pos_cache = None
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: torch.Tensor = None, 
    ):
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        q = (
            self.q_proj(query)
            .view(batch_size, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(batch_size, src_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Cache positional encodings for future forward passes
        if self.pos_cache is None and self.pos_enc is not None:

            if self.pos_enc == "rope":        
                self.pos_cache = build_rope_cache(
                    seq_len=self.max_len,
                    n_elem=self.model_dim // self.num_heads,
                    dtype=q.dtype,
                    device=q.device,
                )
            elif self.pos_enc == "absolute":
                self.pos_cache = build_sinusoidal_cache(
                    seq_len=self.max_len,
                    n_elem=self.model_dim // self.num_heads,
                    dtype=q.dtype,
                    device=q.device,
                )

        if self.pos_enc == "rope":
            q = apply_rope(q, self.pos_cache)
            k = apply_rope(k, self.pos_cache)
        elif self.pos_enc == "absolute":
            q = apply_sinusoidal(q, self.pos_cache)
            k = apply_sinusoidal(k, self.pos_cache)
        
        if attn_mask is not None:
            attn_mask = attn_mask.view(batch_size, 1, tgt_len, tgt_len).expand(
                -1, self.num_heads, -1, -1
            )
            attn_mask = attn_mask[..., :src_len]

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.model_dim)
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class Block(nn.Module):
    def __init__(
        self, 
        model_dim: int,
        num_heads: int,
        bias: bool,
        dropout: float,
        max_len: int,
        dim_feedforward: int,
        pos_enc: str = "rope",
    ):
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            max_len=max_len,
            pos_enc=pos_enc,
        )

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, dim_feedforward, bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, model_dim, bias),
        )

        # Intermediate layers
        self.norm_sa = nn.LayerNorm(model_dim)
        self.norm_ff = nn.LayerNorm(model_dim)
        self.dropout_sa = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def _sa_block(self, x: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        y = self.norm_sa(x)
        y = self.self_attn(y, y, y, mask, *args, **kwargs)
        x = x + self.dropout_sa(y)
        
        return x

    def _ff_block(self, x: torch.Tensor):
        y = self.norm_ff(x)
        y = self.linear_net(y)
        x = x + self.dropout_ff(y)
        
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, *args, **kwargs):
        return self._ff_block(self._sa_block(x, mask, *args, **kwargs))


class Transformer(nn.Module):
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        bias: bool,
        dropout: float,
        max_len: int,
        dim_feedforward: int,
        num_layers: int,
        pos_enc: str = "rope",
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                Block(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    bias=bias,
                    dropout=dropout,
                    max_len=max_len,
                    dim_feedforward=dim_feedforward,
                    pos_enc=pos_enc,
                ) for _ in range(num_layers)
            ]
        )

        self.model_dim = model_dim
        self.max_len = max_len
        self.num_layers = num_layers

    def forward(self, x: torch.tensor, mask: torch.Tensor, **kwargs):
        for layer in self.layers:
            x = layer(x, mask, **kwargs)
        return x
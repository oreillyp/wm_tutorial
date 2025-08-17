"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Folco Bertini Baldassini
Source: https://gitlab.com/folbaeni/linguistic-watermark
--------------------------------------------------------
Helpers for positional encoding.
"""

import math
import torch
import torch.nn as nn


def build_sinusoidal_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:

    index = torch.arange(seq_len).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, n_elem, 2) * (-math.log(10000.0) / n_elem))
    encoding = torch.zeros(seq_len, 1, n_elem)
    encoding[:, 0, 0::2] = torch.sin(index * frequency)
    encoding[:, 0, 1::2] = torch.cos(index * frequency)
    
    return encoding.to(dtype).to(device)


def apply_sinusoidal(
    x: torch.Tensor, 
    sinusoidal_cache: torch.Tensor
) -> torch.Tensor:

    assert x.ndim == 4  # (n_batch, n_heads, seq_len, head_dim)
    assert sinusoidal_cache.ndim == 3  # (seq_len, 1, head_dim)
    assert x.shape[-1] == sinusoidal_cache.shape[-1]
    assert x.shape[2] <= sinusoidal_cache.shape[0]
    
    x = x.permute(2, 0, 1, 3)  # (seq_len, n_batch, n_heads, head_dim)

    x_out = x + sinusoidal_cache[:x.shape[0]].unsqueeze(2)

    return x_out.permute(1, 2, 0, 3)
    

def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> torch.Tensor:
    """Enhanced Transformer with Rotary Position Embedding.
    Derived from:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py
    MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """  # noqa : E501
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    # Compute cache. Because polar only takes float32 or float64, we need to cast
    # when working with 16 bit floats (float16 or bfloat16)
    working_dtype = (
        torch.float32 if (dtype == torch.float16 or dtype == torch.bfloat16) else dtype
    )
    complex_dtype = (
        torch.complex32
        if (dtype == torch.float16 or dtype == torch.bfloat16)
        else torch.complex64
    )
    cache = torch.polar(
        torch.ones_like(idx_theta).to(working_dtype), idx_theta.to(working_dtype)
    ).to(complex_dtype)
    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 2)

    # Truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # Cast because `view_as_complex` does not support 16-bit tensors
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope_cache = rope_cache.view(1, xc.size(1), 1, xc.size(3))
    x_out = torch.view_as_real(xc * rope_cache).flatten(3)
    return x_out.transpose(1, 2).type_as(x)
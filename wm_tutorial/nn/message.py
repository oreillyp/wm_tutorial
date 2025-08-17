import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import numpy as np

from typing import Union
from audiotools.core.util import random_state

from .transformer import Block


def _get_orthogonal_matrix(n: int, seed: Union[int, np.random.RandomState]):
    """
    Generate an (n x n) orthogonal matrix and normalize each row to unit length
    """
    state = random_state(seed)

    mat = scipy.stats.ortho_group(dim=n, seed=state).rvs()
    mat = torch.from_numpy(mat)
    mat = mat / mat.norm(dim=-1, keepdim=True)
    return mat.float()


class MessageEmbedding(nn.Module):
    """
    Embed binary messages as continuous vectors; reserve one embedding for each
    possible value of each bit, resulting in an embedding table of size 
    (2 x n_bits).
    """

    def __init__(
        self, 
        n_bits: int, 
        model_dim: int, 
        init: str = None,  # "orthogonal", "binary"
        learnable: bool = True
    ):

        super().__init__()

        assert init in [None, "none", "orthogonal", "binary"], \
            f"Invalid embedding initialization `init`=={init}"
        
        self.emb = nn.Embedding(2 * n_bits, model_dim)

        # Initialize embeddings as random orthogonal vectors (preserve original
        # scale)
        if init == "orthogonal":

            orig_norm = self.emb.weight.norm(dim=-1, keepdim=True)  # (2 * n_bits, 1)
            
            mat = _get_orthogonal_matrix(
                max(self.emb.weight.shape), seed=torch.randint(0, int(1e6), (1,)).item()
            )[:2 * n_bits, :model_dim]  # (2 * n_bits, model_dim)

            self.emb.weight.data = mat * orig_norm

        # Initialize embeddings as one-bit binary masks (sum to exact binary 
        # message)
        elif init == "binary":

            mat = torch.zeros_like(self.emb.weight)
            mat[::2, :] = 0
            mat[1::2, :n_bits] = torch.eye(n_bits, dtype=mat.dtype, device=mat.device)

            self.emb.weight.data = mat
        
        self.n_bits = n_bits

        if not learnable:
            for p in self.emb.parameters():
                p.requires_grad_(False)
        else:
            for p in self.emb.parameters():
                p.requires_grad_(True)

    def forward(self, msg: torch.Tensor, mask: torch.Tensor = None):
        """
        Embed a given binary message as a continuous vector.

        Parameters
        ----------
        msg : torch.Tensor
            Binary message, shape (n_batch, n_bits)
        mask : torch.Tensor
            Optional boolean mask to remove information from certain bit 
            positions, shape (n_batch, n_bits)

        Returns
        -------
        torch.Tensor
            Continuous message embedding, shape (n_batch, model_dim)
        """
        
        assert msg.ndim == 2  # (n_batch, n_bits)
        assert msg.shape[-1] == self.n_bits
        n_batch, n_bits = msg.shape

        # Optionally, provide bit mask to zero embeddings for certain bits
        if mask is not None:
            assert mask.ndim == 2  # (n_batch, n_bits)
            assert mask.shape[-1] == n_bits      
        else:
            mask = torch.ones(n_batch, n_bits, dtype=msg.dtype, device=msg.device)

        # Compute embedding index (each bit has both a 0 and 1 entry in 
        # embedding table)
        idx = 2 * torch.arange(
            n_bits, device=msg.device
        ).unsqueeze(0)
        idx = (idx + msg).long()  # (n_batch, n_bits)

        # Compute embedding vector for each bit
        emb = self.emb(idx)  # (n_batch, n_bits, model_dim)

        # Apply mask
        emb = emb * mask.to(emb).unsqueeze(-1)

        # Sum embedding vector for each bit
        emb = emb.sum(dim=1)  # (n_batch, model_dim)

        return emb

    
class MessageBlock(Block):
    """
    Condition generation on continuous message embeddings using cross-attention,
    in which queries are drawn from input sequence while keys and values are 
    drawn from message embeddings (treated as length-1 sequences). Because each
    position in the input sequence attends to the single position in the message
    "sequence", there is no need for masking.
    """

    def __init__(
        self, 
        model_dim: int,
        num_heads: int,
        bias: bool,
        dropout: float,
        max_len: int,
        dim_feedforward: int,
        pos_enc: str,
        *args,
        **kwargs,
    ):
        super().__init__(
            model_dim,
            num_heads,
            bias,
            dropout,
            max_len,
            dim_feedforward,
            pos_enc,
        )

    def _ca_block(self, x: torch.Tensor, msg: torch.Tensor, *args, **kwargs):

        x_norm = self.norm_sa(x)
        msg_norm = self.norm_ff(msg)
        y = self.self_attn(x_norm, msg_norm, msg_norm, *args,  **kwargs)
        x = x + self.dropout_sa(y)
        
        return x

    def forward(self, x: torch.Tensor, msg: torch.Tensor, *args, **kwargs):

        assert x.ndim == 3  # (n_batch, n_seq_q, model_dim)
        assert msg.ndim == 3  # (n_batch, n_seq_kv, model_dim)

        x = self._ca_block(
            x,
            msg, 
            *args, 
            **kwargs
        )
            
        return x

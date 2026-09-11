"""A multi-head scaled dot-product self-attention module."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention over a ``[B, L, D]`` sequence.

    The width is split into ``num_heads`` heads of equal size, attention runs
    per head, and the heads are concatenated and projected back. Like
    :class:`~snn_interpreter.topology.attention.Attention` it is stateless
    across steps and has no faithful ``nir`` rendering, so export rejects it
    with a typed error rather than approximating it.
    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Store the head split and create the projection linears."""
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be a positive multiple of heads")
        self._heads = num_heads
        self._head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, L, D]`` into ``[B, heads, L, head_dim]``."""
        batch, length, _ = x.shape
        shaped = x.reshape(batch, length, self._heads, self._head_dim)
        return shaped.transpose(1, 2)

    def _attend(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """Return the per-head scaled dot-product attention output."""
        scale = self._head_dim ** 0.5
        weights = torch.softmax(
            query @ key.transpose(-2, -1) / scale, dim=-1
        )
        return weights @ value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the projected multi-head attention output for ``x``."""
        batch, length, width = x.shape
        query = self._split(self.q_proj(x))
        key = self._split(self.k_proj(x))
        value = self._split(self.v_proj(x))
        merged = self._attend(query, key, value)
        joined = merged.transpose(1, 2).reshape(batch, length, width)
        return self.out_proj(joined)

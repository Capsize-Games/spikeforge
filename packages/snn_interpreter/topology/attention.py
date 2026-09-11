"""A single-head scaled dot-product self-attention module."""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Single-head self-attention over a ``[B, L, D]`` sequence.

    The projection, softmax, and weighted-sum arithmetic are ordinary torch
    operations; the module is stateless across simulator time steps, which is
    all the generic temporal loop supports. There is no faithful ``nir``
    attention primitive, so this kind is export-unexportable by design.
    """

    def __init__(self, embed_dim: int) -> None:
        """Store the width and create the query/key/value projections."""
        super().__init__()
        self._embed_dim = int(embed_dim)
        self.q_proj = nn.Linear(self._embed_dim, self._embed_dim)
        self.k_proj = nn.Linear(self._embed_dim, self._embed_dim)
        self.v_proj = nn.Linear(self._embed_dim, self._embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the attention-weighted values for ``x``."""
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        scale = self._embed_dim ** 0.5
        weights = torch.softmax(
            query @ key.transpose(-2, -1) / scale, dim=-1
        )
        return weights @ value

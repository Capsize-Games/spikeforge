"""A deterministic sinusoidal positional-encoding module."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add a fixed sinusoidal position signal to a ``[B, L, D]`` sequence.

    The table is a non-persistent registration buffer, so it moves with the
    module's device but stays out of the learned parameter set. Because the
    encoding is a fixed constant with no ``nir`` constant node, this kind is
    export-unexportable and stays simulation-only.
    """

    def __init__(self, embed_dim: int, max_length: int = 128) -> None:
        """Build and register the ``[max_length, embed_dim]`` table."""
        super().__init__()
        self.register_buffer("pe", self._table(embed_dim, max_length))

    @staticmethod
    def _table(embed_dim: int, max_length: int) -> torch.Tensor:
        """Return the sinusoidal ``[max_length, embed_dim]`` table."""
        position = torch.arange(max_length).unsqueeze(1).float()
        rate = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-math.log(10000.0) / embed_dim)
        )
        table = torch.zeros(max_length, embed_dim)
        table[:, 0::2] = torch.sin(position * rate)
        table[:, 1::2] = torch.cos(position * rate)
        return table

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus its position signal up to ``x``'s length."""
        return x + self.pe[: x.size(1)]

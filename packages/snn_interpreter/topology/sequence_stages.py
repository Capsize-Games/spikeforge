"""Module factories for the sequence, embedding, and norm stage kinds.

Each factory is a small, pure mapping from a stage's ``params`` to a torch
module. Only ``embedding`` and the two normalisations need custom wiring;
``dropout`` and the pooling kinds live in
:mod:`snn_interpreter.topology.stage_modules` alongside the other built-ins.
"""

from typing import Any, Mapping

import torch.nn as nn

from snn_interpreter.topology.attention import Attention
from snn_interpreter.topology.multihead_attention import MultiHeadAttention
from snn_interpreter.topology.positional_encoding import PositionalEncoding


def embedding(params: Mapping[str, Any]) -> nn.Module:
    """Return an ``nn.Embedding`` token lookup table."""
    return nn.Embedding(
        int(params["num_embeddings"]), int(params["embedding_dim"])
    )


def layer_norm(params: Mapping[str, Any]) -> nn.Module:
    """Return an ``nn.LayerNorm`` over the normalised shape."""
    return nn.LayerNorm(
        int(params["normalized_shape"]),
        eps=float(params.get("eps", 1e-5)),
    )


def batch_norm(params: Mapping[str, Any]) -> nn.Module:
    """Return a channel-first ``nn.BatchNorm1d``/``nn.BatchNorm2d``.

    The kind follows PyTorch's channel-first convention (``[B, C, ...]``),
    which is why a sequence that mixes it with attention must be laid out
    channels-first. It has no ``nir`` primitive, so export rejects it.
    """
    cls = nn.BatchNorm2d if int(params.get("dim", 1)) == 2 else nn.BatchNorm1d
    return cls(
        int(params["num_features"]),
        eps=float(params.get("eps", 1e-5)),
        momentum=float(params.get("momentum", 0.1)),
    )


def positional_encoding(params: Mapping[str, Any]) -> nn.Module:
    """Return the deterministic sinusoidal positional-encoding module."""
    return PositionalEncoding(
        int(params["embed_dim"]), int(params.get("max_length", 128))
    )


def attention(params: Mapping[str, Any]) -> nn.Module:
    """Return a single-head self-attention module."""
    return Attention(int(params["embed_dim"]))


def multihead_attention(params: Mapping[str, Any]) -> nn.Module:
    """Return a multi-head self-attention module."""
    return MultiHeadAttention(
        int(params["embed_dim"]), int(params["num_heads"])
    )


#: Sequence/norm factories registered by the stage-module builder table.
BUILDERS = {
    "embedding": embedding,
    "layer_norm": layer_norm,
    "batch_norm": batch_norm,
    "positional_encoding": positional_encoding,
    "attention": attention,
    "multihead_attention": multihead_attention,
}

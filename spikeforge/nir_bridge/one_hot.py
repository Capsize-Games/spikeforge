"""Expand token ids to the one-hot frames embedding graphs consume.

An exported ``embedding`` stage is an ``Affine`` node over one-hot float
input: NIR has no integer tensor or gather primitive, so the ``long``
``[B, L]`` token ids a simulation feeds the stage cannot enter the graph
directly. :func:`one_hot` is that ingest map, and its return layout is the
graph's input contract: producers one-hot their ids before running an
embedding-first graph, and the graph consumes ``float32`` one-hot
``[B, L, num_embeddings]`` frames.
"""

import torch
import torch.nn.functional as functional


def one_hot(ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
    """Return ``ids`` as float32 one-hot ``[B, L, num_embeddings]``.

    ``ids`` must be a ``long`` ``[B, L]`` tensor with every entry inside
    ``[0, num_embeddings)``. Anything else raises :class:`ValueError`
    naming the breach, so a bad frame never silently becomes a wrong row.
    """
    count = int(num_embeddings)
    if count <= 0:
        raise ValueError(f"num_embeddings must be positive, got {count}")
    if ids.dtype != torch.long:
        raise ValueError(f"token ids must be long [B, L], not {ids.dtype}")
    if ids.dim() != 2:
        raise ValueError(
            f"token ids must be long [B, L], not {ids.dim()}D"
        )
    if bool((ids < 0).any()) or bool((ids >= count).any()):
        raise ValueError("token ids must lie in [0, num_embeddings)")
    return functional.one_hot(ids, count).to(torch.float32)

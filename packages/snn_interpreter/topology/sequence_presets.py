"""Sequence demonstration presets: ``sequence_mlp`` and ``sequence_attn``.

``sequence_mlp`` is built only from NIR-mappable kinds (``linear``, ``leaky``)
applied to a ``[T, B, L, D]`` token sequence, so it exports and validates end
to end. Its neurons default to ``reset="zero"``, which renders a single
``nir.LIF`` per stage instead of the subtract-reset ``LI``/``Threshold``/
``Delay``/``Scale`` expansion, so the exported graph contains no ``Delay``
node and is runnable by the Norse simulator target.

``sequence_attn`` is a spiking-transformer-shaped stack
(``embedding`` -> ``positional_encoding`` -> ``multihead_attention`` ->
``layer_norm`` -> ``linear`` -> neuron). The installed ``nir`` has no
embedding, attention, or normalisation primitive, so export raises the typed
``UnsupportedStageError`` naming the first unexportable stage; it stays
available for simulation and introspection.
"""

from typing import List, Optional

from snn_interpreter.topology.presets import (
    DEFAULT_NEURON,
    NeuronMap,
    NeuronStage,
    ParamsMap,
    _linear_params,
    _stage_factory,
)
from snn_interpreter.topology.spec import TopologySpec, chain
from snn_interpreter.topology.stage import Stage


def _merged_params(
    stage_params: ParamsMap,
    names: tuple,
    threshold: Optional[float],
    reset: Optional[str],
) -> ParamsMap:
    """Overlay the preset's global threshold/reset under per-stage params."""
    base = {}
    if threshold is not None:
        base["threshold"] = threshold
    if reset is not None:
        base["reset"] = reset
    return {
        name: {**base, **(stage_params or {}).get(name, {})}
        for name in names
    }


def sequence_mlp(
    seq_length: int = 8,
    features: int = 8,
    hidden: int = 16,
    beta: float = 0.9,
    num_classes: int = 4,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
    threshold: Optional[float] = None,
    reset: Optional[str] = "zero",
) -> TopologySpec:
    """NIR-mappable per-token MLP over a ``[T, B, L, seq_length]`` sequence.

    Each step consumes ``[B, L, features]``; the linear stages map the
    feature axis so the readout keeps the ``L`` token axis. ``seq_length``
    documents the demonstration task's length and does not enter the graph.
    """
    names = ("lif1", "lif2")
    overrides = _merged_params(stage_params, names, threshold, reset)
    stage = _stage_factory(neurons, overrides)
    return chain(
        [
            Stage("fc1", "linear", _linear_params(features, hidden)),
            stage("lif1", neuron, beta, surrogate),
            Stage("fc2", "linear", _linear_params(hidden, num_classes)),
            stage("lif2", neuron, beta, surrogate),
        ]
    )


def _attn_stages(
    seq_length: int,
    vocab: int,
    embed_dim: int,
    num_heads: int,
    num_classes: int,
    neuron: str,
    beta: float,
    surrogate: Optional[str],
    stage: NeuronStage,
) -> List[Stage]:
    """Return the transformer-shaped stage list for ``sequence_attn``."""
    return [
        Stage(
            "embed",
            "embedding",
            {"num_embeddings": vocab, "embedding_dim": embed_dim},
        ),
        Stage(
            "pos",
            "positional_encoding",
            {"embed_dim": embed_dim, "max_length": seq_length},
        ),
        Stage(
            "attn",
            "multihead_attention",
            {"embed_dim": embed_dim, "num_heads": num_heads},
        ),
        Stage("norm", "layer_norm", {"normalized_shape": embed_dim}),
        Stage("fc", "linear", _linear_params(embed_dim, num_classes)),
        stage("out", neuron, beta, surrogate),
    ]


def sequence_attn(
    seq_length: int = 8,
    vocab: int = 32,
    embed_dim: int = 16,
    num_heads: int = 2,
    beta: float = 0.9,
    num_classes: int = 4,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
    threshold: Optional[float] = None,
    reset: Optional[str] = "subtract",
) -> TopologySpec:
    """Simulation-only spiking-transformer-shaped preset (unexportable).

    The stack mirrors a transformer encoder block; because ``nir`` lacks
    embedding, attention, and normalisation primitives, only its simulation
    and introspection behaviour is faithful. Export fails loudly and names
    the first unexportable stage.
    """
    overrides = _merged_params(stage_params, ("out",), threshold, reset)
    stage = _stage_factory(neurons, overrides)
    stages = _attn_stages(
        seq_length, vocab, embed_dim, num_heads, num_classes,
        neuron, beta, surrogate, stage,
    )
    return chain(stages)

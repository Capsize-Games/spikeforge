"""Stage kind names recognised by the topology package."""

from typing import Tuple

from spikeforge.neurons.registry import neuron_kinds

#: Module kinds rendered by a torch module factory.
MODULE_KINDS: Tuple[str, ...] = (
    "flatten",
    "linear",
    "conv2d",
    "conv1d",
    "avgpool2d",
    "sumpool2d",
    "maxpool1d",
    "maxpool2d",
    "embedding",
    "layer_norm",
    "batch_norm",
    "dropout",
    "positional_encoding",
    "attention",
    "multihead_attention",
)
#: Kinds that carry no parameters or submodule of their own.
PARAMETERLESS_KINDS: Tuple[str, ...] = ("add",)
#: Every kind a :class:`~spikeforge.topology.stage.Stage` may use.
STAGE_KINDS: Tuple[str, ...] = (
    *MODULE_KINDS,
    *PARAMETERLESS_KINDS,
    *neuron_kinds(),
)

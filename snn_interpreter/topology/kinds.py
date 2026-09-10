"""Stage kind names recognised by the topology package."""

from typing import Tuple

from snn_interpreter.neurons.registry import neuron_kinds

#: Module kinds rendered by a torch module factory.
MODULE_KINDS: Tuple[str, ...] = (
    "flatten",
    "linear",
    "conv2d",
    "avgpool2d",
    "sumpool2d",
)
#: Kinds that carry no parameters or submodule of their own.
PARAMETERLESS_KINDS: Tuple[str, ...] = ("add",)
#: Every kind a :class:`~snn_interpreter.topology.stage.Stage` may use.
STAGE_KINDS: Tuple[str, ...] = (
    *MODULE_KINDS,
    *PARAMETERLESS_KINDS,
    *neuron_kinds(),
)

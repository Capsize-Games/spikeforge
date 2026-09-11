"""Naming convention linking topology stages to their exported NIR nodes.

A spiking stage always names its spike-carrying node after the stage itself,
so graph edges read stage-to-stage and the simulator's ``outputs[stage]``
lines up with the NIR node value. Integrator and reset nodes that a stage
expands into carry a documented suffix, which is how the validator finds the
membrane trace for a stage without re-deriving the mapping.
"""

from typing import Mapping, Optional

#: Suffix of the membrane integrator node of a split neuron stage.
MEMBRANE_SUFFIX = "__mem"
#: Suffix of the one-step reset feedback delay node.
RESET_DELAY_SUFFIX = "__reset_delay"
#: Suffix of the reset feedback scaling node.
RESET_SCALE_SUFFIX = "__reset_scale"

#: Kinds that split into a membrane node plus a spike node.
SPLIT_KINDS = ("leaky", "lapicque")


def spike_node(stage_name: str) -> str:
    """Return the node name carrying a stage's spike output."""
    return stage_name


def membrane_node(stage_name: str, kind: str) -> str:
    """Return the primary membrane node name for a neuron ``kind``."""
    if kind in SPLIT_KINDS:
        return f"{stage_name}{MEMBRANE_SUFFIX}"
    return stage_name


def reset_delay_node(stage_name: str) -> str:
    """Return the reset feedback delay node name for a stage."""
    return f"{stage_name}{RESET_DELAY_SUFFIX}"


def reset_scale_node(stage_name: str) -> str:
    """Return the reset feedback scaling node name for a stage."""
    return f"{stage_name}{RESET_SCALE_SUFFIX}"


def find_membrane(
    stage_name: str, kind: str, membranes: Mapping[str, object]
) -> Optional[str]:
    """Return whichever membrane node name is present in ``membranes``.

    Split stages expose ``<stage>__mem``; single-node neurons expose the
    stage name itself. Returns ``None`` when neither trace was recorded.
    """
    candidate = membrane_node(stage_name, kind)
    if candidate in membranes:
        return candidate
    if stage_name in membranes:
        return stage_name
    return None

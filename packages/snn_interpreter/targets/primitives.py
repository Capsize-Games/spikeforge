"""The NIR primitives the topology mapper can emit.

This mirrors the mapper's builder tables in one place so a target's declared
support and the capability matrix agree on the same vocabulary. It is a plain
constant, not a probe: a target declares what it *can* run, independent of
what this machine happens to have installed.
"""

from typing import FrozenSet

#: Every NIR node class the mapper emits for a shipped topology.
EMITTED_PRIMITIVES: FrozenSet[str] = frozenset(
    {
        "Input",
        "Output",
        "Affine",
        "Linear",
        "Conv2d",
        "Conv1d",
        "Flatten",
        "AvgPool2d",
        "SumPool2d",
        "LIF",
        "LI",
        "CubaLIF",
        "Delay",
        "Scale",
        "Threshold",
        "IF",
    }
)

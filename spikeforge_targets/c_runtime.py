"""Capability profile for the nir-c-runtime C deployment target.

The Python reference interpreter implements 14 node kinds, but the C
runtime's loader (``KIND_TABLE`` in ``include/nir_c_runtime/graph.h``)
implements only 11: ``Linear``/``Conv1d``/``LIF``/``IF``/``CubaLIF`` fail
loading with an ``unsupported node kind`` error. This module owns the
supported-kind set as data so it can track the C repository's releases.

This is a profile, not a registry entry: :func:`spec` returns a
:class:`TargetSpec` that :func:`classify` accepts, so capability rows can
be produced on demand, but the target stays out of ``builtin_targets()``
because it has no executable simulator backend and every registered
target is required to have one.
"""

from typing import Dict, FrozenSet

from spikeforge_targets.target_spec import TargetSpec

#: Registry-style name used in reports and matrices.
TARGET_NAME = "nir-c-runtime"
#: nir-c-runtime commit whose ``KIND_TABLE`` this profile tracks.
#: Re-audit ``graph.h``/``graph.c``/``ops.c``/``runtime.c`` when it moves.
TRACKED_COMMIT = "7348c82"

#: Every NIR node kind the C loader implements, in ``KIND_TABLE`` order.
#: Single-sourced: the validator and the profile doc both reference this.
SUPPORTED: FrozenSet[str] = frozenset(
    {
        "Input",
        "Output",
        "Conv2d",
        "Affine",
        "Flatten",
        "AvgPool2d",
        "SumPool2d",
        "LI",
        "Threshold",
        "Scale",
        "Delay",
    }
)

#: spikeforge-side origins of the known unsupported kinds, for messages.
KNOWN_GAPS: Dict[str, str] = {
    "Linear": (
        "bias-free linear stages export as Linear; give the stage a "
        "bias to emit the C-supported Affine instead"
    ),
    "Conv1d": (
        "conv1d stages export as Conv1d, which the C loader does not implement"
    ),
    "LIF": (
        "zero-reset leaky/lapicque and recurrent stages export as LIF; "
        "subtract-reset renders as LI+Threshold+Delay+Scale instead"
    ),
    "IF": "IF nodes have no C implementation in the tracked runtime",
    "CubaLIF": (
        "every synaptic stage maps to CubaLIF, which the C loader "
        "does not implement"
    ),
}


def spec() -> TargetSpec:
    """Return the ``nir-c-runtime`` target spec for capability checks."""
    return TargetSpec(
        name=TARGET_NAME,
        kind="hardware",
        description=(
            "nir-c-runtime C loader for embedded targets; validated "
            "pure-Python against its declared node-kind profile."
        ),
        extra="nir-c-runtime",
        supported=SUPPORTED,
        constraints={
            "dtype": "float32",
            "timestep_ms": 1.0,
            "quantization": "none",
            "activation_quantization": "none",
        },
    )

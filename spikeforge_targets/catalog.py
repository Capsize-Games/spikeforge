"""Built-in target specifications.

The reference interpreter is always available and declares support for every
primitive the mapper can emit. Every other entry is an honest placeholder: it
names the pip extra that would install the enabling SDK, declares only the
primitives it can genuinely run (or substitute), and stays reported
unavailable until that SDK is present. Nothing here imports a backend.
"""

from typing import Any, Dict, FrozenSet, List

from spikeforge_targets.primitives import EMITTED_PRIMITIVES
from spikeforge_targets.target_spec import TargetSpec

#: Constraint keys every target declares, beyond the quantization field.
_BASE_CONSTRAINTS: Dict[str, Any] = {
    "dtype": "float32",
    "timestep_ms": 1.0,
}

#: Primitives SynSense Speck runs: a single conv/LIF chain, no pooling.
_SPECK_SUPPORTED: FrozenSet[str] = frozenset(
    {
        "Input",
        "Output",
        "Conv2d",
        "Linear",
        "Flatten",
        "LIF",
        "Scale",
        "Threshold",
    }
)

#: Primitives SynSense Xylo runs: a low-power LIF fabric, no conv/pooling.
_XYLO_SUPPORTED: FrozenSet[str] = frozenset(
    {
        "Input",
        "Output",
        "Affine",
        "Linear",
        "LIF",
        "LI",
        "Scale",
        "Threshold",
        "Delay",
    }
)


def _constrained(quantization: str) -> Dict[str, Any]:
    """Return the standard constraint mapping for a target."""
    return {**_BASE_CONSTRAINTS, "quantization": quantization}


def _without(*primitives: str) -> FrozenSet[str]:
    """Return every emitted primitive except ``primitives``."""
    return frozenset(EMITTED_PRIMITIVES.difference(primitives))


def _reference() -> TargetSpec:
    """Return the always-available in-process reference interpreter."""
    return TargetSpec(
        name="reference",
        kind="reference",
        description="In-process NIR interpreter; no SDK required.",
        extra=None,
        supported=EMITTED_PRIMITIVES,
        constraints=_constrained("none"),
    )


def _lava() -> TargetSpec:
    """Return the Lava / Loihi 2 placeholder target.

    Loihi 2 has no native average pooling, so ``AvgPool2d`` is substituted by
    a sum-pool followed by a ``Scale`` of ``1/k`` (both supported primitives),
    and the third-order ``CubaLIF`` and non-leaky ``IF`` are unsupported.
    """
    return TargetSpec(
        name="lava_loihi2",
        kind="hardware",
        description="Lava SDK path to Intel Loihi 2.",
        extra="lava",
        supported=_without("AvgPool2d", "CubaLIF", "IF"),
        constraints=_constrained("weight_int8"),
        substitutions={"AvgPool2d": "SumPool2d"},
    )


def _spinnaker() -> TargetSpec:
    """Return the SpiNNaker2 placeholder target (no third-order neuron)."""
    return TargetSpec(
        name="spinnaker2",
        kind="hardware",
        description="SpiNNaker2 digital neuromorphic hardware.",
        extra="spinnaker2",
        supported=_without("CubaLIF"),
        constraints=_constrained("weight_int8"),
    )


def _speck() -> TargetSpec:
    """Return the SynSense Speck placeholder target."""
    return TargetSpec(
        name="speck",
        kind="hardware",
        description="SynSense Speck edge chip; one conv/LIF chain.",
        extra="speck",
        supported=_SPECK_SUPPORTED,
        constraints=_constrained("weight_int8"),
    )


def _xylo() -> TargetSpec:
    """Return the SynSense Xylo placeholder target."""
    return TargetSpec(
        name="xylo",
        kind="hardware",
        description="SynSense Xylo low-power LIF fabric; no conv.",
        extra="xylo",
        supported=_XYLO_SUPPORTED,
        constraints=_constrained("weight_uint8"),
    )


def _norse() -> TargetSpec:
    """Return the Norse PyTorch simulator placeholder target.

    Norse has no ``CubaLIF``, no explicit ``Delay`` node, and no non-leaky
    ``IF``; a ``beta=0`` LIF reproduces ``IF`` exactly, so that primitive is
    substituted rather than dropped.
    """
    return TargetSpec(
        name="norse",
        kind="simulator",
        description="Norse PyTorch simulator; no CubaLIF or Delay.",
        extra="norse",
        supported=_without("CubaLIF", "Delay", "IF"),
        constraints=_constrained("none"),
        substitutions={"IF": "LIF"},
    )


def builtin_targets() -> List[TargetSpec]:
    """Return the built-in target specs in registry order."""
    return [
        _reference(),
        _lava(),
        _spinnaker(),
        _speck(),
        _xylo(),
        _norse(),
    ]

"""Named fixed-point grids and their applied/refused reports.

Split out of the ``activation_quant`` package root so each scheme class
lives in its own file. A scheme names a symmetric grid width and the
tensors it applies to; :func:`scheme_report` renders the report for a
requested name without ever approximating an unknown one.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge_targets.activation_quant_records import RangeRecords
from spikeforge_targets.activation_quant_report import (
    ActivationQuantizationReport,
)
from spikeforge_targets.fixed_point import EPS, levels_for

if TYPE_CHECKING:
    from spikeforge_targets.activation_quant.calibration import Calibration

#: Quantize spiking activations only.
TARGET_ACTIVATION = "activation"
#: Quantize membrane potentials and neuron input currents only.
TARGET_MEMBRANE = "membrane"
#: Quantize activations, membranes, and input currents.
TARGET_BOTH = "both"
#: Every target a scheme may name.
TARGETS: Tuple[str, ...] = (TARGET_ACTIVATION, TARGET_MEMBRANE, TARGET_BOTH)

#: Reason recorded when a scheme opts out of quantizing anything.
NO_ACTIVATION_SCHEME = "activation quantization is disabled"
#: Reason recorded for a scheme name this module cannot apply.
UNKNOWN_ACTIVATION_SCHEME = "unknown activation quantization scheme {scheme!r}"


@dataclass(frozen=True)
class FixedPointScheme:
    """One symmetric fixed-point grid and the tensors it applies to."""

    name: str
    bits: int
    target: str

    def levels(self) -> float:
        """Return the largest magnitude the scheme's grid can represent."""
        return levels_for(self.bits)

    def covers(self, target: str) -> bool:
        """Return True when the scheme quantizes ``target``'s tensors."""
        return self.target in (target, TARGET_BOTH)

    def grid(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Return the codes, scale, and level count for ``tensor``."""
        bound = float(tensor.abs().max()) if tensor.numel() else 0.0
        levels = self.levels()
        if bound <= EPS or levels <= 0.0:
            return torch.zeros_like(tensor), 0.0, int(levels)
        scale = bound / levels
        codes = torch.round(tensor / scale).clamp(-levels, levels)
        return codes, scale, int(levels)


def _none_scheme() -> FixedPointScheme:
    """Return the explicit no-op scheme."""
    return FixedPointScheme("none", 0, "none")


def _scheme(
    name: str, bits: int, target: str
) -> FixedPointScheme:
    """Return a named scheme with its grid width and target."""
    return FixedPointScheme(name, bits, target)


#: The executable scheme for each declared activation-quantization name.
SCHEMES: Dict[str, FixedPointScheme] = {
    "none": _none_scheme(),
    "activation_int8": _scheme(
        "activation_int8", 8, TARGET_ACTIVATION
    ),
    "membrane_int8": _scheme("membrane_int8", 8, TARGET_MEMBRANE),
    "activation_membrane_int8": _scheme(
        "activation_membrane_int8", 8, TARGET_BOTH
    ),
}


def scheme_names() -> Tuple[str, ...]:
    """Return the supported activation-quantization scheme names."""
    return tuple(SCHEMES)


def scheme_for(name: str) -> Optional[FixedPointScheme]:
    """Return the scheme called ``name``, or ``None`` when unsupported."""
    return SCHEMES.get(str(name))


def refused_report(
    scheme: str,
    reason: str,
    *,
    steps: int = 0,
    calibration: Optional[Mapping[str, Any]] = None,
) -> ActivationQuantizationReport:
    """Return the report of ``scheme`` left unapplied for ``reason``."""
    resolved = scheme_for(scheme)
    return ActivationQuantizationReport(
        scheme=scheme,
        bits=0 if resolved is None else resolved.bits,
        target="unknown" if resolved is None else resolved.target,
        applied=False,
        reason=reason,
        steps=steps,
        calibration=calibration,
    )


def _applied_report(
    scheme: FixedPointScheme,
    records: RangeRecords,
    steps: int,
    block: Optional[Mapping[str, Any]],
) -> ActivationQuantizationReport:
    """Return the applied report for ``scheme`` and its ``records``."""
    return ActivationQuantizationReport(
        scheme=scheme.name,
        bits=scheme.bits,
        target=scheme.target,
        applied=True,
        reason="",
        layers=records.layers(),
        steps=steps,
        calibration=block,
    )


def _refused_by_name(
    requested: str, scheme: Optional[FixedPointScheme], steps: int,
    block: Optional[Mapping[str, Any]],
) -> ActivationQuantizationReport:
    """Return the refusal report for an unknown or no-op ``requested``."""
    if scheme is None:
        reason = UNKNOWN_ACTIVATION_SCHEME.format(scheme=requested)
        return refused_report(
            requested, reason, steps=steps, calibration=block
        )
    return refused_report(
        scheme.name, NO_ACTIVATION_SCHEME, steps=steps, calibration=block
    )


def scheme_report(
    requested: str,
    records: RangeRecords,
    *,
    steps: int,
    calibration: Optional["Calibration"],
) -> ActivationQuantizationReport:
    """Return the report for ``requested``: applied, or refused by name."""
    scheme = scheme_for(requested)
    block = None if calibration is None else calibration.to_dict()
    if scheme is None or scheme.target == "none":
        return _refused_by_name(requested, scheme, steps, block)
    return _applied_report(scheme, records, steps, block)

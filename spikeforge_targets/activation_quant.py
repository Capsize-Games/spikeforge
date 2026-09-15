"""Simulated fixed-point activation and membrane quantization.

The core step path stays scheme-agnostic: it accepts an optional ``post_step``
transform, and this module supplies one. :class:`ActivationQuantizer` is that
transform. It maps every floating activation (a stage output, and the
previous-step mirror a delayed edge reads) and every floating membrane or
input-current tensor in the carried state onto a symmetric fixed-point grid
with ``2 ** (bits - 1) - 1`` levels, recording the per-stage ranges and the
absolute error the grid introduced.

Weight-only quantization remains the default everywhere; nothing is applied
until a caller passes a quantizer to
:meth:`spikeforge.serving.session.InferenceSession.load`. A scheme name this
module does not implement is refused in the report rather than approximated,
and ``none`` is an explicit no-op. :func:`calibrate` is the calibration-dataset
hook: feed it observed tensors once, then hand the resulting ranges to the
quantizer so the grid is chosen from data instead of per-step extremes.

The same schemes, grid, and report shape serve the NIR side: the
:class:`~spikeforge_targets.activation_quant_graph.GraphActivationQuantizer`
hook applies them to a reference-interpreter run.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, TypeGuard

import torch

from spikeforge.topology.stage_module import CURRENT_KEY, PREV_KEY
from spikeforge_targets.activation_quant_records import (
    RangeRecords,
    tensor_range,
)
from spikeforge_targets.activation_quant_report import (
    ActivationQuantizationReport,
)
from spikeforge_targets.fixed_point import EPS, levels_for, snap

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

#: A running (min, max) per calibration key.
Ranges = Dict[str, Tuple[float, float]]


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
    if scheme is None:
        reason = UNKNOWN_ACTIVATION_SCHEME.format(scheme=requested)
        return refused_report(
            requested, reason, steps=steps, calibration=block
        )
    if scheme.target == "none":
        return refused_report(
            scheme.name, NO_ACTIVATION_SCHEME, steps=steps, calibration=block
        )
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


def is_float(value: Any) -> TypeGuard[torch.Tensor]:
    """Return True when ``value`` is a floating torch tensor.

    Declared as a :class:`TypeGuard` so a caller holding an optional or
    duck-typed value is narrowed to ``Tensor`` by the check itself, rather
    than asserting the narrowing separately.
    """
    return torch.is_tensor(value) and value.is_floating_point()


@dataclass(frozen=True)
class Calibration:
    """Per-stage ranges observed on a calibration dataset.

    ``ranges`` maps the same stage keys the quantizer uses to an ``(min, max)``
    pair, and ``samples`` records how many tensors were folded in. A quantizer
    handed a calibration uses these bounds in place of its per-step extremes,
    so the grid is stable from the first served step. ``source`` names where
    the ranges came from, so a report can say whether the grid was chosen
    from a separate dataset or from the very fixture it is checked on.
    """

    bits: int
    target: str
    ranges: Mapping[str, Tuple[float, float]]
    samples: int = 0
    source: str = ""

    def bound(self, key: str) -> Optional[float]:
        """Return the symmetric magnitude for ``key``, or ``None``."""
        span = self.ranges.get(key)
        if span is None:
            return None
        return max(abs(float(span[0])), abs(float(span[1])))

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the calibration."""
        return {
            "bits": self.bits,
            "target": self.target,
            "samples": self.samples,
            "source": self.source,
            "ranges": {
                key: [float(span[0]), float(span[1])]
                for key, span in self.ranges.items()
            },
        }


def fold_range(seen: Ranges, key: str, value: Any) -> bool:
    """Fold ``value``'s range into ``seen`` under ``key``.

    Returns True when a floating tensor was folded and False when the value
    was ignored, so a caller can count samples.
    """
    if not is_float(value):
        return False
    low, high = tensor_range(torch.as_tensor(value).float())
    earlier = seen.get(key)
    if earlier is None:
        seen[key] = (low, high)
    else:
        seen[key] = (min(earlier[0], low), max(earlier[1], high))
    return True


def calibrate(
    tensors: Iterable[Tuple[str, Any]],
    *,
    bits: int = 8,
    target: str = TARGET_BOTH,
    source: str = "",
) -> Calibration:
    """Return per-key ranges folded from an observed tensor iterable.

    ``tensors`` yields ``(key, tensor)`` pairs - typically the stage outputs
    and carried-state tensors of a short calibration pass. Non-floating
    values are ignored; the ranges are the running min/max per key.
    """
    seen: Ranges = {}
    samples = 0
    for key, value in tensors:
        if fold_range(seen, key, value):
            samples += 1
    return Calibration(int(bits), str(target), seen, samples, str(source))


def calibrated_bound(
    calibration: Optional[Calibration], key: str, tensor: torch.Tensor
) -> float:
    """Return the calibration bound for ``key``, or the tensor's own peak."""
    if calibration is not None:
        bound = calibration.bound(key)
        if bound is not None and bound > EPS:
            return bound
    return float(tensor.abs().max()) if tensor.numel() else 0.0


class ActivationQuantizer:
    """A ``post_step`` transform that simulates fixed-point quantization.

    Instances are callables: ``quantizer(outputs, state)`` returns a new
    ``(outputs, state)`` pair with the targeted tensors restricted to the
    scheme's grid. Nothing is mutated in place. :meth:`report` returns an
    :class:`ActivationQuantizationReport` naming the scheme, each stage's
    ranges, and the error introduced, or the honest reason it was refused.
    """

    def __init__(
        self,
        scheme: str = "activation_membrane_int8",
        *,
        calibration: Optional[Calibration] = None,
        neuron_names: Optional[Iterable[str]] = None,
    ) -> None:
        """Resolve ``scheme`` and optionally restrict which stages apply."""
        self._requested = str(scheme)
        self._scheme = scheme_for(scheme)
        self._calibration = calibration
        self._neurons = (
            None if neuron_names is None else frozenset(neuron_names)
        )
        self._records = RangeRecords()
        self._steps = 0

    @property
    def applied(self) -> bool:
        """Return True when a scheme will quantize at least one tensor."""
        return self._scheme is not None and self._scheme.target != "none"

    def _quantize(
        self, key: str, tensor: torch.Tensor, kind: str
    ) -> torch.Tensor:
        """Return ``tensor`` quantized and fold its error into the record."""
        levels = self._scheme.levels() if self._scheme else 0.0
        if levels <= 0.0:
            return tensor
        bound = calibrated_bound(self._calibration, key, tensor)
        quantized = snap(tensor, bound, levels)
        self._records.fold(key, kind, tensor, quantized)
        return quantized

    def _outputs(
        self, outputs: Mapping[str, torch.Tensor], active: bool
    ) -> Dict[str, torch.Tensor]:
        """Return ``outputs`` with activations quantized when ``active``."""
        if not active:
            return dict(outputs)
        return {
            name: (
                self._quantize(name, value, TARGET_ACTIVATION)
                if is_float(value)
                else value
            )
            for name, value in outputs.items()
        }

    def _neuron(
        self, name: str, value: Any, active: bool
    ) -> Any:
        """Return one neuron state entry with its membranes quantized."""
        if not active:
            return value
        if is_float(value):
            return self._quantize(name, value, TARGET_MEMBRANE)
        if isinstance(value, tuple):
            return tuple(
                self._quantize(
                    f"{name}[{index}]", item, TARGET_MEMBRANE
                )
                if is_float(item)
                else item
                for index, item in enumerate(value)
            )
        return value

    def _currents(self, current: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the input-current entries quantized as membranes."""
        return {
            name: (
                self._quantize(f"{name}.current", value, TARGET_MEMBRANE)
                if is_float(value)
                else value
            )
            for name, value in current.items()
        }

    def __call__(
        self,
        outputs: Mapping[str, torch.Tensor],
        state: Mapping[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Return the step's outputs and state with the grids applied."""
        if not self.applied or self._scheme is None:
            return dict(outputs), dict(state)
        activation = self._scheme.covers(TARGET_ACTIVATION)
        membrane = self._scheme.covers(TARGET_MEMBRANE)
        new_outputs = self._outputs(outputs, activation)
        new_state: Dict[str, Any] = dict(state)
        previous = state.get(PREV_KEY)
        if activation and isinstance(previous, Mapping):
            new_state[PREV_KEY] = {
                name: new_outputs.get(name, value)
                for name, value in previous.items()
            }
        current = state.get(CURRENT_KEY)
        if membrane and isinstance(current, Mapping):
            new_state[CURRENT_KEY] = self._currents(current)
        for name, value in state.items():
            if name in (PREV_KEY, CURRENT_KEY):
                continue
            if self._neurons is not None and name not in self._neurons:
                continue
            new_state[name] = self._neuron(name, value, membrane)
        self._steps += 1
        return new_outputs, new_state

    def report(self) -> ActivationQuantizationReport:
        """Return the report of everything observed so far."""
        return scheme_report(
            self._requested,
            self._records,
            steps=self._steps,
            calibration=self._calibration,
        )

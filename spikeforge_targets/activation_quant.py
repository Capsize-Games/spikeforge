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
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from spikeforge.topology.stage_module import CURRENT_KEY, PREV_KEY
from spikeforge_targets.activation_quant_report import (
    ActivationQuantizationReport,
)

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

#: Peak below which a tensor is treated as all-zero.
_EPS = 1e-12


@dataclass(frozen=True)
class FixedPointScheme:
    """One symmetric fixed-point grid and the tensors it applies to."""

    name: str
    bits: int
    target: str

    def levels(self) -> float:
        """Return the largest magnitude the scheme's grid can represent."""
        if self.bits <= 0:
            return 0.0
        return float(2 ** (self.bits - 1) - 1)

    def grid(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Return the codes, scale, and level count for ``tensor``."""
        bound = float(tensor.abs().max()) if tensor.numel() else 0.0
        levels = self.levels()
        if bound <= _EPS or levels <= 0.0:
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


def _is_float(value: Any) -> bool:
    """Return True when ``value`` is a floating torch tensor."""
    return torch.is_tensor(value) and value.is_floating_point()


def _range(tensor: torch.Tensor) -> Tuple[float, float]:
    """Return the (min, max) of ``tensor`` as floats, or zeros when empty."""
    if not tensor.numel():
        return (0.0, 0.0)
    return (float(tensor.min()), float(tensor.max()))


@dataclass(frozen=True)
class Calibration:
    """Per-stage ranges observed on a calibration dataset.

    ``ranges`` maps the same stage keys the quantizer uses to an ``(min, max)``
    pair, and ``samples`` records how many tensors were folded in. A quantizer
    handed a calibration uses these bounds in place of its per-step extremes,
    so the grid is stable from the first served step.
    """

    bits: int
    target: str
    ranges: Mapping[str, Tuple[float, float]]
    samples: int = 0

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
            "ranges": {
                key: [float(span[0]), float(span[1])]
                for key, span in self.ranges.items()
            },
        }


def calibrate(
    tensors: Iterable[Tuple[str, Any]],
    *,
    bits: int = 8,
    target: str = TARGET_BOTH,
) -> Calibration:
    """Return per-key ranges folded from an observed tensor iterable.

    ``tensors`` yields ``(key, tensor)`` pairs - typically the stage outputs
    and carried-state tensors of a short calibration pass. Non-floating
    values are ignored; the ranges are the running min/max per key.
    """
    seen: Dict[str, Tuple[float, float]] = {}
    samples = 0
    for key, value in tensors:
        if not _is_float(value):
            continue
        low, high = _range(torch.as_tensor(value).float())
        samples += 1
        if key in seen:
            earlier = seen[key]
            seen[key] = (min(earlier[0], low), max(earlier[1], high))
        else:
            seen[key] = (low, high)
    return Calibration(int(bits), str(target), seen, samples)


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
        self._records: Dict[str, Dict[str, Any]] = {}
        self._steps = 0

    @property
    def applied(self) -> bool:
        """Return True when a scheme will quantize at least one tensor."""
        return self._scheme is not None and self._scheme.target != "none"

    def _bound(self, key: str, tensor: torch.Tensor) -> float:
        """Return the calibration bound for ``key``, or the tensor's peak."""
        if self._calibration is not None:
            bound = self._calibration.bound(key)
            if bound is not None and bound > _EPS:
                return bound
        return float(tensor.abs().max()) if tensor.numel() else 0.0

    def _quantize(
        self, key: str, tensor: torch.Tensor, kind: str
    ) -> torch.Tensor:
        """Return ``tensor`` quantized and fold its error into the record."""
        levels = self._scheme.levels() if self._scheme else 0.0
        if levels <= 0.0:
            return tensor
        bound = self._bound(key, tensor)
        if bound <= _EPS:
            quantized = torch.zeros_like(tensor)
        else:
            scale = bound / levels
            codes = torch.round(tensor / scale).clamp(-levels, levels)
            quantized = codes * scale
        self._record(key, kind, tensor, quantized)
        return quantized

    def _record(
        self,
        key: str,
        kind: str,
        before: torch.Tensor,
        after: torch.Tensor,
    ) -> None:
        """Fold one tensor's before/after ranges and error into the record."""
        record = self._records.get(key)
        if record is None:
            record = {
                "name": key,
                "kind": kind,
                "before": [float("inf"), float("-inf")],
                "after": [float("inf"), float("-inf")],
                "max_abs": 0.0,
                "_abs_sum": 0.0,
                "_count": 0,
            }
            self._records[key] = record
        low, high = _range(before)
        record["before"][0] = min(record["before"][0], low)
        record["before"][1] = max(record["before"][1], high)
        low, high = _range(after)
        record["after"][0] = min(record["after"][0], low)
        record["after"][1] = max(record["after"][1], high)
        difference = (after - before).abs()
        if difference.numel():
            record["max_abs"] = max(
                record["max_abs"], float(difference.max())
            )
            record["_abs_sum"] += float(difference.sum())
            record["_count"] += int(difference.numel())

    def _outputs(
        self, outputs: Mapping[str, torch.Tensor], active: bool
    ) -> Dict[str, torch.Tensor]:
        """Return ``outputs`` with activations quantized when ``active``."""
        if not active:
            return dict(outputs)
        return {
            name: (
                self._quantize(name, value, TARGET_ACTIVATION)
                if _is_float(value)
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
        if _is_float(value):
            return self._quantize(name, value, TARGET_MEMBRANE)
        if isinstance(value, tuple):
            return tuple(
                self._quantize(
                    f"{name}[{index}]", item, TARGET_MEMBRANE
                )
                if _is_float(item)
                else item
                for index, item in enumerate(value)
            )
        return value

    def __call__(
        self,
        outputs: Mapping[str, torch.Tensor],
        state: Mapping[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Return the step's outputs and state with the grids applied."""
        if not self.applied:
            return dict(outputs), dict(state)
        target = self._scheme.target if self._scheme else "none"
        activation = target in (TARGET_ACTIVATION, TARGET_BOTH)
        membrane = target in (TARGET_MEMBRANE, TARGET_BOTH)
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
            new_state[CURRENT_KEY] = {
                name: (
                    self._quantize(
                        f"{name}.current", value, TARGET_MEMBRANE
                    )
                    if _is_float(value)
                    else value
                )
                for name, value in current.items()
            }
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
        scheme = self._scheme
        if scheme is None:
            return ActivationQuantizationReport(
                scheme=self._requested,
                bits=0,
                target="unknown",
                applied=False,
                reason=UNKNOWN_ACTIVATION_SCHEME.format(
                    scheme=self._requested
                ),
                steps=self._steps,
                calibration=self._calibration_dict(),
            )
        if scheme.target == "none":
            return ActivationQuantizationReport(
                scheme=scheme.name,
                bits=scheme.bits,
                target=scheme.target,
                applied=False,
                reason=NO_ACTIVATION_SCHEME,
                steps=self._steps,
                calibration=self._calibration_dict(),
            )
        layers = tuple(
            self._layer(record)
            for record in sorted(
                self._records.values(), key=lambda item: item["name"]
            )
        )
        return ActivationQuantizationReport(
            scheme=scheme.name,
            bits=scheme.bits,
            target=scheme.target,
            applied=True,
            reason="",
            layers=layers,
            steps=self._steps,
            calibration=self._calibration_dict(),
        )

    def _layer(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        """Return the JSON-able per-stage record without private counters."""
        count = int(record["_count"])
        return {
            "name": record["name"],
            "kind": record["kind"],
            "before": [float(record["before"][0]), float(record["before"][1])],
            "after": [float(record["after"][0]), float(record["after"][1])],
            "max_abs": float(record["max_abs"]),
            "mean_abs": (
                float(record["_abs_sum"]) / count if count else 0.0
            ),
        }

    def _calibration_dict(self) -> Optional[Dict[str, Any]]:
        """Return the calibration block for the report, or ``None``."""
        if self._calibration is None:
            return None
        return self._calibration.to_dict()

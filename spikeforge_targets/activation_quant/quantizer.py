"""The ``post_step`` transform simulating fixed-point quantization.

Split out of the ``activation_quant`` package root so each scheme class
lives in its own file. :class:`ActivationQuantizer` maps every floating
activation and every floating membrane or input-current tensor onto the
scheme's symmetric fixed-point grid, recording the ranges and the error
the grid introduced.
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from spikeforge.topology.stage_module import CURRENT_KEY, PREV_KEY
from spikeforge_targets.activation_quant.calibration import (
    Calibration,
    calibrated_bound,
    is_float,
)
from spikeforge_targets.activation_quant.scheme import (
    TARGET_ACTIVATION,
    TARGET_MEMBRANE,
    scheme_for,
    scheme_report,
)
from spikeforge_targets.activation_quant_records import RangeRecords
from spikeforge_targets.activation_quant_report import (
    ActivationQuantizationReport,
)
from spikeforge_targets.fixed_point import snap


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

    def _mirror(
        self,
        state: Mapping[str, Any],
        new_outputs: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, Any]]:
        """Return the quantized previous-step mirror, or ``None``."""
        previous = state.get(PREV_KEY)
        if not isinstance(previous, Mapping):
            return None
        return {
            name: new_outputs.get(name, value)
            for name, value in previous.items()
        }

    def _neuron_entries(
        self, state: Mapping[str, Any], new_state: Dict[str, Any],
        membrane: bool,
    ) -> None:
        """Quantize the per-neuron entries of ``state`` into ``new_state``."""
        for name, value in state.items():
            if name in (PREV_KEY, CURRENT_KEY):
                continue
            if self._neurons is not None and name not in self._neurons:
                continue
            new_state[name] = self._neuron(name, value, membrane)

    def _state(
        self,
        state: Mapping[str, Any],
        new_outputs: Dict[str, torch.Tensor],
        activation: bool,
        membrane: bool,
    ) -> Dict[str, Any]:
        """Return ``state`` with mirror, currents, neurons quantized."""
        new_state: Dict[str, Any] = dict(state)
        if activation:
            mirrored = self._mirror(state, new_outputs)
            if mirrored is not None:
                new_state[PREV_KEY] = mirrored
        current = state.get(CURRENT_KEY)
        if membrane and isinstance(current, Mapping):
            new_state[CURRENT_KEY] = self._currents(current)
        self._neuron_entries(state, new_state, membrane)
        return new_state

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
        new_state = self._state(state, new_outputs, activation, membrane)
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

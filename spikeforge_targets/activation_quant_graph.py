"""Simulated fixed-point quantization of a reference-interpreter run.

Activation and membrane quantization cannot live in a NIR graph: the graph
carries weights and neuron parameters, not the values that flow through it.
So the NIR side simulates it at execution time.
:class:`GraphActivationQuantizer` is a ``post_node`` hook for
:class:`~spikeforge.nir_bridge.interpreter.NirInterpreter` that snaps every
computed, non-spike node output (an ``activation``), every carried neuron
state tensor, and the membrane the interpreter records (a ``membrane``, or a
CubaLIF synaptic ``current``) onto the same symmetric fixed-point grid the
serving-side
:class:`~spikeforge_targets.activation_quant.ActivationQuantizer` uses, and
records ranges and error under the same report shape.

Which tensors a node contributes, and the keys they are reported under, are
:mod:`~spikeforge_targets.activation_quant_keys`'s business alone.

Spikes are binary, exact on any grid, and never listed. The snap happens
after each node's floating update, so the threshold comparison sees the float
membrane, while the grid applies to what is carried into the next step and to
the membrane the interpreter records. Only the carried value propagates; the
recorded one is read by no node, so snapping it changes the reported trace
and nothing else. A vendor kernel that computes the update itself in fixed
point can differ, which is why this is a simulation and never a device result.
:func:`calibrate_graph` folds per-key ranges from one plain run so the grid
is fixed for a whole check instead of rescaled to each call's peak.
"""

from typing import Any, Optional, Tuple

import torch

from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.ops_registry import OUTPUT
from spikeforge_targets.activation_quant import (
    TARGET_ACTIVATION,
    TARGET_BOTH,
    TARGET_MEMBRANE,
    Calibration,
    FixedPointScheme,
    Ranges,
    calibrated_bound,
    fold_range,
    is_float,
    scheme_for,
    scheme_report,
)
from spikeforge_targets.activation_quant_keys import (
    EXACT_KINDS,
    MEMBRANE_SUFFIX,
    node_tensors,
    state_keys,
)
from spikeforge_targets.activation_quant_records import RangeRecords
from spikeforge_targets.activation_quant_report import (
    ActivationQuantizationReport,
)
from spikeforge_targets.fixed_point import snap


def calibrate_graph(
    graph: Any,
    spikes: torch.Tensor,
    *,
    bits: int = 8,
    target: str = TARGET_BOTH,
    source: str = "",
) -> Calibration:
    """Return the per-key ranges of one plain run of ``graph`` on ``spikes``.

    The run is unquantized, and the ranges are folded as the interpreter
    goes, so nothing beyond the running extremes is held.
    """
    seen: Ranges = {}
    samples = 0

    def observe(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        nonlocal samples
        observed = node_tensors(name, kind, output, state, membrane)
        for key, _kind, tensor in observed:
            samples += int(fold_range(seen, key, tensor))
        return output, state, membrane

    NirInterpreter(graph, post_node=observe).run(spikes)
    return Calibration(int(bits), str(target), seen, samples, str(source))


class GraphActivationQuantizer:
    """A ``post_node`` hook that simulates fixed-point quantization.

    ``quantizer(name, kind, output, state, membrane)`` returns the triple
    with the scheme's targets snapped: the output when the scheme covers
    activations, the carried state and the recorded membrane when it covers
    membranes. Without a calibration each tensor is scaled to its own peak on
    every call; with one the grid is fixed by the calibrated bound and values
    beyond it clip. :meth:`report` has the serving quantizer's shape and
    counts one step per ``Output`` evaluation.
    """

    def __init__(
        self,
        scheme: str = "activation_membrane_int8",
        *,
        calibration: Optional[Calibration] = None,
    ) -> None:
        """Resolve ``scheme`` and keep the optional calibrated bounds."""
        self._requested = str(scheme)
        self._scheme = scheme_for(scheme)
        self._calibration = calibration
        self._records = RangeRecords()
        self._steps = 0

    @property
    def scheme(self) -> Optional[FixedPointScheme]:
        """Return the resolved scheme, or ``None`` when unsupported."""
        return self._scheme

    @property
    def applied(self) -> bool:
        """Return True when the scheme will snap at least one tensor kind."""
        return self._scheme is not None and self._scheme.target != "none"

    @property
    def targets(self) -> Tuple[str, ...]:
        """Return the tensor kinds the scheme snaps, in report order."""
        if self._scheme is None or not self.applied:
            return ()
        return tuple(
            kind
            for kind in (TARGET_ACTIVATION, TARGET_MEMBRANE)
            if self._scheme.covers(kind)
        )

    def _snap(self, key: str, kind: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` on the grid and fold its error into the record."""
        levels = self._scheme.levels() if self._scheme else 0.0
        bound = calibrated_bound(self._calibration, key, tensor)
        after = snap(tensor, bound, levels)
        self._records.fold(key, kind, tensor, after)
        return after

    def _snap_state(self, name: str, kind: str, state: Any) -> Any:
        """Return ``state`` with every floating tensor on the grid."""
        if is_float(state):
            return self._snap(name + MEMBRANE_SUFFIX, TARGET_MEMBRANE, state)
        if isinstance(state, tuple):
            keys = state_keys(name, kind, len(state))
            return tuple(
                self._snap(key, TARGET_MEMBRANE, item)
                if is_float(item)
                else item
                for key, item in zip(keys, state)
            )
        return state

    def _snap_membrane(
        self, name: str, membrane: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return the recorded membrane snapped, passing ``None`` through.

        ``None`` means the node exposes no membrane trace, which the
        interpreter reads to decide whether to record one at all.
        """
        if not is_float(membrane):
            return membrane
        return self._snap(name + MEMBRANE_SUFFIX, TARGET_MEMBRANE, membrane)

    def __call__(
        self,
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any, Optional[torch.Tensor]]:
        """Return the node's output, state and membrane on the grids."""
        if kind == OUTPUT:
            self._steps += 1
        if self._scheme is None or not self.applied:
            return output, state, membrane
        activation = self._scheme.covers(TARGET_ACTIVATION)
        if activation and kind not in EXACT_KINDS and is_float(output):
            output = self._snap(name, TARGET_ACTIVATION, output)
        if self._scheme.covers(TARGET_MEMBRANE):
            state = self._snap_state(name, kind, state)
            membrane = self._snap_membrane(name, membrane)
        return output, state, membrane

    def report(self) -> ActivationQuantizationReport:
        """Return the report of everything observed so far."""
        return scheme_report(
            self._requested,
            self._records,
            steps=self._steps,
            calibration=self._calibration,
        )

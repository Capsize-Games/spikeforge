"""Vendor-simulator runners behind the isolated SDK probe.

Split out of the ``api`` package root so each vendor's runner lives in its
own file. Speck, Xylo, and SpiNNaker2 share one stepping loop over a lowered
dense/neuron chain; each runner names its simulator path so a result is never
read as an on-chip measurement.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from spikeforge_targets.backends.api.probes import require
from spikeforge_targets.backends.errors import BackendUnavailableError

#: Node kinds a lowered program expresses as a neuron layer.
NEURON_KINDS = ("LIF", "LI")


def _weighted(weight: Any, value: Any) -> Any:
    """Apply one dense layer's weight matrix to a step's activations."""
    matrix = torch.as_tensor(
        np.asarray(weight), dtype=value.dtype, device=value.device
    )
    return value @ matrix.t()


def _neuron_step(neuron: Any, value: Any, state: Any) -> Tuple[Any, Any]:
    """Return the ``(output, state)`` pair for one vendor neuron step."""
    result = neuron(value) if state is None else neuron(value, state)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, result


def _neurons(program: Dict[str, Any], cls: Any) -> List[Any]:
    """Build one neuron per neuron layer and ``None`` for dense layers."""
    return [
        cls(**dict(layer["params"]))
        if layer["kind"] in NEURON_KINDS
        else None
        for layer in program["layers"]
    ]


def _run_step(
    program: Dict[str, Any], value: Any, neurons: List[Any],
    states: Dict[int, Any],
) -> Any:
    """Step one input ``value`` through every layer of ``program``."""
    for position, layer in enumerate(program["layers"]):
        if layer["kind"] in NEURON_KINDS:
            value, states[position] = _neuron_step(
                neurons[position], value, states.get(position)
            )
        else:
            value = _weighted(layer["params"]["weight"], value)
    return value


def _run_program(
    program: Dict[str, Any], spikes: Any, neurons: List[Any]
) -> Any:
    """Step a lowered program through ``neurons`` and return the readout.

    The readout follows the reference/Norse convention: the value reaching the
    output node averaged over the run, so a vendor trajectory compares directly
    against the reference interpreter's.
    """
    steps = int(spikes.shape[0])
    states: Dict[int, Any] = {}
    total = None
    for index in range(steps):
        stepped = _run_step(program, spikes[index], neurons, states)
        total = stepped if total is None else total + stepped
    readout = total / steps if total is not None else spikes.new_zeros(0)
    return np.asarray(readout.detach().cpu())


def _neuron_class(
    module: Any, name: str, candidates: Sequence[str]
) -> Any:
    """Return the first available neuron class, or raise a named reason."""
    holders = (
        module,
        getattr(module, "layers", None),
        getattr(module, "nn", None),
        getattr(module, "snn", None),
    )
    for holder in holders:
        if holder is None:
            continue
        for attr in candidates:
            cls = getattr(holder, attr, None)
            if cls is not None:
                return cls
    raise BackendUnavailableError(
        name, f"installed SDK exposes none of {', '.join(candidates)}"
    )


def _vendor_readout(
    program: Dict[str, Any], spikes: Any, cls: Any, path: str
) -> Dict[str, Any]:
    """Run a lowered program through a vendor neuron class; name the path."""
    return {
        "readout": _run_program(program, spikes, _neurons(program, cls)),
        "path": path,
    }


def sinabs_run(program: Dict[str, Any], spikes: Any) -> Dict[str, Any]:
    """Execute a lowered linear program on the Sinabs / Speck simulator.

    Speck is SynSense's edge chip; this runs the lowered dense/neuron chain on
    the Sinabs software simulator built from the installed LIF neuron. The
    returned path names the simulator, never a probed chip, and upstream API
    drift raises here so the caller reports a named error.
    """
    sinabs = require("sinabs", "speck")
    cls = _neuron_class(sinabs, "speck", ("LIF",))
    return _vendor_readout(program, spikes, cls, "speck_simulator")


def rockpool_run(program: Dict[str, Any], spikes: Any) -> Dict[str, Any]:
    """Execute a lowered linear program on the Rockpool / Xylo simulator.

    Xylo is SynSense's low-power LIF fabric; this runs the lowered chain on
    the Rockpool software simulator. The path names the simulator so a result
    is never read as an on-chip measurement.
    """
    rockpool = require("rockpool", "xylo")
    cls = _neuron_class(rockpool, "xylo", ("LIF", "TorchLIF", "LIFTorch"))
    return _vendor_readout(program, spikes, cls, "xylo_simulator")


def spinnaker2_run(program: Dict[str, Any], spikes: Any) -> Dict[str, Any]:
    """Execute a lowered linear program on the SpiNNaker2 host simulator.

    SpiNNaker2's software simulator runs the lowered chain in process; the path
    names the host simulator so a result is never read as a board measurement.
    """
    spinnaker2 = require("spinnaker2", "spinnaker2")
    cls = _neuron_class(
        spinnaker2, "spinnaker2", ("LIF", "SpikingNeuron", "Neuron")
    )
    return _vendor_readout(program, spikes, cls, "spinnaker2_simulator")

"""Isolated probes and imports for the optional backend SDKs.

This is the only module in the backend package that imports a backend SDK.
Imports happen inside functions, never at module import time, so a missing
package never breaks importing the backends; callers read
:func:`module_available` first and degrade honestly. Norse neuron factories
and the Lava execution touch-point are wrapped here so upstream API churn is
contained to one file.
"""

import os
from importlib import import_module
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from spikeforge_targets.backends.errors import BackendUnavailableError

#: Backend name -> the module whose presence makes it available.
BACKEND_MODULES: Dict[str, str] = {
    "norse": "norse",
    "lava_loihi2": "lava",
    "speck": "sinabs",
    "xylo": "rockpool",
    "spinnaker2": "spinnaker2",
}

#: Backend name -> a minimal capability the SDK must expose. Any one of the
#: named public attributes (or submodules) is enough to count the simulator as
#: present; an installed but unrecognised distribution is reported unavailable
#: rather than trusted, so a shim or a partial install never reads as ready.
BACKEND_CAPABILITIES: Mapping[str, Tuple[str, ...]] = {
    "speck": ("Network", "layers"),
    "xylo": ("XyloSim", "devices", "TimedModuleWrapper"),
    "spinnaker2": ("Spinnaker2", "SNN", "sim"),
}
#: Environment variable that opts a Lava run onto attached hardware.
LAVA_DEVICE_ENV = "SPIKEFORGE_LAVA_DEVICE"
#: Legacy environment variable kept working as a fallback.
LAVA_DEVICE_LEGACY_ENV = "SNN_LAVA_DEVICE"


def module_available(name: str) -> bool:
    """Return True when ``name`` can be imported in this environment."""
    try:
        import_module(name)
    except ImportError:
        return False
    return True


def module_with_capability(name: str, attrs: Sequence[str]) -> bool:
    """Return True when ``name`` imports and exposes one capability attr.

    A plain import is not enough to trust a simulator: a shim, a partial
    install, or a package that only re-exports a name would import cleanly but
    cannot run. Requiring one of the SDK's public attributes keeps the
    availability flag honest without pinning a single upstream spelling.
    """
    if not module_available(name):
        return False
    module = import_module(name)
    return any(getattr(module, attr, None) is not None for attr in attrs)


def backend_available(name: str) -> bool:
    """Return True when ``name``'s SDK imports and meets its capability check.

    Backends without a declared capability fall back to a plain import, which
    is how ``norse`` and ``lava`` have always been probed.
    """
    module = BACKEND_MODULES.get(name)
    if module is None:
        return False
    capabilities = BACKEND_CAPABILITIES.get(name)
    if not capabilities:
        return module_available(module)
    return module_with_capability(module, capabilities)


def version(name: str) -> Optional[str]:
    """Return ``name``'s version string, or ``None`` when absent."""
    try:
        module = import_module(name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return str(value) if value else None


def require(module_name: str, extra: str) -> Any:
    """Import ``module_name`` or raise a named unavailable error."""
    try:
        return import_module(module_name)
    except ImportError as exc:
        raise BackendUnavailableError(extra, extra) from exc


def norse_lif(
    p: float, v_leak: float, v_threshold: float, v_reset: Optional[float]
) -> Any:
    """Return a Norse ``LIF`` neuron with leak ``p`` and its thresholds."""
    norse = require("norse", "norse")
    return norse.LIF(
        p=p, v_leak=v_leak, v_threshold=v_threshold, v_reset=v_reset
    )


def norse_li(p: float, v_leak: float) -> Any:
    """Return a Norse ``LI`` integrator with leak ``p`` and leak potential."""
    norse = require("norse", "norse")
    return norse.LI(p=p, v_leak=v_leak)


def lava_device_present() -> bool:
    """Return True when a Lava device run has been explicitly opted into.

    A physical Loihi 2 cannot be probed safely from a pure import, so the
    default is the CPU emulator and the caller names that path in its report
    rather than claiming a measured device result.
    """
    return bool(
        os.environ.get(LAVA_DEVICE_ENV)
        or os.environ.get(LAVA_DEVICE_LEGACY_ENV)
    )


def _lava_attr(module_name: str, attr: str) -> Any:
    """Return ``attr`` from a Lava submodule or raise a named error."""
    module = require(module_name, "lava")
    value = getattr(module, attr, None)
    if value is None:
        reason = f"installed lava is missing {module_name}.{attr}"
        raise BackendUnavailableError("lava_loihi2", reason)
    return value


def _lava_run_config(device: bool) -> Any:
    """Return the Lava run config for the device or emulator path."""
    name = "Loihi2HwCfg" if device else "Loihi2SimCfg"
    return _lava_attr("lava.magma.core.run_configs", name)()


def _lava_lif_layer(cls: Any, params: Dict[str, Any], size: int) -> Any:
    """Build a Lava LIF layer matching the reference LIF recurrence."""
    return cls(
        shape=(size,),
        du=1,
        dv=params["decay"],
        vth=params["v_threshold"],
    )


def _lava_dense_layer(cls: Any, params: Dict[str, Any]) -> Any:
    """Build a Lava dense synapse from a linear node's weight matrix."""
    return cls(weights=np.asarray(params["weight"], dtype=np.float32))


def _connect_lava(layers: Sequence[Any]) -> None:
    """Chain Lava layers head-to-tail on their spike ports."""
    for source, target in zip(layers, layers[1:]):
        source.s_out.connect(target.a_in)


def _lava_monitor(
    layers: Sequence[Any], monitor: Any, steps: int
) -> Any:
    """Probe the final layer's spike output and return the monitor."""
    monitor.probe(layers[-1].s_out, num_steps=steps)
    return monitor


def _lava_build(
    program: Dict[str, Any], by_name: Dict[str, Any]
) -> List[Any]:
    """Build and connect the Lava layer chain of a lowered program."""
    dense = by_name["dense"]
    lif = by_name["lif"]
    layers: List[Any] = []
    for layer in program["layers"]:
        if layer["kind"] in ("LIF", "LI"):
            layers.append(_lava_lif_layer(lif, layer["params"], layer["size"]))
        else:
            layers.append(_lava_dense_layer(dense, layer["params"]))
    _connect_lava(layers)
    return layers


def lava_run(
    program: Dict[str, Any], spikes: Any, device: bool
) -> Dict[str, Any]:
    """Execute a lowered linear program on Lava and return its traces.

    The layer graph is built from ``lava.proc`` processes and run with the
    hardware config when ``device`` is set, else the Loihi 2 CPU emulator.
    The returned mapping names the readout and spike traces so the caller can
    shape a :class:`BackendResult`. Upstream API drift raises here and is
    surfaced as a named backend error rather than a silent wrong answer.
    """
    by_name = {
        "dense": _lava_attr("lava.proc.dense.process", "Dense"),
        "lif": _lava_attr("lava.proc.lif.process", "LIF"),
    }
    monitor_cls = _lava_attr("lava.proc.monitor.process", "Monitor")
    steps_cls = _lava_attr("lava.magma.core.run_conditions", "RunSteps")
    layers = _lava_build(program, by_name)
    monitor = _lava_monitor(layers, monitor_cls(), int(spikes.shape[0]))
    layers[0].run(
        condition=steps_cls(num_steps=int(spikes.shape[0])),
        run_cfg=_lava_run_config(device),
    )
    return {
        "readout": np.asarray(monitor.get_data()),
        "path": "loihi2_device" if device else "loihi2_emulator",
    }


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
        value = spikes[index]
        for position, layer in enumerate(program["layers"]):
            if layer["kind"] in NEURON_KINDS:
                value, states[position] = _neuron_step(
                    neurons[position], value, states.get(position)
                )
            else:
                value = _weighted(layer["params"]["weight"], value)
        total = value if total is None else total + value
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

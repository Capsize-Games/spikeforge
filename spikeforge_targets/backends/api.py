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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from spikeforge_targets.backends.errors import BackendUnavailableError

#: Backend name -> the module whose presence makes it available.
BACKEND_MODULES: Dict[str, str] = {"norse": "norse", "lava_loihi2": "lava"}
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

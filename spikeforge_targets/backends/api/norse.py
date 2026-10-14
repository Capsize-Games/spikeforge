"""Norse neuron factories behind the isolated SDK probe.

Split out of the ``api`` package root so each vendor's runner lives in its
own file. Upstream API churn stays contained here.
"""

from typing import Any, Optional

from spikeforge_targets.backends.api.probes import require


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

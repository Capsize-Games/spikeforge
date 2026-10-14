"""Export a NIR graph as a summary with quantized weights.

:func:`export_quantized_summary` is the ``graph_summary`` shape with every
floating ``weight`` replaced by the per-tensor envelope the C runtime
loader accepts: ``{data, dtype, scale, zero_point}``, decoded as
``(code - zero_point) * scale``. Codes come from the same
:mod:`spikeforge_targets.quantize_schemes` functions the drift simulation
dequantizes, so the C readout matches the simulated-quantization drift run
by construction rather than by a second implementation.

Only ``weight`` tensors are enveloped, exactly the tensors the drift
simulation quantizes; biases and every other parameter stay the plain
``graph_summary`` values. The ``none`` scheme returns the float summary
itself, so it is trivially byte-comparable to the existing path.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from spikeforge.compression.grid import Grid
from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.errors import QuantizationError
from spikeforge_targets.quantize_schemes import (
    int8_codes,
    node_weight,
    uint8_codes,
)

#: No quantization: the float summary is returned unchanged.
SCHEME_NONE = "none"
#: Symmetric per-tensor int8 weights.
SCHEME_WEIGHT_INT8 = "weight_int8"
#: Asymmetric per-tensor uint8 weights.
SCHEME_WEIGHT_UINT8 = "weight_uint8"

#: One coder: integer codes plus their grid for a weight tensor.
_Coder = Callable[[np.ndarray], Tuple[np.ndarray, Grid]]
#: The coder and envelope dtype tag behind each quantized scheme name.
_CODERS: Dict[str, Tuple[_Coder, str]] = {
    SCHEME_WEIGHT_INT8: (int8_codes, "int8"),
    SCHEME_WEIGHT_UINT8: (uint8_codes, "uint8"),
}


def _weight_envelope(
    node: Any, coder: _Coder, dtype: str
) -> Optional[Dict[str, Any]]:
    """Return the quantized envelope for ``node``'s weight, if it has one.

    Non-finite weights are refused: no finite grid can represent them,
    and the C loader only accepts finite scales.
    """
    array = node_weight(node)
    if array is None:
        return None
    if not bool(np.isfinite(array).all()):
        raise QuantizationError(
            f"weight of node {type(node).__name__!r} is not finite"
        )
    codes, grid = coder(array)
    return {
        "data": codes.tolist(),
        "dtype": dtype,
        "scale": float(grid.scale),
        "zero_point": int(grid.zero_point),
    }


def _quantized(graph: Any, scheme: str) -> Dict[str, Any]:
    """Return ``graph``'s summary with every weight enveloped."""
    coder, dtype = _CODERS[scheme]
    summary = graph_summary(graph)
    for record in summary["nodes"]:
        envelope = _weight_envelope(
            graph.nodes[record["name"]], coder, dtype
        )
        if envelope is not None:
            record["params"]["weight"] = envelope
    return summary


def export_quantized_summary(
    graph_or_spec: Any, scheme: str
) -> Dict[str, Any]:
    """Return the ``graph_summary`` shape with quantized ``weight`` tensors.

    ``graph_or_spec`` is a NIR graph or a :class:`TopologySpec` (exported
    without a module, so its placeholder weights quantize as zeros).
    ``scheme`` is ``none`` (the float summary itself), ``weight_int8``,
    or ``weight_uint8``; anything else raises :class:`QuantizationError`.
    """
    if scheme == SCHEME_NONE:
        return graph_summary(graph_or_spec)
    if scheme not in _CODERS:
        raise QuantizationError(
            f"unknown quantization scheme {scheme!r}; expected one of "
            f"{(SCHEME_NONE, SCHEME_WEIGHT_INT8, SCHEME_WEIGHT_UINT8)}"
        )
    if isinstance(graph_or_spec, TopologySpec):
        graph = to_nir(graph_or_spec)
    else:
        graph = graph_or_spec
    return _quantized(graph, scheme)

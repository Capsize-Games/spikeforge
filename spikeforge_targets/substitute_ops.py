"""Pure rewrite rules that execute a target's declared substitutions.

Each rule turns one node the target lacks into a node (or nodes) it declares
support for, returning a :class:`RewriteOutcome` that names the replacement's
entry and output so the executor can rewire the surrounding edges. Rules are
small, pure, and side-effect free; the executor owns graph surgery and the
report.

``IF`` -> ``LIF`` uses a leak-free identity integrator: a very large ``tau``
gives a decay arbitrarily close to one, and the resistance is rescaled by
``1 / (1 - decay)`` so the zero-order-hold input gain still equals the IF
node's ``r``. The residual leak is tiny but nonzero, which is exactly why the
executor measures and reports a post-rewrite drift rather than claiming
exactness.

``AvgPool2d`` -> ``SumPool2d`` + ``Scale`` is exact for non-overlapping,
unpadded windows: the sum pool multiplies the average by the window area and
the scale divides it back out.
"""

from math import log
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from spikeforge.nir_bridge.require import require_node
from spikeforge_targets.rewrite_outcome import RewriteOutcome

#: Leak fraction used by the ``IF`` -> ``LIF`` rule. It is a power of two so
#: the float32 zero-order-hold step reproduces ``decay`` and its complement
#: exactly: the rescaled resistance ``r / IF_LEAK`` then restores the exact
#: unit input gain, leaving only the tiny, reported residual leak.
IF_LEAK = 2.0 ** -20
#: Identity-integrator decay (``beta`` semantics: one means no leak).
IF_DECAY = 1.0 - IF_LEAK
#: Suffix of the ``Scale`` node appended by the average-pool rule.
SCALE_SUFFIX = "__scale"

Rule = Callable[[str, Any], RewriteOutcome]


def _scalar(value: Any) -> float:
    """Return a node parameter as a Python float."""
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _float32(value: float) -> np.ndarray:
    """Return ``value`` as a scalar float32 array."""
    return np.array(value, dtype=np.float32)


def _if_tau() -> float:
    """Return the large membrane constant for the leak-free rule."""
    return -1.0 / log(IF_DECAY)


def if_to_lif(name: str, node: Any) -> RewriteOutcome:
    """Return the ``nir.LIF`` rendering of an ``nir.IF`` node."""
    cls = require_node("LIF", "if")
    reset = None if node.v_reset is None else _float32(_scalar(node.v_reset))
    lif = cls(
        _float32(_if_tau()),
        _float32(_scalar(node.r) / IF_LEAK),
        _float32(0.0),
        _float32(_scalar(node.v_threshold)),
        reset,
    )
    return RewriteOutcome(((name, lif),), (), name, name, "beta=0")


def _pool_node(node: Any, cls: Any) -> Any:
    """Return a fresh pooling node copying ``node``'s window settings."""
    return cls(
        np.array(node.kernel_size), np.array(node.stride),
        np.array(node.padding),
    )


def avgpool_to_sumpool_scale(name: str, node: Any) -> RewriteOutcome:
    """Return a ``SumPool2d`` plus ``Scale`` rendering of ``AvgPool2d``."""
    pool_cls = require_node("SumPool2d", "avgpool2d")
    scale_cls = require_node("Scale", "avgpool2d")
    area = float(np.asarray(node.kernel_size).reshape(-1).prod())
    scale_name = f"{name}{SCALE_SUFFIX}"
    scale = scale_cls(_float32(1.0 / area))
    return RewriteOutcome(
        ((name, _pool_node(node, pool_cls)), (scale_name, scale)),
        ((name, scale_name),),
        name,
        scale_name,
        f"1/{int(area)}",
    )


#: The executable rule for each declared ``(primitive, substitute)`` pair.
REWRITES: Dict[Tuple[str, str], Rule] = {
    ("IF", "LIF"): if_to_lif,
    ("AvgPool2d", "SumPool2d"): avgpool_to_sumpool_scale,
}


def apply_rewrite(
    primitive: str, substitute: str, name: str, node: Any
) -> Optional[RewriteOutcome]:
    """Return the outcome of rewriting ``node``, or ``None`` with no rule."""
    rule = REWRITES.get((primitive, substitute))
    return None if rule is None else rule(name, node)

"""Availability-annotated summaries of the built-in deployment targets.

The CLI, the WebSocket ``targets`` action, and the dashboard all report the
same shape: a target's declared :meth:`TargetSpec.to_dict` plus its live
availability and the size of its supported-primitive set. Keeping the two
additive keys here means one definition serves every surface.
"""

from typing import Any, Dict, List

from spikeforge_targets.registry import (
    available,
    get_target,
    target_names,
)


def target_summary(name: str) -> Dict[str, Any]:
    """Return a target's declared spec plus availability and support count.

    ``available`` and ``supported_count`` are additive on top of
    ``TargetSpec.to_dict``, so existing consumers of the spec keep working
    while a caller can honestly report whether the target is actually usable
    on this machine and how many primitives it declares.
    """
    spec = get_target(name)
    payload = spec.to_dict()
    payload["available"] = available(name)
    payload["supported_count"] = len(spec.supported)
    return payload


def target_summaries() -> List[Dict[str, Any]]:
    """Return one availability-annotated summary per built-in target."""
    return [target_summary(name) for name in target_names()]

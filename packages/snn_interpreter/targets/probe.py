"""Isolated capability probe for the optional target SDKs.

This is the only module in the project that imports a target backend SDK.
Imports happen inside :func:`module_available`, never at module import time,
so a missing package never breaks importing the targets package; callers
instead read the boolean returned by :func:`extra_available`. The reference
target has no extra and is therefore always available.
"""

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Tuple

#: Modules that enable each declared pip extra.
EXTRA_MODULES: Mapping[str, Tuple[str, ...]] = {
    "lava": ("lava",),
    "spinnaker2": ("spinnaker2",),
    "speck": ("sinabs",),
    "xylo": ("rockpool",),
    "norse": ("norse",),
}


def module_available(name: str) -> bool:
    """Return True when ``name`` can be imported in this environment."""
    try:
        import_module(name)
    except ImportError:
        return False
    return True


def extra_available(extra: Optional[str]) -> bool:
    """Return True when every module for ``extra`` is importable.

    A ``None`` extra is the always-available reference target. An extra that
    is not declared here is reported unavailable rather than guessed at.
    """
    if extra is None:
        return True
    modules = EXTRA_MODULES.get(extra)
    if not modules:
        return False
    return all(module_available(name) for name in modules)


def report() -> Dict[str, Any]:
    """Return a JSON-able availability report for every declared extra."""
    return {
        extra: extra_available(extra) for extra in sorted(EXTRA_MODULES)
    }

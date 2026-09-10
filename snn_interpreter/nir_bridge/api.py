"""Isolated capability probe for the optional ``nir``/``nirtorch`` packages.

This is the only module in the project that imports ``nir`` or ``nirtorch``.
Imports are defensive so a missing dependency never raises at import time;
callers instead read the report returned by :func:`capability`.
"""

from importlib import import_module
from typing import Any, Dict, List, Optional

# Keys guaranteed to be present in the dict returned by ``capability``.
REPORT_KEYS = (
    "nir_available",
    "nirtorch_available",
    "nir_version",
    "nirtorch_version",
    "node_primitives",
)


def _safe_import(name: str) -> Optional[Any]:
    """Import ``name``, returning ``None`` when it is unavailable."""
    try:
        return import_module(name)
    except ImportError:
        return None


def _version_of(module: Optional[Any]) -> Optional[str]:
    """Return the module's version string when one is exposed."""
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _public_names(module: Optional[Any]) -> List[str]:
    """Return the sorted public attribute names exposed by ``module``."""
    if module is None:
        return []
    return sorted(name for name in dir(module) if not name.startswith("_"))


def _is_node_class(candidate: Any, base: type) -> bool:
    """Return True when ``candidate`` is a concrete ``base`` subclass."""
    if not isinstance(candidate, type) or candidate is base:
        return False
    return issubclass(candidate, base)


def _node_primitives(module: Optional[Any]) -> List[str]:
    """Return public ``NIRNode`` subclasses exposed by the ``nir`` module."""
    base = getattr(module, "NIRNode", None)
    if not isinstance(base, type):
        return []
    return [
        name
        for name in _public_names(module)
        if _is_node_class(getattr(module, name), base)
    ]


def capability() -> Dict[str, Any]:
    """Return a JSON-able report of the NIR dependency surface."""
    nir_module = _safe_import("nir")
    nirtorch_module = _safe_import("nirtorch")
    return {
        "nir_available": nir_module is not None,
        "nirtorch_available": nirtorch_module is not None,
        "nir_version": _version_of(nir_module),
        "nirtorch_version": _version_of(nirtorch_module),
        "node_primitives": _node_primitives(nir_module),
    }


def node_class(name: str) -> Optional[type]:
    """Return the ``nir`` class called ``name``, or ``None`` if absent.

    Callers use this instead of importing ``nir`` so every direct dependency
    on the package stays inside this module.
    """
    nir_module = _safe_import("nir")
    if nir_module is None:
        return None
    candidate = getattr(nir_module, name, None)
    return candidate if isinstance(candidate, type) else None

"""Isolated capability probe for the optional ``tonic`` package.

This is the only module in the project that imports ``tonic``. Imports are
defensive so a missing dependency never raises at import time; callers
instead read the report returned by :func:`capability` and reach the
package's classes through :func:`dataset_class`. Installing the ``events``
extra (``pip install -e ".[events]"``) adds ``tonic``.
"""

from importlib import import_module
from typing import Any, Dict, List, Optional

# Keys guaranteed to be present in the dict returned by ``capability``.
REPORT_KEYS = (
    "tonic_available",
    "tonic_version",
    "dataset_classes",
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


def available() -> bool:
    """Return True when ``tonic`` can be imported in this environment."""
    return _safe_import("tonic") is not None


def dataset_classes() -> List[str]:
    """Return the public dataset class names (empty when absent)."""
    datasets = _safe_import("tonic.datasets")
    return [
        name
        for name in _public_names(datasets)
        if isinstance(getattr(datasets, name), type)
    ]


def dataset_class(name: str) -> Optional[type]:
    """Return the ``tonic.datasets`` class called ``name``, or ``None``.

    Callers use this instead of importing ``tonic`` so every direct
    dependency on the package stays inside this module.
    """
    datasets = _safe_import("tonic.datasets")
    if datasets is None:
        return None
    candidate = getattr(datasets, name, None)
    return candidate if isinstance(candidate, type) else None


def capability() -> Dict[str, Any]:
    """Return a JSON-able report of the tonic dependency surface."""
    module = _safe_import("tonic")
    return {
        "tonic_available": module is not None,
        "tonic_version": _version_of(module),
        "dataset_classes": dataset_classes(),
    }

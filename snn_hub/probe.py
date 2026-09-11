"""Isolated capability probe for the optional ``huggingface_hub`` package.

This is the only module in the project that imports ``huggingface_hub``.
Imports are defensive so a missing dependency never raises at import time;
callers read the report from :func:`capability` and reach the package through
:func:`module`. Installing the ``hub`` extra adds it.
"""

from importlib import import_module
from typing import Any, Dict, Optional

# Keys guaranteed to be present in the dict returned by ``capability``.
REPORT_KEYS = (
    "huggingface_hub_available",
    "huggingface_hub_version",
)


def _safe_import(name: str) -> Optional[Any]:
    """Import ``name``, returning ``None`` when it is unavailable."""
    try:
        return import_module(name)
    except ImportError:
        return None


def module() -> Optional[Any]:
    """Return the imported ``huggingface_hub`` module, or ``None``.

    Callers use this instead of importing the package so every direct
    dependency on it stays inside this module.
    """
    return _safe_import("huggingface_hub")


def available() -> bool:
    """Return True when ``huggingface_hub`` can be imported."""
    return module() is not None


def version() -> Optional[str]:
    """Return the installed ``huggingface_hub`` version, or ``None``."""
    found = module()
    value = getattr(found, "__version__", None) if found else None
    return str(value) if value else None


def capability() -> Dict[str, Any]:
    """Return a JSON-able report of the hub dependency surface."""
    return {
        "huggingface_hub_available": available(),
        "huggingface_hub_version": version(),
    }

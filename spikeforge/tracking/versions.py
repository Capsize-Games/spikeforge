"""Resolve the library versions recorded in a reproducibility manifest.

``nir``/``nirtorch`` are read through the isolated capability probe in
:mod:`spikeforge.nir_bridge.api`, so no module outside that probe imports
the optional packages directly.
"""

import platform
from typing import Any, Dict, Optional

import snntorch
import torch


def _version_of(module: Any) -> Optional[str]:
    """Return a module's version string when one is exposed."""
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _nir_versions() -> Dict[str, Optional[str]]:
    """Return the probed ``nir``/``nirtorch`` versions, imported lazily."""
    from spikeforge.nir_bridge.api import capability

    report = capability()
    return {
        "nir": report["nir_version"],
        "nirtorch": report["nirtorch_version"],
    }


def library_versions() -> Dict[str, Optional[str]]:
    """Return the interpreter and tracked library versions, JSON-able.

    A library that is not installed is reported as ``None`` rather than
    raising, so a manifest can always be written.
    """
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": _version_of(torch),
        "snntorch": _version_of(snntorch),
    }
    versions.update(_nir_versions())
    return versions

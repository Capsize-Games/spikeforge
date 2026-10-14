"""Availability probes for the optional backend SDKs.

Split out of the ``api`` package root so each vendor's runner lives in its
own file. Callers read :func:`backend_available` first and degrade honestly;
a missing package never breaks importing the backends.
"""

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from spikeforge_targets.backends.errors import BackendUnavailableError

#: Backend name -> the module whose presence makes it available.
BACKEND_MODULES: Dict[str, str] = {
    "norse": "norse",
    "lava_loihi2": "lava",
    "speck": "sinabs",
    "xylo": "rockpool",
    "spinnaker2": "spinnaker2",
}

#: Backend name -> a minimal capability the SDK must expose. Any one of the
#: named public attributes (or submodules) is enough to count the simulator as
#: present; an installed but unrecognised distribution is reported unavailable
#: rather than trusted, so a shim or a partial install never reads as ready.
BACKEND_CAPABILITIES: Mapping[str, Tuple[str, ...]] = {
    "speck": ("Network", "layers"),
    "xylo": ("XyloSim", "devices", "TimedModuleWrapper"),
    "spinnaker2": ("Spinnaker2", "SNN", "sim"),
}


def module_available(name: str) -> bool:
    """Return True when ``name`` can be imported in this environment."""
    try:
        import_module(name)
    except ImportError:
        return False
    return True


def module_with_capability(name: str, attrs: Sequence[str]) -> bool:
    """Return True when ``name`` imports and exposes one capability attr.

    A plain import is not enough to trust a simulator: a shim, a partial
    install, or a package that only re-exports a name would import cleanly but
    cannot run. Requiring one of the SDK's public attributes keeps the
    availability flag honest without pinning a single upstream spelling.
    """
    if not module_available(name):
        return False
    module = import_module(name)
    return any(getattr(module, attr, None) is not None for attr in attrs)


def backend_available(name: str) -> bool:
    """Return True when ``name``'s SDK imports and meets its capability check.

    Backends without a declared capability fall back to a plain import, which
    is how ``norse`` and ``lava`` have always been probed.
    """
    module = BACKEND_MODULES.get(name)
    if module is None:
        return False
    capabilities = BACKEND_CAPABILITIES.get(name)
    if not capabilities:
        return module_available(module)
    return module_with_capability(module, capabilities)


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

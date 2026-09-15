"""Headless HTTP inference service for a ``.spkf`` deployment bundle.

This distribution owns the FastAPI/uvicorn dependency: the ``spikeforge`` core
must stay web-framework free, so the ASGI surface and the pure serving core
live here and only here.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from spikeforge_serve.app import SERVE_VERSION, create_app
from spikeforge_serve.service import DEFAULT_SESSION, ServingService

try:
    #: Read from installed metadata so it cannot drift from the wheel.
    __version__ = _distribution_version("spikeforge-serve")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "DEFAULT_SESSION",
    "SERVE_VERSION",
    "ServingService",
    "create_app",
]

"""Headless HTTP inference service for a ``.spkf`` deployment bundle.

This distribution owns the FastAPI/uvicorn dependency: the ``spikeforge`` core
must stay web-framework free, so the ASGI surface and the pure serving core
live here and only here.
"""

from spikeforge_serve.app import SERVE_VERSION, create_app
from spikeforge_serve.service import DEFAULT_SESSION, ServingService

__all__ = [
    "DEFAULT_SESSION",
    "SERVE_VERSION",
    "ServingService",
    "create_app",
]

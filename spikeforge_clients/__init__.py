"""Python client SDK for the ``spikeforge-serve`` inference service.

The distribution owns no web framework: :class:`HttpTransport` uses the
standard library for requests and the optional ``websockets`` package only for
streaming, so importing ``spikeforge_clients`` stays cheap. The core
``spikeforge`` package is never imported -- the wire contract is the only
coupling, and it is validated against the schemas under ``protocol/serve/``.
"""

from spikeforge_clients.bundle import BundleInfo
from spikeforge_clients.client import ServeClient
from spikeforge_clients.config import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    ClientConfig,
)
from spikeforge_clients.errors import (
    ClientError,
    ServiceError,
    StreamError,
    TransportError,
)
from spikeforge_clients.health import Health, Readiness
from spikeforge_clients.inprocess import ASGITransport
from spikeforge_clients.prediction import (
    PredictionResult,
    PredictResponse,
    ResetResult,
    TensorValue,
)
from spikeforge_clients.stream import StreamEvent

__all__ = [
    "ASGITransport",
    "BundleInfo",
    "ClientConfig",
    "ClientError",
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT",
    "Health",
    "PredictResponse",
    "PredictionResult",
    "Readiness",
    "ResetResult",
    "ServeClient",
    "ServiceError",
    "StreamError",
    "StreamEvent",
    "TensorValue",
    "TransportError",
]

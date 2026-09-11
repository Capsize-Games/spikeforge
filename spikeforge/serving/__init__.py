"""Stateful inference runtime and the portable deployment bundle.

Importing this package pulls in no optional SDK: it is pure torch and stdlib,
so a headless core install can build and serve a model without the server
stack, the hub, or any backend SDK.
"""

from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
    ServingError,
    StateError,
)
from spikeforge.serving.prediction import Prediction
from spikeforge.serving.session import InferenceSession
from spikeforge.serving.state_tree import StateTree

__all__ = [
    "BundleCompatibilityError",
    "BundleError",
    "BundleFormatError",
    "BundleIntegrityError",
    "BundleNotFoundError",
    "DeploymentBundle",
    "InferenceSession",
    "Prediction",
    "ServingError",
    "StateError",
    "StateTree",
    "build",
]

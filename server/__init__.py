"""FastAPI server exposing SNN encoding experiments over WebSocket."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

try:
    #: Read from installed metadata so it cannot drift from the wheel.
    __version__ = _distribution_version("spikeforge-server")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"


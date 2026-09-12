"""Connection settings for the ``spikeforge-clients`` SDK."""

from dataclasses import dataclass
from typing import Optional

#: Default base URL of a locally-run ``spikeforge-serve``.
DEFAULT_BASE_URL = "http://127.0.0.1:8899"

#: Default per-request timeout, in seconds.
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class ClientConfig:
    """Where the service lives, how long to wait, and how to authenticate."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT
    token: Optional[str] = None
    session_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize the base URL so paths concatenate cleanly."""
        if not self.base_url:
            raise ValueError("base_url must be a non-empty string")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

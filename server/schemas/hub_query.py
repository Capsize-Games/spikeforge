"""Additive inbound query fields for the model-hub WebSocket actions."""

from typing import Optional

from pydantic import BaseModel


class HubQuery(BaseModel):
    """Filter and locator fields carried by the ``hub_*`` client actions.

    A dedicated model keeps the existing ``query`` (a :class:`ModelQuery`)
    untouched, so the protocol stays additive for current clients.
    """

    query: str = ""
    limit: int = 20
    framework: Optional[str] = None
    kind: Optional[str] = None
    available: Optional[bool] = None
    topology: Optional[str] = None

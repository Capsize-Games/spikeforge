"""Inbound filters for the model-registry WebSocket actions."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ModelQuery(BaseModel):
    """Search filters plus the pair of names a metadata diff operates on."""

    dataset: Optional[str] = None
    topology: Optional[str] = None
    coding: Optional[str] = None
    device: Optional[str] = None
    min_accuracy: Optional[float] = None
    name: Optional[str] = None
    names: List[str] = Field(default_factory=list)

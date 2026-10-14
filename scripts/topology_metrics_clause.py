"""One trigger clause and its evaluated state."""

from typing import NamedTuple, Optional


class Clause(NamedTuple):
    """One trigger clause and its evaluated state (None == unavailable)."""

    text: str
    state: Optional[bool]


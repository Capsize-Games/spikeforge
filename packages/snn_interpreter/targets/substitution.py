"""A node whose primitive a target replaces with another primitive."""

from typing import Dict, NamedTuple


class Substitution(NamedTuple):
    """A single ``primitive -> substitute`` classification for a node."""

    name: str
    primitive: str
    substitute: str

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serialisable record for this substitution."""
        return {
            "name": self.name,
            "primitive": self.primitive,
            "substitute": self.substitute,
        }

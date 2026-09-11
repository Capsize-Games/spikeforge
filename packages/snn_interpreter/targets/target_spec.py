"""Declaration of a deployment or simulation target."""

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Literal, Mapping, Optional

#: Whether a target is the in-process reference, a simulator, or hardware.
TargetKind = Literal["reference", "simulator", "hardware"]


@dataclass(frozen=True)
class TargetSpec:
    """A deployable target and the NIR primitives it declares support.

    ``supported`` names the primitive classes the target runs natively.
    ``substitutions`` maps a primitive the target lacks onto the primitive it
    uses instead, so such a node is neither supported nor dropped.
    ``constraints`` is a small JSON-serialisable mapping of dtype, timestep,
    and quantization limits. ``extra`` names the pip extra that would install
    the enabling SDK; the reference target has ``None`` and is always
    available.
    """

    name: str
    kind: TargetKind
    description: str
    extra: Optional[str]
    supported: FrozenSet[str]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    substitutions: Mapping[str, str] = field(default_factory=dict)

    def supports(self, primitive: str) -> bool:
        """Return True when ``primitive`` is in the declared supported set."""
        return primitive in self.supported

    def substitute(self, primitive: str) -> Optional[str]:
        """Return the primitive used in place of ``primitive``, if any."""
        return self.substitutions.get(primitive)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable description of the target."""
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "extra": self.extra,
            "supported": sorted(self.supported),
            "constraints": dict(self.constraints),
            "substitutions": dict(self.substitutions),
        }

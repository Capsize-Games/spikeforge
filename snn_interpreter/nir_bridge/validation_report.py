"""A JSON-serialisable report comparing a snnTorch and a NIR trajectory."""

from typing import Any, Mapping, Optional, Sequence


class ValidationReport(dict):
    """Validation result as a plain, JSON-serialisable mapping.

    Keys: ``within_tolerance`` (bool), ``steps`` (int), ``layers`` (mapping
    of stage name to per-quantity metrics and a per-layer pass flag),
    ``readout`` (metrics), ``worst`` (a descriptor naming the worst
    layer/quantity/metric, or ``None``) and ``notes`` (list of strings).
    Subclassing ``dict`` is deliberate: ``json.dumps`` serialises the report
    directly, which is what the streaming server needs.
    """

    def __init__(
        self,
        within_tolerance: bool,
        steps: int,
        layers: Mapping[str, Any],
        readout: Mapping[str, float],
        worst: Optional[Mapping[str, Any]],
        notes: Sequence[str] = (),
    ) -> None:
        """Build the report mapping from its parts."""
        super().__init__(
            within_tolerance=bool(within_tolerance),
            steps=int(steps),
            layers=dict(layers),
            readout=dict(readout),
            worst=dict(worst) if worst is not None else None,
            notes=list(notes),
        )

    @property
    def within_tolerance(self) -> bool:
        """Return whether every compared quantity stayed within tolerance."""
        return bool(self["within_tolerance"])

    @property
    def worst(self) -> Optional[Mapping[str, Any]]:
        """Return the descriptor of the worst offending metric."""
        return self["worst"]

    @property
    def layers(self) -> Mapping[str, Any]:
        """Return the per-stage metric mapping."""
        return self["layers"]

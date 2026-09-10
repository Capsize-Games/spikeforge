"""Resolve a required ``nir`` primitive or fail with a typed error."""

from snn_interpreter.nir_bridge import api
from snn_interpreter.nir_bridge.errors import UnsupportedStageError


def require_node(primitive: str, kind: str) -> type:
    """Return the ``nir`` class ``primitive`` for the stage kind ``kind``.

    Raises :class:`UnsupportedStageError` naming ``kind`` when the installed
    ``nir`` does not expose the primitive, so an unmappable stage is never
    silently dropped.
    """
    cls = api.node_class(primitive)
    if cls is None:
        detail = f"installed nir is missing the {primitive!r} primitive"
        raise UnsupportedStageError(kind, detail)
    return cls

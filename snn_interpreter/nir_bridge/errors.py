"""Typed errors raised while mapping or interpreting NIR."""


class UnsupportedNodeError(Exception):
    """Raised when the reference interpreter meets an unimplemented node.

    The offending ``kind`` (the node class name) and ``name`` (its graph key)
    are stored as attributes so callers can react without parsing the text.
    """

    def __init__(self, kind: str, name: str) -> None:
        """Record ``kind`` and ``name`` and build a clear message."""
        message = (
            f"reference interpreter has no implementation for node "
            f"{name!r} of kind {kind!r}"
        )
        super().__init__(message)
        self.kind: str = kind
        self.name: str = name


class UnsupportedStageError(Exception):
    """Raised when a stage kind has no mapping in the installed ``nir``.

    The offending ``kind`` is stored as an attribute so callers can react
    without having to parse the message.
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        """Record ``kind`` and fold ``detail`` into the message."""
        message = f"stage kind {kind!r} cannot be mapped to NIR"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.kind: str = kind

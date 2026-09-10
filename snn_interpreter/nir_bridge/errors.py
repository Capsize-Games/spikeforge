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


class GraphFileError(Exception):
    """Base error for reading or writing a persisted NIR graph.

    The offending ``path`` and a human-readable ``detail`` are stored as
    attributes so callers can react without parsing the message.
    """

    def __init__(self, path: str, detail: str) -> None:
        """Record ``path`` and ``detail`` and build a clear message."""
        super().__init__(f"{detail} (graph file: {path})")
        self.path: str = path
        self.detail: str = detail


class GraphNotFoundError(GraphFileError):
    """Raised when a graph file does not exist at the requested path."""

    def __init__(self, path: str) -> None:
        """Record the missing ``path``."""
        super().__init__(path, "graph file not found")


class MalformedGraphError(GraphFileError):
    """Raised when a graph file cannot be parsed or rebuilt faithfully."""

    def __init__(self, path: str, detail: str) -> None:
        """Record ``path`` and the reason the content is unusable."""
        super().__init__(path, f"malformed graph file: {detail}")


class UnknownNodeKindError(GraphFileError):
    """Raised when a stored node kind cannot be rebuilt by ``nir``.

    The offending ``kind`` (the node type name) and ``name`` (its graph key)
    are stored as attributes so the failure never hides which node it was.
    """

    def __init__(self, path: str, kind: str, name: str) -> None:
        """Record ``path``, ``kind`` and ``name``."""
        detail = f"unknown node kind {kind!r} for node {name!r}"
        super().__init__(path, detail)
        self.kind: str = kind
        self.name: str = name

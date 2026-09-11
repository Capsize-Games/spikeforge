"""Typed errors raised while compiling or running a backend."""


class BackendUnavailableError(Exception):
    """Raised when a backend's SDK cannot be imported.

    The backend ``name`` and the pip ``extra`` that would install its SDK are
    stored as attributes so a caller can report an honest, named reason
    instead of parsing the message.
    """

    def __init__(self, name: str, extra: str) -> None:
        """Record ``name`` and ``extra`` and build a clear message."""
        message = (
            f"backend {name!r} is unavailable: install the {extra!r} extra"
        )
        super().__init__(message)
        self.name: str = name
        self.extra: str = extra

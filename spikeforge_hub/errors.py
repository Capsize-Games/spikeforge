"""Typed errors raised across the model-hub package.

Grouping them here mirrors :mod:`spikeforge.nir_bridge.errors` and
:mod:`spikeforge.data.event_errors`: every failure a caller might want to
react to is a named type, so an unavailable capability or a failed
verification is reported rather than guessed at.
"""


class HubError(Exception):
    """Base class for every typed hub failure."""


class HubCatalogError(HubError):
    """Raised when a catalog entry cannot be validated.

    The offending ``entry_id`` (or its position when it has no id) and a human
    ``detail`` are stored as attributes so the loader can report every problem
    instead of failing on the first one.
    """

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record ``entry_id`` and ``detail`` and build a clear message."""
        super().__init__(f"invalid catalog entry {entry_id!r}: {detail}")
        self.entry_id: str = entry_id
        self.detail: str = detail


class HubExtraMissingError(HubError):
    """Raised when the ``hub`` extra (``huggingface_hub``) is absent.

    The message names the extra and the install command so a caller never has
    to guess why live hub access is unavailable.
    """

    def __init__(self, subject: str = "") -> None:
        """Name the ``subject`` that needed the extra and how to install it."""
        detail = f" for {subject}" if subject else ""
        super().__init__(
            "huggingface_hub is not installed, so live hub access is"
            f" unavailable{detail}; install the `hub` extra with "
            'pip install -e ".[hub]"'
        )
        self.subject: str = subject


class HubDownloadError(HubError):
    """Raised when an entry cannot be fetched into the offline cache."""

    def __init__(self, subject: str, detail: str) -> None:
        """Record ``subject`` and ``detail`` and build a clear message."""
        super().__init__(f"hub download failed for {subject!r}: {detail}")
        self.subject: str = subject
        self.detail: str = detail


class HubDownloadCancelledError(HubError):
    """Raised when a hub download is cancelled by the user.

    Mirrors :class:`server.download_errors.DownloadCancelledError` without
    importing the server package, keeping the hub core self-contained.
    """

    def __init__(self, entry_id: str = "") -> None:
        """Name the cancelled ``entry_id`` when one is known."""
        detail = f" for {entry_id!r}" if entry_id else ""
        super().__init__(f"hub download cancelled{detail}")
        self.entry_id: str = entry_id


class HubArtifactError(HubError):
    """Raised when a downloaded artifact cannot be identified or read.

    The offending ``subject`` (an entry id or a path) and a human ``detail``
    are stored as attributes so an unreadable artifact is reported by name
    instead of being silently loaded wrong.
    """

    def __init__(self, subject: str, detail: str) -> None:
        """Record ``subject`` and ``detail`` and build a clear message."""
        super().__init__(f"hub artifact unusable for {subject!r}: {detail}")
        self.subject: str = subject
        self.detail: str = detail


class HubImportError(HubError):
    """Raised when a compatible artifact cannot be promoted into the store.

    The offending ``entry_id`` and a human ``detail`` are stored as attributes
    so a failed promotion names the artifact rather than a bare traceback.
    """

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record ``entry_id`` and ``detail`` and build a clear message."""
        super().__init__(f"hub import failed for {entry_id!r}: {detail}")
        self.entry_id: str = entry_id
        self.detail: str = detail

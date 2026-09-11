"""Typed errors raised while building, loading, or running a deployment.

Every error carries its attributes (``path``, ``detail``, and where relevant
the offending ``entry``) so a caller can react honestly — report the install
hint, name the tampered file, or surface a compatibility mismatch — without
parsing the message text. The hierarchy mirrors
:mod:`spikeforge.nir_bridge.errors` and
:mod:`spikeforge_targets.backends.errors`.
"""


class ServingError(Exception):
    """Base error for the stateful serving runtime."""


class BundleError(ServingError):
    """Base error for building or reading a deployment bundle.

    The offending bundle ``path`` and a human-readable ``detail`` are stored
    as attributes.
    """

    def __init__(self, path: str, detail: str) -> None:
        """Record ``path`` and ``detail`` and build a clear message."""
        super().__init__(f"{detail} (bundle: {path})")
        self.path: str = path
        self.detail: str = detail


class BundleNotFoundError(BundleError):
    """Raised when no readable bundle exists at the requested path."""

    def __init__(self, path: str) -> None:
        """Record the missing ``path``."""
        super().__init__(path, "bundle file not found")


class BundleFormatError(BundleError):
    """Raised when a bundle is unreadable, incomplete, or inconsistent.

    Covers a wrong or absent format marker, an unsupported ``version``, a
    malformed ``manifest.json``, a missing required entry, and weights that do
    not load into the module the manifest's spec describes.
    """

    def __init__(self, path: str, detail: str) -> None:
        """Record the malformed ``detail``."""
        super().__init__(path, f"malformed bundle: {detail}")


class BundleIntegrityError(BundleError):
    """Raised when a bundle's contents do not match its checksums.

    The offending ``entry`` is stored as an attribute so a tampered archive
    names the exact file.
    """

    def __init__(
        self, path: str, entry: str, detail: str = "checksum mismatch"
    ) -> None:
        """Record ``entry`` and the integrity ``detail``."""
        super().__init__(path, f"integrity failure for {entry!r}: {detail}")
        self.entry: str = entry


class BundleCompatibilityError(BundleError):
    """Raised when a bundle was built against an incompatible runtime."""

    def __init__(self, path: str, detail: str) -> None:
        """Record the compatibility ``detail``."""
        super().__init__(path, f"incompatible bundle: {detail}")


class StateError(ServingError):
    """Raised when a carried state does not match the session's spec."""

    def __init__(self, detail: str) -> None:
        """Record ``detail`` and build a clear message."""
        super().__init__(f"invalid inference state: {detail}")
        self.detail: str = detail

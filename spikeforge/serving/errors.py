"""Typed errors raised while building, loading, or running a deployment.

Every error carries its attributes (``path``, ``detail``, and where relevant
the offending ``entry``) so a caller can react honestly — report the install
hint, name the tampered file, or surface a compatibility mismatch — without
parsing the message text. The hierarchy mirrors
:mod:`spikeforge.nir_bridge.errors` and
:mod:`spikeforge_targets.backends.errors`.
"""

from typing import Any, Sequence, Tuple


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


def _refusal_detail(
    kinds: Tuple[str, ...], violations: Sequence[Tuple[str, str, str]]
) -> str:
    """Return the refusal message for ``kinds`` and ``violations``."""
    listed = ", ".join(repr(kind) for kind in kinds)
    if not violations:
        return f"graph uses node kinds the C runtime rejects: {listed}"
    details = "; ".join(
        f"{name!r} ({kind}): {reason}"
        for name, kind, reason in violations
    )
    return f"graph the C runtime rejects (kinds {listed}): {details}"


class CExportError(BundleError):
    """Raised when a bundle cannot be exported as a C-loadable package.

    The offending node ``kinds`` are stored as an attribute so a caller can
    name exactly which topology nodes the C runtime does not implement,
    without parsing the message text. When the gate reports per-node
    ``violations`` -- ``(name, kind, reason)`` triples -- they are stored
    too, and the message names each node alongside its reason.
    """

    def __init__(
        self,
        path: str,
        kinds: Sequence[str],
        violations: Sequence[Tuple[str, str, str]] = (),
    ) -> None:
        """Record ``kinds`` and per-node ``violations`` truthfully."""
        ordered = tuple(sorted(kinds))
        super().__init__(path, _refusal_detail(ordered, violations))
        self.kinds: Tuple[str, ...] = ordered
        self.violations: Tuple[Tuple[str, str, str], ...] = tuple(
            violations
        )


class BackendError(ServingError):
    """Raised when an inference backend is misused or misconfigured.

    Covers an unknown backend name, a backend/option combination the
    session cannot honor (such as a torch ``post_step`` hook on the C
    backend), and use after :meth:`close`. Frames a backend cannot
    consume are plain :class:`ValueError`, like every other frame
    refusal on the serving path. The offending ``detail`` is stored
    as an attribute.
    """

    def __init__(self, detail: str) -> None:
        """Record ``detail`` and build a clear message."""
        super().__init__(f"invalid inference backend use: {detail}")
        self.detail: str = detail


class CRuntimeLibraryError(BackendError):
    """Raised when the ``nir-c-runtime`` shared library cannot load.

    The attempted ``path`` (an explicit path, the ``NIR_C_RUNTIME_LIB``
    environment variable, or ``None`` when neither was given) is stored
    as an attribute, and the message explains how to build the library
    from the ``nir-c-runtime`` repository.
    """

    def __init__(self, path: Any, detail: str) -> None:
        """Record ``path`` and the load-failure ``detail``."""
        super().__init__(f"cannot load the C runtime ({path}): {detail}")
        self.path: Any = path


class CRuntimeExecutionError(BackendError):
    """Raised when a ``nir-c-runtime`` call itself fails.

    The failing C ``operation`` (``parse``, ``load``, ``init``, ``run``)
    is stored as an attribute; the message carries the runtime's own
    text after the operation, so a caller can report what the C side
    rejected without parsing prose.
    """

    def __init__(self, operation: str, detail: str) -> None:
        """Record ``operation`` and the C runtime's ``detail``."""
        super().__init__(
            f"C runtime {operation} failed: {detail}"
        )
        self.operation: str = operation


class EncodeSpecError(ServingError):
    """Raised when an encode spec is malformed or an unsupported version.

    The offending ``detail`` is stored as an attribute so a caller can show
    which parameter or version was refused without parsing the message.
    """

    def __init__(self, detail: str) -> None:
        """Record ``detail`` and build a clear message."""
        super().__init__(f"invalid encode spec: {detail}")
        self.detail: str = detail


class StateError(ServingError):
    """Raised when a carried state does not match the session's spec."""

    def __init__(self, detail: str) -> None:
        """Record ``detail`` and build a clear message."""
        super().__init__(f"invalid inference state: {detail}")
        self.detail: str = detail

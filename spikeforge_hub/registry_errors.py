"""Typed errors raised by the registry-governance layer.

Grouped here (mirroring :mod:`spikeforge_hub.errors` and
:mod:`spikeforge.serving.errors`) so every governance refusal a caller might
want to react to is a named type with structured attributes, rather than a bare
string. A malformed entry, an unsupported schema/version, a lifecycle block, a
missing approver, a bad stage transition, a failed signature, and an integrity
mismatch are all distinct so a CLI can gate on the exact reason.

The hierarchy hangs off :class:`RegistryError`, which is itself a
:class:`~spikeforge_hub.errors.HubError` so an existing ``except HubError``
handler still catches a governance failure.
"""

from typing import Optional

from spikeforge_hub.errors import HubError


class RegistryError(HubError):
    """Base class for every typed registry-governance failure."""


class RegistrySchemaError(RegistryError):
    """Raised when a registry entry omits or malforms a required field.

    The offending ``entry_id`` (or its position when it has no id) and a human
    ``detail`` are stored as attributes so a loader can report every problem
    instead of failing on the first one.
    """

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record ``entry_id`` and ``detail`` and build a clear message."""
        super().__init__(f"invalid registry entry {entry_id!r}: {detail}")
        self.entry_id: str = entry_id
        self.detail: str = detail


class RegistryVersionError(RegistryError):
    """Raised when a registry or entry version is unsupported/malformed.

    The registry document pins ``REGISTRY_SCHEMA_VERSION``; an entry pins a
    concrete dotted ``version``. A mismatch or an unparseable value is refused
    rather than coerced, so an incompatible document never loads silently.
    """

    def __init__(self, subject: str, detail: str) -> None:
        """Record the ``subject`` and the version ``detail``."""
        super().__init__(
            f"unsupported registry version for {subject!r}: {detail}"
        )
        self.subject: str = subject
        self.detail: str = detail


class RegistryLifecycleError(RegistryError):
    """Raised when an entry's lifecycle status forbids the requested action.

    A ``deprecated`` or ``retired`` entry may not be newly promoted; only an
    ``active`` entry can advance a stage.
    """

    def __init__(self, entry_id: str, status: str, detail: str) -> None:
        """Record the ``entry_id``, its ``status``, and the ``detail``."""
        super().__init__(
            f"registry entry {entry_id!r} is {status!r}: {detail}"
        )
        self.entry_id: str = entry_id
        self.status: str = status
        self.detail: str = detail


class RegistryApprovalError(RegistryError):
    """Raised when a promotion does not carry a recorded approver."""

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record the ``entry_id`` and the approval ``detail``."""
        super().__init__(
            f"registry promotion of {entry_id!r} needs approval: {detail}"
        )
        self.entry_id: str = entry_id
        self.detail: str = detail


class RegistryPromotionError(RegistryError):
    """Raised when a stage transition is not allowed (for example a skip)."""

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record the ``entry_id`` and the transition ``detail``."""
        super().__init__(
            f"invalid promotion for registry entry {entry_id!r}: {detail}"
        )
        self.entry_id: str = entry_id
        self.detail: str = detail


class RegistryCompatibilityError(RegistryError):
    """Raised when an entry's compatibility verdict forbids a promotion.

    A ``prod`` promotion requires an ``exact``/``mappable`` verdict from the
    hub compatibility funnel; an ``incompatible`` (or unknown) verdict is
    refused by name.
    """

    def __init__(self, entry_id: str, verdict: str, detail: str) -> None:
        """Record the ``entry_id``, its ``verdict``, and the ``detail``."""
        super().__init__(
            f"registry entry {entry_id!r} is incompatible ({verdict!r}): "
            f"{detail}"
        )
        self.entry_id: str = entry_id
        self.verdict: str = verdict
        self.detail: str = detail


class RegistrySignatureError(RegistryError):
    """Raised when a signature is missing or does not verify."""

    def __init__(
        self, entry_id: str, detail: str, expected: Optional[str] = None
    ) -> None:
        """Record the ``entry_id``, the ``detail``, and any ``expected``."""
        super().__init__(
            f"registry signature check failed for {entry_id!r}: {detail}"
        )
        self.entry_id: str = entry_id
        self.detail: str = detail
        self.expected: Optional[str] = expected


class RegistryIntegrityError(RegistryError):
    """Raised when an artifact does not match its recorded checksum/size."""

    def __init__(self, entry_id: str, detail: str) -> None:
        """Record the ``entry_id`` and the integrity ``detail``."""
        super().__init__(
            f"registry integrity check failed for {entry_id!r}: {detail}"
        )
        self.entry_id: str = entry_id
        self.detail: str = detail

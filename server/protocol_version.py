"""Single source of the WebSocket protocol version.

The authoritative string lives in ``protocol/protocol_version.txt`` so the
Python server, the JSON schemas, and the generated TypeScript all read the
same value. A safe fallback keeps the server importable when the file is
absent, for example in a wheel that did not ship the contract directory.
"""

from pathlib import Path

#: Used when the contract file is missing or empty.
_DEFAULT = "1.0"

_VERSION_FILE = (
    Path(__file__).resolve().parent.parent
    / "protocol"
    / "protocol_version.txt"
)


def _read_version() -> str:
    """Return the protocol version, falling back to ``1.0`` when absent."""
    try:
        value = _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return _DEFAULT
    return value or _DEFAULT


#: The full protocol version string, for example ``"1.0"``.
PROTOCOL_VERSION: str = _read_version()

#: The MAJOR component, compared during inbound negotiation.
PROTOCOL_MAJOR: str = PROTOCOL_VERSION.split(".", 1)[0]

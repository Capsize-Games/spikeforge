"""Sign the outgoing verification report for the hub API callback.

Mirrors the HMAC-SHA256 pattern already used in this repository for a
signed record (:func:`spikeforge_hub.registry.sign_entry`), adapted for an
HTTP callback: the signature covers the exact request-body bytes, the same
convention GitHub itself uses for webhook signatures
(``X-Hub-Signature-256``), so the receiving side only needs to hash the raw
body it read -- no canonical re-serialization step to keep in sync across
two repositories.
"""

import hashlib
import hmac

#: The callback header carrying the signature, as ``sha256=<hex digest>``.
SIGNATURE_HEADER = "X-Spikeforge-Hub-Signature"


def _as_key(secret: str) -> bytes:
    """Return ``secret`` as bytes, refusing an empty one."""
    if not secret:
        raise ValueError("verification secret must not be empty")
    return secret.encode("utf-8")


def sign_body(body: bytes, secret: str) -> str:
    """Return the ``sha256=<hex>`` signature of ``body`` under ``secret``."""
    digest = hmac.new(_as_key(secret), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def signature_matches(body: bytes, secret: str, header_value: str) -> bool:
    """Return True when ``header_value`` is ``body``'s signature.

    Provided for the hub-api side (or a test standing in for it) to verify
    a callback with the same constant-time comparison this repository uses
    elsewhere for signed records.
    """
    return hmac.compare_digest(sign_body(body, secret), header_value)

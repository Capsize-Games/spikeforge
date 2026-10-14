"""Dependency-free signing and verification of registry entries.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. A signature is an HMAC-SHA256 over the
canonical (sorted-key) JSON of the entry's unsigned projection.
"""

import hashlib
import hmac
import json
from typing import Any

from spikeforge_hub.registry_entry import RegistryEntry
from spikeforge_hub.registry_errors import RegistrySignatureError


def canonical(entry: RegistryEntry) -> str:
    """Return the canonical JSON signed/verified for ``entry``."""
    payload = entry.unsigned()
    payload.pop("signature", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_entry(entry: RegistryEntry, key: Any) -> str:
    """Return the hex HMAC-SHA256 signature of ``entry`` under ``key``."""
    secret = _as_key(key)
    digest = hmac.new(secret, canonical(entry).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def verify_entry(entry: RegistryEntry, key: Any) -> bool:
    """Return True when ``entry`` carries a matching signature.

    Raises :class:`RegistrySignatureError` when the signature is absent or does
    not match, so a caller that needs a boolean can catch the typed error.
    """
    if not entry.signature:
        raise RegistrySignatureError(entry.id, "entry carries no signature")
    expected = sign_entry(entry, key)
    if not hmac.compare_digest(entry.signature, expected):
        raise RegistrySignatureError(
            entry.id, "signature does not match the entry", expected
        )
    return True


def _as_key(key: Any) -> bytes:
    """Return ``key`` as bytes, refusing an empty secret."""
    if isinstance(key, bytes):
        secret = key
    elif isinstance(key, str):
        secret = key.encode("utf-8")
    else:
        raise RegistrySignatureError(
            "(key)", "signing key must be str or bytes"
        )
    if not secret:
        raise RegistrySignatureError("(key)", "signing key must not be empty")
    return secret

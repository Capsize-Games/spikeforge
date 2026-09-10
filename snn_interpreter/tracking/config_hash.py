"""Stable hashing of a run's reproducibility-relevant configuration."""

import hashlib
import json
from typing import Any, Mapping


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return a deterministic JSON rendering of ``payload``.

    Keys are sorted and the separators are tightened so equivalent configs
    hash identically regardless of how the caller built the mapping.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 hash of ``config``'s canonical JSON form.

    The hash covers only the reproducibility-relevant configuration, so two
    runs with identical settings compare equal regardless of when they ran or
    which metric history they produced.
    """
    digest = hashlib.sha256(canonical_json(config).encode("utf-8"))
    return digest.hexdigest()

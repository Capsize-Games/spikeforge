"""Schema constants for the governed artifact registry.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. Importers keep using the ``registry`` path;
this module is where the document shape lives.
"""

import re
from typing import Sequence, Tuple

from spikeforge_hub.compat import EXACT, INCOMPATIBLE, MAPPABLE

#: Bumped whenever the registry document schema changes incompatibly.
REGISTRY_SCHEMA_VERSION = 1

#: Promotion stages, in the only order a promotion may advance through.
STAGES: Tuple[str, ...] = ("dev", "staging", "prod")

#: Lifecycle statuses an entry may carry. ``active`` may be promoted; the other
#: two are terminal for promotion, and ``deprecated`` is additionally reported.
STATUSES: Tuple[str, ...] = ("active", "deprecated", "retired")

#: Lineage links, in the order the plan names them (dataset -> ... deployment).
LINEAGE_ORDER: Tuple[str, ...] = ("dataset", "model", "bundle", "deployment")

#: Compatibility verdicts the hub can record; only the first two may ship.
COMPATIBILITY_VERDICTS: Tuple[str, ...] = (EXACT, MAPPABLE, INCOMPATIBLE)

#: Default registry document name under :data:`spikeforge.config.REGISTRY_DIR`.
REGISTRY_FILE = "registry.json"

#: An entry id that cannot escape a directory or confuse the cache layout.
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
#: A concrete dotted version, e.g. ``0.1`` or ``0.1.0``.
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){0,2}$")
#: A hex sha256 digest.
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED: Tuple[str, ...] = ("id", "name", "stage", "status", "version")
_FIELDS: Tuple[str, ...] = _REQUIRED + (
    "artifact_path",
    "artifact_sha256",
    "compat",
    "approver",
    "approved_at",
    "signature",
    "lineage",
    "created_at",
    "schema_version",
)

#: Re-exported so callers can build a stage list without importing STAGES.
StageList = Sequence[str]

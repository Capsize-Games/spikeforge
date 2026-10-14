"""Governed, signable registry of promoted artifacts and their lineage.

The model hub already validates a *catalog* of curated artifacts
(:mod:`spikeforge_hub.catalog`) and classifies an artifact against a shipped
preset (:mod:`spikeforge_hub.compat`). This module adds the missing
**governance** layer the PT-W7 workstream calls for: an explicit promotion
lifecycle (``dev -> staging -> prod``), a recorded approver, an artifact
signature that can be verified, and lineage from dataset to deployment.

Design choices (documented because the plan leaves them open):

* **Extend the hub, not a new distribution.** The plan offers
  ``spikeforge-registry`` as *new (or extend hub)*; the governance layer lives
  in ``spikeforge_hub`` and owns the ``spikeforge-registry`` console script, so
  no third distribution (and no extra import root) is introduced.
* **Reuse the existing machinery.** Compatibility reuses the
  :data:`~spikeforge_hub.compat.EXACT`/:data:`~spikeforge_hub.compat.MAPPABLE`
  vocabulary, and checksum/verification reuses
  :func:`spikeforge_hub.verify.verify_file`, so there is one definition of
  "verified" and "compatible".
* **Signing is dependency-free.** A signature is an HMAC-SHA256 over the
  canonical (sorted-key) JSON of the entry's unsigned projection. It detects a
  tampered or absent signature without pulling in a crypto stack; a caller that
  needs asymmetric signing can swap :func:`sign_entry` for its own.

This module is the import root: the schema lives in
:mod:`spikeforge_hub.registry_schema`, the validators in
:mod:`spikeforge_hub.registry_checks`, the entry type in
:mod:`spikeforge_hub.registry_entry`, signing in
:mod:`spikeforge_hub.registry_signing`, promotion in
:mod:`spikeforge_hub.registry_promotion`, storage in
:mod:`spikeforge_hub.registry_store`, and catalog governance in
:mod:`spikeforge_hub.registry_governance`. Everything is re-exported here
so importers keep using this path.
"""

from spikeforge_hub.compat import EXACT
from spikeforge_hub.registry_entry import (
    RegistryEntry,
    lineage_report,
    stage_rank,
    verify_artifact,
)
from spikeforge_hub.registry_governance import (
    catalog_governance,
    validate_catalog,
)
from spikeforge_hub.registry_promotion import promote
from spikeforge_hub.registry_schema import (
    COMPATIBILITY_VERDICTS,
    LINEAGE_ORDER,
    REGISTRY_FILE,
    REGISTRY_SCHEMA_VERSION,
    STAGES,
    STATUSES,
    StageList,
)
from spikeforge_hub.registry_signing import (
    canonical,
    sign_entry,
    verify_entry,
)
from spikeforge_hub.registry_store import Registry, default_registry_path

__all__ = [
    "COMPATIBILITY_VERDICTS",
    "EXACT",
    "LINEAGE_ORDER",
    "REGISTRY_FILE",
    "REGISTRY_SCHEMA_VERSION",
    "STAGES",
    "STATUSES",
    "Registry",
    "RegistryEntry",
    "StageList",
    "canonical",
    "catalog_governance",
    "default_registry_path",
    "lineage_report",
    "promote",
    "sign_entry",
    "stage_rank",
    "validate_catalog",
    "verify_artifact",
    "verify_entry",
]

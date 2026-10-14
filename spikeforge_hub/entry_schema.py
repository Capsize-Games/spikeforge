"""Schema constants for a curated model-hub catalog entry.

The allowed vocabularies (frameworks, kinds, sources), the field list, and
the compiled shape rules live here so the validators in
:mod:`spikeforge_hub.entry_checks` and the :class:`HubEntry
<spikeforge_hub.entry.HubEntry>` dataclass share one definition.
:mod:`spikeforge_hub.entry` re-exports the public names, so importers keep
using that module.
"""

import re
from typing import Tuple

#: Frameworks the curated catalog may declare.
#:
#: ``spikeforge`` is the community index's framework: a hosted entry is a
#: ``.spkf`` bundle built from a saved model of this project's own
#: topologies, so it belongs to no third-party framework.
FRAMEWORKS: Tuple[str, ...] = (
    "snntorch",
    "nir",
    "spikingjelly",
    "norse",
    "lava",
    "hf",
    "spikeforge",
)
#: Artifact kinds an entry may describe.
#:
#: ``bundle`` is a portable ``.spkf`` archive: manifest, weights, and frozen
#: encode config in one file. Only community entries use it.
KINDS: Tuple[str, ...] = (
    "nir_graph",
    "state_dict",
    "framework_weights",
    "bundle",
)
#: Where an entry's artifact comes from.
#:
#: ``bundled`` renders a shipped topology preset to a NIR graph -- structure
#: with freshly-initialised weights. ``reference`` is the one source that
#: carries *trained* weights: a checkpoint this project trained itself, shipped
#: inside the distribution and named by :attr:`HubEntry.weights`. The
#: distinction matters because a catalog of untrained shapes and a catalog of
#: trained models are different products, and the entry has to say which it is.
#: ``community`` is a hosted upload, served by the hub API's index and never
#: hand-added to the shipped catalog; see ``CURATION.md``.
SOURCES: Tuple[str, ...] = (
    "bundled",
    "reference",
    "url",
    "hf_repo",
    "community",
)

#: Directory inside this package holding the shipped reference checkpoints.
WEIGHTS_DIR = "weights"

#: Explicit marker for a candidate whose upstream license is not verified.
#:
#: An entry carrying this marker is allowed to load only so it can be reported
#: as *not ready* (see :func:`spikeforge_hub.catalog.availability`); it is
#: never presented as an available, ready-to-use artifact.
UNVERIFIED_CANDIDATE = "unverified-candidate"
#: One SPDX-style id (no whitespace), optionally joined by SPDX operators.
_LICENSE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9.+-]*"
    r"(?:\s+(?:AND|OR|WITH)\s+[A-Za-z0-9][A-Za-z0-9.+-]*)*$"
)
#: Single words that are not license ids, however id-shaped they look.
#:
#: The shape rule above rejects ``"see upstream"`` because it has a space, but
#: ``"unknown"``, ``"TBD"`` and ``"NOASSERTION"`` are one token each and used
#: to pass it -- the exact escape the rule exists to close, wearing an
#: identifier's shape. Matching is case-insensitive and whole-string, so real
#: ids that merely resemble one of these (``Unlicense``) are untouched. An
#: entry whose license genuinely is not established declares
#: :data:`UNVERIFIED_CANDIDATE`, which is explicit and machine-readable.
_LICENSE_ESCAPES = frozenset(
    {
        "unknown",
        "unspecified",
        "unverified",
        "unlicensed",
        "tbd",
        "todo",
        "none",
        "noassertion",
        "n/a",
        "na",
        "other",
        "custom",
        "various",
        "proprietary",
        "private",
    }
)

_REQUIRED: Tuple[str, ...] = (
    "id",
    "name",
    "framework",
    "kind",
    "source",
    "license",
    "notes",
)
_FIELDS: Tuple[str, ...] = _REQUIRED + (
    "url",
    "hf_repo",
    "sha256",
    "size_bytes",
    "max_bytes",
    "topology",
    "input_shape",
    "weights",
    "dataset",
    "dataset_license",
    "dataset_attribution",
    "test_accuracy",
    "test_samples",
)
#: A packaged weights filename: one path segment, no directory traversal.
_WEIGHTS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pt$")
#: A catalog id. Curated ids look like ``reference/mnist-fc``; a community
#: id is ``@handle/slug``, and the leading ``@`` is what keeps the two
#: unambiguous in a list, a URL, and a command-line argument. The hub API
#: reserves every curated prefix as a handle, so this repo renders both
#: shapes without re-validating the split.
_ID = re.compile(r"^@?[A-Za-z0-9][A-Za-z0-9._/-]*$")

#: (field, message) pairs a reference entry must supply non-empty text for.
_REFERENCE_TEXT_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("sha256", "source 'reference' needs a 'sha256' checksum"),
    ("dataset", "source 'reference' needs the 'dataset' it trained on"),
    (
        "dataset_license",
        "source 'reference' needs a 'dataset_license'; an entry that "
        "cannot say what its training data permits is not one this "
        "project can stand behind",
    ),
    (
        "dataset_attribution",
        "source 'reference' needs a 'dataset_attribution'; the credit "
        "the dataset's publisher asks for has to travel with the entry",
    ),
)

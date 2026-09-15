"""Typed metadata for a curated model-hub catalog entry.

Each entry is validated against the schema in ``models.json`` before it is
usable, so a malformed record is reported as a typed
:class:`~spikeforge_hub.errors.HubCatalogError` instead of being
silently skipped. Unknown frameworks, kinds, and sources are rejected by
name, honouring the project's honesty rule.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from spikeforge_hub.errors import HubCatalogError

#: Frameworks the curated catalog may declare.
FRAMEWORKS: Tuple[str, ...] = (
    "snntorch",
    "nir",
    "spikingjelly",
    "norse",
    "lava",
    "hf",
)
#: Artifact kinds an entry may describe.
KINDS: Tuple[str, ...] = ("nir_graph", "state_dict", "framework_weights")
#: Where an entry's artifact comes from.
#:
#: ``bundled`` renders a shipped topology preset to a NIR graph -- structure
#: with freshly-initialised weights. ``reference`` is the one source that
#: carries *trained* weights: a checkpoint this project trained itself, shipped
#: inside the distribution and named by :attr:`HubEntry.weights`. The
#: distinction matters because a catalog of untrained shapes and a catalog of
#: trained models are different products, and the entry has to say which it is.
SOURCES: Tuple[str, ...] = ("bundled", "reference", "url", "hf_repo")

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
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True)
class HubEntry:
    """A curated catalog entry: where a model comes from and what it is."""

    id: str
    name: str
    framework: str
    kind: str
    source: str
    license: str
    notes: str
    url: Optional[str] = None
    hf_repo: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    topology: Optional[str] = None
    input_shape: Optional[str] = None
    #: For ``source: "reference"``: the checkpoint filename inside
    #: ``spikeforge_hub/weights/``.
    weights: Optional[str] = None
    #: What the shipped checkpoint was trained on, and what it scores on that
    #: dataset's complete held-out split. Present only for trained entries, so
    #: an entry that claims an accuracy is exactly an entry that has weights.
    dataset: Optional[str] = None
    #: The **training data's** own licence, which is not
    #: :attr:`license`: that one describes the weights, this project's own
    #: artifact. A trained entry has to say what the data it encodes permits,
    #: so the two are recorded separately and neither stands in for the other.
    dataset_license: Optional[str] = None
    #: The credit the dataset's publisher asks for, verbatim where they
    #: request specific wording.
    dataset_attribution: Optional[str] = None
    test_accuracy: Optional[float] = None
    test_samples: Optional[int] = None

    @property
    def trained(self) -> bool:
        """Return True when this entry carries trained weights."""
        return self.source == "reference"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HubEntry":
        """Validate ``data`` and build an entry, raising on any problem."""
        _check_required(data)
        _check_id(data)
        _check_allowed(data)
        _check_license(data)
        _check_dataset_license(data)
        _check_source(data)
        return cls(**_fields(data))

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry as a JSON-able mapping in schema field order."""
        values = asdict(self)
        return {name: values[name] for name in _FIELDS}


def _label(data: Mapping[str, Any]) -> str:
    """Return the entry id for messages, falling back to ``(unnamed)``."""
    value = data.get("id")
    return value if isinstance(value, str) and value else "(unnamed)"


def _text(value: Any) -> bool:
    """Return True when ``value`` is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def _check_required(data: Mapping[str, Any]) -> None:
    """Reject an entry that omits or blanks a required text field."""
    missing = [key for key in _REQUIRED if not _text(data.get(key))]
    if missing:
        raise HubCatalogError(
            _label(data), f"missing required fields: {', '.join(missing)}"
        )


def _check_id(data: Mapping[str, Any]) -> None:
    """Reject an id that could escape or confuse the cache layout."""
    entry_id = str(data.get("id", ""))
    if not _ID.match(entry_id):
        raise HubCatalogError(
            _label(data), "id must match [A-Za-z0-9][A-Za-z0-9._/-]*"
        )
    if ".." in entry_id:
        raise HubCatalogError(_label(data), "id must not contain '..'")


def _check_allowed(data: Mapping[str, Any]) -> None:
    """Report an unknown framework, kind, or source by name."""
    for key, allowed in (
        ("framework", FRAMEWORKS),
        ("kind", KINDS),
        ("source", SOURCES),
    ):
        value = data.get(key)
        if value not in allowed:
            raise HubCatalogError(
                _label(data),
                f"unknown {key} {value!r}; expected one of "
                f"{', '.join(allowed)}",
            )


def _concrete_license(value: Any) -> bool:
    """Return True when ``value`` is a concrete SPDX-style id or the marker."""
    if value == UNVERIFIED_CANDIDATE:
        return True
    if not isinstance(value, str):
        return False
    if value.strip().lower() in _LICENSE_ESCAPES:
        return False
    return bool(_LICENSE.match(value))


def _check_license(data: Mapping[str, Any]) -> None:
    """Reject a license that is neither a concrete id nor the marker.

    ``"see upstream"`` and other free-text escapes are refused so an entry can
    never present an unverified license as if it were a concrete one. A remote
    entry that is not yet verified must declare the explicit
    :data:`UNVERIFIED_CANDIDATE` marker, which loads but is reported
    unavailable by :func:`spikeforge_hub.catalog.availability`.
    """
    value = data.get("license")
    if _concrete_license(value):
        return
    raise HubCatalogError(
        _label(data),
        f"license {value!r} is not a concrete SPDX-style id or the "
        f"{UNVERIFIED_CANDIDATE!r} marker",
    )


def _check_dataset_license(data: Mapping[str, Any]) -> None:
    """Hold ``dataset_license`` to exactly the rule ``license`` is held to.

    The same rule, deliberately: a looser one would let the training data's
    terms be recorded as the free text the weights' own licence may not be,
    which is the escape hatch this validation exists to close. The marker is
    accepted here too -- a dataset whose upstream terms could not be read
    from a primary source must say so rather than carry an inferred licence.
    Unlike :attr:`HubEntry.license`, this field does not gate availability:
    what the training data permits is disclosure for a reader to judge, not a
    claim about whether the shipped weights load.
    """
    value = data.get("dataset_license")
    if value is None or _concrete_license(value):
        return
    raise HubCatalogError(
        _label(data),
        f"dataset_license {value!r} is not a concrete SPDX-style id or the "
        f"{UNVERIFIED_CANDIDATE!r} marker",
    )


def _check_source(data: Mapping[str, Any]) -> None:
    """Reject a source whose required locator field is missing."""
    source = data.get("source")
    if source == "url" and not _text(data.get("url")):
        raise HubCatalogError(_label(data), "source 'url' needs a 'url'")
    if source == "hf_repo" and not _text(data.get("hf_repo")):
        raise HubCatalogError(
            _label(data), "source 'hf_repo' needs an 'hf_repo'"
        )
    if source == "reference":
        _check_reference(data)
    _check_optional(data)
    _check_scores(data)


def _check_reference(data: Mapping[str, Any]) -> None:
    """Enforce what a shipped, trained entry has to declare.

    A reference entry is the only kind whose bytes this project vouches for,
    so it has to name the packaged file, pin its checksum, and say what it was
    trained on -- otherwise "trained weights" is an unfalsifiable claim. It
    also has to say what that training data permits and who to credit for it:
    :attr:`HubEntry.license` covers the weights only, and an entry that
    records one licence while staying silent on the other invites a reader to
    assume they are the same.
    """
    weights = data.get("weights")
    if not _text(weights):
        raise HubCatalogError(
            _label(data), "source 'reference' needs a 'weights' filename"
        )
    if not _WEIGHTS_NAME.match(str(weights)):
        raise HubCatalogError(
            _label(data),
            f"weights {weights!r} must be a bare '*.pt' filename inside "
            f"the packaged {WEIGHTS_DIR}/ directory",
        )
    if not _text(data.get("sha256")):
        raise HubCatalogError(
            _label(data), "source 'reference' needs a 'sha256' checksum"
        )
    if not _text(data.get("dataset")):
        raise HubCatalogError(
            _label(data),
            "source 'reference' needs the 'dataset' it trained on",
        )
    if not _text(data.get("dataset_license")):
        raise HubCatalogError(
            _label(data),
            "source 'reference' needs a 'dataset_license'; an entry that "
            "cannot say what its training data permits is not one this "
            "project can stand behind",
        )
    if not _text(data.get("dataset_attribution")):
        raise HubCatalogError(
            _label(data),
            "source 'reference' needs a 'dataset_attribution'; the credit "
            "the dataset's publisher asks for has to travel with the entry",
        )
    if data.get("test_accuracy") is None:
        raise HubCatalogError(
            _label(data),
            "source 'reference' needs a 'test_accuracy'; a trained entry "
            "that does not say what it scores is not a useful one",
        )


def _check_scores(data: Mapping[str, Any]) -> None:
    """Reject a malformed or impossible reported score."""
    accuracy = data.get("test_accuracy")
    if accuracy is not None:
        if isinstance(accuracy, bool) or not isinstance(
            accuracy, (int, float)
        ):
            raise HubCatalogError(
                _label(data), "test_accuracy must be a number"
            )
        if not 0.0 <= float(accuracy) <= 100.0:
            raise HubCatalogError(
                _label(data),
                f"test_accuracy {accuracy} is outside 0-100 percent",
            )
    samples = data.get("test_samples")
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int)
    ):
        raise HubCatalogError(_label(data), "test_samples must be an integer")


def _check_optional(data: Mapping[str, Any]) -> None:
    """Reject malformed optional checksum and size fields."""
    size = data.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise HubCatalogError(_label(data), "size_bytes must be an integer")
    sha = data.get("sha256")
    if sha is not None and not _text(sha):
        raise HubCatalogError(
            _label(data), "sha256 must be a non-empty string or null"
        )


def _fields(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Project ``data`` onto the declared schema fields."""
    return {name: data.get(name) for name in _FIELDS}

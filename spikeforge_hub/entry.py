"""Typed metadata for a curated model-hub catalog entry.

Each entry is validated against the schema in ``models.json`` before it is
usable, so a malformed record is reported as a typed
:class:`~spikeforge_hub.errors.HubCatalogError` instead of being
silently skipped. Unknown frameworks, kinds, and sources are rejected by
name, honouring the project's honesty rule.

This module is the import root: the schema constants live in
:mod:`spikeforge_hub.entry_schema`, the validators in
:mod:`spikeforge_hub.entry_checks`, and both are re-exported here so
importers keep using this path.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from spikeforge_hub.entry_checks import (
    _check_allowed,
    _check_dataset_license,
    _check_id,
    _check_license,
    _check_required,
    _check_source,
    _fields,
)
from spikeforge_hub.entry_schema import (
    _FIELDS,
    FRAMEWORKS,
    KINDS,
    SOURCES,
    UNVERIFIED_CANDIDATE,
    WEIGHTS_DIR,
)

__all__ = [
    "FRAMEWORKS",
    "HubEntry",
    "KINDS",
    "SOURCES",
    "UNVERIFIED_CANDIDATE",
    "WEIGHTS_DIR",
]


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

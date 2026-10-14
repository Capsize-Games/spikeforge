"""Validators for a curated model-hub catalog entry.

Every ``_check_*`` function rejects one class of malformed record with a
typed :class:`~spikeforge_hub.errors.HubCatalogError`, so a bad entry is
reported instead of silently skipped. The schema constants they check
against live in :mod:`spikeforge_hub.entry_schema`; :class:`HubEntry
<spikeforge_hub.entry.HubEntry>` calls these from ``from_dict``. The
required-field, id, and projection primitives are shared with the
governed registry through :mod:`spikeforge_hub._validation`.
"""

from typing import Any, Dict, Mapping, Sequence, Tuple

from spikeforge_hub._validation import (
    check_id,
    check_required,
    project_fields,
)
from spikeforge_hub._validation import label as _label
from spikeforge_hub._validation import text as _text
from spikeforge_hub.entry_schema import (
    _FIELDS,
    _ID,
    _LICENSE,
    _LICENSE_ESCAPES,
    _REFERENCE_TEXT_FIELDS,
    _REQUIRED,
    _WEIGHTS_NAME,
    FRAMEWORKS,
    KINDS,
    SOURCES,
    UNVERIFIED_CANDIDATE,
    WEIGHTS_DIR,
)
from spikeforge_hub.errors import HubCatalogError


def _check_required(data: Mapping[str, Any]) -> None:
    """Reject an entry that omits or blanks a required text field."""
    check_required(data, _REQUIRED, HubCatalogError)


def _check_id(data: Mapping[str, Any]) -> None:
    """Reject an id that could escape or confuse the cache layout."""
    check_id(
        data, _ID, "@?[A-Za-z0-9][A-Za-z0-9._/-]*", HubCatalogError
    )


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


def _check_url_checksum(data: Mapping[str, Any]) -> None:
    """Reject a ``url`` source without a pinned ``sha256`` checksum."""
    if data.get("source") == "url" and not _text(data.get("sha256")):
        raise HubCatalogError(
            _label(data),
            "source 'url' needs a 'sha256' checksum; a download "
            "without a pinned checksum cannot be verified at import",
        )


def _check_source(data: Mapping[str, Any]) -> None:
    """Reject a source whose required locator field is missing."""
    source = data.get("source")
    if source in ("url", "community") and not _text(data.get("url")):
        raise HubCatalogError(
            _label(data), f"source {source!r} needs a 'url'"
        )
    if source == "hf_repo" and not _text(data.get("hf_repo")):
        raise HubCatalogError(
            _label(data), "source 'hf_repo' needs an 'hf_repo'"
        )
    _check_url_checksum(data)
    if source == "reference":
        _check_reference(data)
    _check_optional(data)
    _check_scores(data)


def _check_text_fields(
    data: Mapping[str, Any], fields: Sequence[Tuple[str, str]]
) -> None:
    """Raise for the first ``fields`` entry missing non-empty text."""
    for field, message in fields:
        if not _text(data.get(field)):
            raise HubCatalogError(_label(data), message)


def _check_weights_field(data: Mapping[str, Any]) -> None:
    """Reject a missing or malformed reference ``weights`` filename."""
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


def _check_test_accuracy(data: Mapping[str, Any]) -> None:
    """Reject a reference entry missing a reported test accuracy."""
    if data.get("test_accuracy") is None:
        raise HubCatalogError(
            _label(data),
            "source 'reference' needs a 'test_accuracy'; a trained entry "
            "that does not say what it scores is not a useful one",
        )


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
    _check_weights_field(data)
    _check_text_fields(data, _REFERENCE_TEXT_FIELDS)
    _check_test_accuracy(data)


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
    """Reject a malformed optional checksum, size, or byte cap."""
    size = data.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise HubCatalogError(_label(data), "size_bytes must be an integer")
    limit = data.get("max_bytes")
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
    ):
        raise HubCatalogError(
            _label(data), "max_bytes must be a positive integer"
        )
    sha = data.get("sha256")
    if sha is not None and not _text(sha):
        raise HubCatalogError(
            _label(data), "sha256 must be a non-empty string or null"
        )


def _fields(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Project ``data`` onto the declared schema fields."""
    return project_fields(data, _FIELDS)

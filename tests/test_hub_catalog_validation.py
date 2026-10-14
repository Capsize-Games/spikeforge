"""Catalog entry validation, availability gating, and the hub probe."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

from spikeforge_hub import probe
from spikeforge_hub.catalog import SCHEMA_VERSION, availability, load_catalog
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError

_EXPECTED_KEYS = {"huggingface_hub_available", "huggingface_hub_version"}


def _valid_entry(**overrides: Any) -> Dict[str, Any]:
    """Return a minimal schema-valid entry, with optional overrides."""
    data: Dict[str, Any] = {
        "id": "nir/example",
        "name": "Example",
        "framework": "nir",
        "kind": "nir_graph",
        "source": "bundled",
        "license": "BSD-3-Clause",
        "notes": "unit-test entry",
    }
    data.update(overrides)
    return data


def test_malformed_entry_is_reported_not_dropped(tmp_path: Path) -> None:
    """A bad entry is reported while the good entries still load."""
    payload = {
        "version": SCHEMA_VERSION,
        "entries": [
            _valid_entry(),
            _valid_entry(id="nir/bad", framework="unknown-framework"),
        ],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    found, problems = load_catalog(path)
    assert [entry.id for entry in found] == ["nir/example"]
    assert len(problems) == 1
    assert "unknown-framework" in problems[0]


def test_non_object_entry_is_reported(tmp_path: Path) -> None:
    """A non-object list item is reported instead of crashing the loader."""
    payload = {"version": SCHEMA_VERSION, "entries": ["nope"]}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    found, problems = load_catalog(path)
    assert found == []
    assert problems == ["entry #0: not a JSON object"]


def test_unknown_framework_is_named_in_the_error() -> None:
    """An unknown framework is rejected by name (honesty rule)."""
    with pytest.raises(HubCatalogError) as ctx:
        HubEntry.from_dict(_valid_entry(framework="pytorch"))
    assert "framework" in str(ctx.value)


def test_url_source_requires_a_url() -> None:
    """A URL-sourced entry without a ``url`` is rejected."""
    with pytest.raises(HubCatalogError):
        HubEntry.from_dict(_valid_entry(source="url"))


def test_url_source_requires_a_checksum() -> None:
    """A URL-sourced entry without a ``sha256`` is rejected.

    A download without a pinned checksum cannot be verified at import,
    so the catalog refuses to carry one -- the same bar reference
    entries already meet.
    """
    with pytest.raises(HubCatalogError, match="sha256"):
        HubEntry.from_dict(
            _valid_entry(source="url", url="https://example.test/m.pt")
        )
    entry = HubEntry.from_dict(
        _valid_entry(
            source="url",
            url="https://example.test/m.pt",
            sha256="0" * 64,
        )
    )
    assert entry.sha256 == "0" * 64


def test_hf_source_requires_a_repo() -> None:
    """An HF-sourced entry without an ``hf_repo`` is rejected."""
    with pytest.raises(HubCatalogError):
        HubEntry.from_dict(_valid_entry(source="hf_repo"))


def test_missing_required_field_is_rejected() -> None:
    """An entry that omits a required field is rejected."""
    data = _valid_entry()
    data.pop("license")
    with pytest.raises(HubCatalogError):
        HubEntry.from_dict(data)


def test_placeholder_license_is_rejected() -> None:
    """A free-text 'see upstream' license is rejected, not silently kept."""
    with pytest.raises(HubCatalogError) as ctx:
        HubEntry.from_dict(_valid_entry(license="see upstream"))
    assert "license" in str(ctx.value)


def test_concrete_spdx_license_is_accepted() -> None:
    """A concrete SPDX-style id (including a compound expression) is valid."""
    single = HubEntry.from_dict(_valid_entry(license="Apache-2.0"))
    assert single.license == "Apache-2.0"
    compound = HubEntry.from_dict(_valid_entry(license="MIT OR Apache-2.0"))
    assert compound.license == "MIT OR Apache-2.0"


def test_unverified_candidate_marker_disables_the_entry() -> None:
    """The marker loads but the entry reports unavailable with a reason."""
    entry = HubEntry.from_dict(_valid_entry(license=UNVERIFIED_CANDIDATE))
    available, reason = availability(entry)
    assert available is False
    assert reason and "unverified" in reason


def test_remote_entry_with_placeholder_license_is_flagged(
    tmp_path: Path,
) -> None:
    """A remote entry with a free-text license is reported, not kept."""
    payload = {
        "version": SCHEMA_VERSION,
        "entries": [
            _valid_entry(),
            _valid_entry(
                id="hf/example",
                source="hf_repo",
                hf_repo="org/repo",
                license="see upstream",
            ),
        ],
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    found, problems = load_catalog(path)
    assert [entry.id for entry in found] == ["nir/example"]
    assert len(problems) == 1
    assert "license" in problems[0]


def test_probe_reports_expected_keys() -> None:
    """The probe reports exactly the documented keys."""
    assert set(probe.capability()) == _EXPECTED_KEYS
    assert set(probe.capability()) == set(probe.REPORT_KEYS)


def test_probe_survives_absent_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing dependency must not raise and stays reported absent."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    report = probe.capability()
    assert report["huggingface_hub_available"] is False
    assert report["huggingface_hub_version"] is None
    assert probe.available() is False
    assert probe.module() is None


def test_hf_entries_are_unavailable_without_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HF entry names the missing extra instead of appearing available."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    entry = HubEntry.from_dict(
        _valid_entry(source="hf_repo", hf_repo="org/repo")
    )
    available, reason = availability(entry)
    assert available is False
    assert reason and "hub" in reason


def test_installed_huggingface_hub_is_detected() -> None:
    """When the extra is installed the probe reports it available."""
    if not probe.available():
        pytest.skip("huggingface_hub is not installed")
    assert probe.capability()["huggingface_hub_available"] is True

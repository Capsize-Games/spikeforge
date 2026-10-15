"""The import integrity gate: checksums, opt-ins, and verification meta."""

import hashlib
import importlib
import io
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.topology.registry import build_topology
from spikeforge_hub import cache
from spikeforge_hub.catalog import entries as catalog_entries
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubIntegrityError
from spikeforge_hub.import_model import ALLOW_UNVERIFIED_ENV, import_model

pytest.importorskip("nir")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path

def _seed_blob(entry_id: str, blob: bytes) -> None:
    """Write ``blob`` as the cached artifact for ``entry_id``."""
    dest = Path(cache.entry_dir(entry_id))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "artifact.pt").write_bytes(blob)


def _fetched_entry(
    entry_id: str, source: str, sha256: Optional[str]
) -> HubEntry:
    """Return a valid fetched-source entry, checksummed when given one."""
    data: Dict[str, Any] = {
        "id": entry_id,
        "name": "fetched",
        "framework": "spikeforge",
        "kind": "state_dict",
        "source": source,
        "license": "BSD-3-Clause",
        "notes": "unit-test entry",
        "url": "https://example.test/artifact.bin",
    }
    if sha256 is not None:
        data["sha256"] = sha256
    return HubEntry.from_dict(data)


def _announce(
    monkeypatch: pytest.MonkeyPatch, entry: HubEntry
) -> None:
    """Serve ``entry`` from the catalog for its own id only."""
    # The package attribute ``spikeforge_hub.catalog`` is the listing
    # function, not the module, so resolve the module explicitly.
    module = importlib.import_module("spikeforge_hub.catalog")

    def fake(entry_id: str) -> Optional[HubEntry]:
        """Return the announced entry, delegating every other id."""
        if entry_id == entry.id:
            return entry
        return catalog_get(entry_id)

    monkeypatch.setattr(module, "get", fake)


def _state_dict_blob() -> bytes:
    """Return a freshly built fc_small state dict, serialized."""
    _spec, module = build_topology("fc_small")
    buffer = io.BytesIO()
    torch.save(dict(module.state_dict()), buffer)
    return buffer.getvalue()


def test_tampered_url_entry_is_refused_and_writes_nothing(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes that mismatch their pinned checksum raise, saving nothing."""
    entry = _fetched_entry("test/tampered", "url", "0" * 64)
    _seed_blob(entry.id, b"tampered-bytes")
    _announce(monkeypatch, entry)
    with pytest.raises(HubIntegrityError, match="sha256 mismatch"):
        import_model(entry.id)
    models = sandbox / "models"
    assert not models.exists() or not list(models.glob("*.pt"))


def test_verified_url_entry_promotes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bytes matching their pinned checksum import with verified meta."""
    blob = _state_dict_blob()
    entry = _fetched_entry(
        "test/verified", "url", hashlib.sha256(blob).hexdigest()
    )
    _seed_blob(entry.id, blob)
    _announce(monkeypatch, entry)
    result = import_model(entry.id)
    assert result["promoted"] is True
    assert result["meta"]["verification"]["status"] == "verified"
    assert result["verification"]["verified"] is True


def test_checksum_less_entry_is_refused_unless_opted_in(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checksum-less bytes need the explicit opt-in, then record it."""
    entry = _fetched_entry("@ada/no-checksum", "community", None)
    _seed_blob(entry.id, _state_dict_blob())
    _announce(monkeypatch, entry)
    with pytest.raises(HubIntegrityError, match="allow-unverified"):
        import_model(entry.id)
    result = import_model(entry.id, allow_unverified=True)
    assert result["promoted"] is True
    assert result["meta"]["verification"]["status"] == "unverified"


def test_env_opt_in_allows_unverified(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment variable opts in where no flag can reach."""
    entry = _fetched_entry("@ada/no-checksum", "community", None)
    _seed_blob(entry.id, _state_dict_blob())
    _announce(monkeypatch, entry)
    monkeypatch.setenv(ALLOW_UNVERIFIED_ENV, "1")
    result = import_model(entry.id)
    assert result["promoted"] is True
    assert result["meta"]["verification"]["status"] == "unverified"


def test_reference_entries_import_verified(sandbox: Path) -> None:
    """Every shipped reference checkpoint verifies against its pin.

    The gate checks the packaged bytes against the catalog checksum; the
    funnel's own promote/refuse verdict is orthogonal and unchanged.
    """
    found = [
        entry.id for entry in catalog_entries()
        if entry.source == "reference"
    ]
    # Six stock reference checkpoints plus the RSI search policy.
    assert len(found) == 7
    for entry_id in found:
        result = import_model(entry_id)
        assert result["meta"]["verification"]["status"] == "verified"
        assert result["verification"]["verified"] is True


def test_bundled_entry_promotes_and_reports_local_bytes(
    sandbox: Path,
) -> None:
    """A bundled preset promotes; its verdict names the local render."""
    result = import_model("nir/fc_small")
    assert result["promoted"] is True
    assert result["verification"]["status"] == "unverified"
    assert "nothing was fetched" in result["verification"]["reason"]

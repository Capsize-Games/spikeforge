"""The inspect -> compat -> promote funnel and the weight loader."""

import hashlib
import importlib
import io
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import torch

from spikeforge.network import model_store
from spikeforge.nir_bridge import save_graph, to_nir
from spikeforge.topology.registry import build_topology
from spikeforge_hub import cache
from spikeforge_hub.catalog import entries as catalog_entries
from spikeforge_hub.catalog import get as catalog_get
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError, HubIntegrityError
from spikeforge_hub.import_model import ALLOW_UNVERIFIED_ENV, import_model
from spikeforge_hub.weight_map import load_weights

pytest.importorskip("nir")


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def _graph_file(tmp_path: Path, perturb: bool = False) -> str:
    """Save an fc_small graph, optionally with a perturbed fc1 weight."""
    spec, module = build_topology("fc_small")
    graph = to_nir(spec, module)
    if perturb:
        graph.nodes["fc1"].weight = np.zeros((5, 784), np.float32)
    path = tmp_path / "graph.json"
    save_graph(graph, str(path))
    return str(path)


def test_bundled_entry_promotes_with_validation(sandbox: Path) -> None:
    """A bundled NIR preset imports exact, validates, and promotes."""
    result = import_model("nir/fc_small")
    assert result["verdict"]["verdict"] == "exact"
    assert result["promoted"] is True
    assert result["validation"]["within_tolerance"] is True
    assert os.path.exists(result["destination"])


def test_import_records_hub_provenance(sandbox: Path) -> None:
    """The promoted checkpoint's meta names the hub source and topology."""
    result = import_model("nir/fc_small")
    assert result["meta"]["source"] == "hub"
    assert result["meta"]["hub_id"] == "nir/fc_small"
    assert result["meta"]["topology"] == "fc_small"


def test_perturbed_graph_is_rejected_by_stage(sandbox: Path) -> None:
    """A structurally perturbed artifact is not promoted and names fc1."""
    result = import_model(path=_graph_file(sandbox, perturb=True))
    assert result["promoted"] is False
    assert result["verdict"]["verdict"] == "incompatible"
    stages = {item["stage"] for item in result["verdict"]["mismatches"]}
    assert "fc1" in stages
    assert result["nir_ingestable"] is True


def test_framework_weights_are_rejected(sandbox: Path) -> None:
    """Opaque weights are rejected and are not NIR-ingestable."""
    path = sandbox / "blob.bin"
    path.write_bytes(b"xx")
    result = import_model(path=str(path))
    assert result["promoted"] is False
    assert result["nir_ingestable"] is False


def test_promote_false_skips_saving(sandbox: Path) -> None:
    """The funnel can classify without promoting when asked."""
    result = import_model("nir/fc_small", promote=False)
    assert result["verdict"]["verdict"] == "exact"
    assert result["promoted"] is False
    assert result["destination"] is None


def test_unknown_entry_raises() -> None:
    """An unknown catalog id is a typed error, not a silent miss."""
    with pytest.raises(HubArtifactError):
        import_model("does/not-exist")


def test_missing_path_raises() -> None:
    """A missing path is a typed error."""
    with pytest.raises(HubArtifactError):
        import_model(path="missing-artifact.pt")


def test_load_weights_from_graph_matches_module() -> None:
    """A graph's weights map onto a freshly built preset exactly."""
    spec, module = build_topology("fc_small")
    _other_spec, target = build_topology("fc_small")
    result = load_weights(target, graph=to_nir(spec, module))
    assert result.loaded is True
    expected = torch.as_tensor(to_nir(spec, module).nodes["fc1"].weight)
    assert torch.allclose(target.fc1.weight, expected)


def test_load_weights_reports_missing_key() -> None:
    """A state dict missing a weight is refused with the key named."""
    _spec, module = build_topology("fc_small")
    state = dict(module.state_dict())
    state.pop("fc2.weight")
    result = load_weights(module, state_dict=state)
    assert result.loaded is False
    assert "fc2.weight" in result.missing


def test_load_weights_reports_shape_mismatch() -> None:
    """A wrong-shaped tensor is refused rather than broadcast on load."""
    _spec, module = build_topology("fc_small")
    state = dict(module.state_dict())
    state["fc1.weight"] = torch.zeros(5, 784)
    result = load_weights(module, state_dict=state)
    assert result.loaded is False
    assert "fc1.weight" in result.mismatched


def test_load_weights_needs_exactly_one_source() -> None:
    """Supplying no source (or both) is a programming error."""
    _spec, module = build_topology("fc_small")
    with pytest.raises(ValueError):
        load_weights(module)


# --- integrity gate ---------------------------------------------------------


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
    assert len(found) == 6
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

"""Resolve complete Hugging Face snapshots before inspecting their files."""

from pathlib import Path

import pytest

from spikeforge_hub import cache
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError
from spikeforge_hub.inspect import (
    FRAMEWORK_WEIGHTS,
    inspect_artifact,
    resolve_path,
)


def _entry(source: str = "hf_repo") -> HubEntry:
    """Return an offline test entry for the requested remote source."""
    return HubEntry.from_dict({
        "id": "hf/example",
        "name": "Example snapshot",
        "framework": "hf",
        "kind": "framework_weights",
        "source": source,
        "hf_repo": "org/repo",
        "url": "https://example.test/model.safetensors",
        "license": "Apache-2.0",
        "notes": "unit-test entry",
    })


@pytest.mark.parametrize("nested", [False, True])
def test_snapshot_resolves_to_directory_and_lists_all_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, nested: bool,
) -> None:
    """Config files and nested weights belong to the same opaque snapshot."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    root = Path(cache.entry_dir(_entry().id))
    payload = root / "weights" if nested else root
    payload.mkdir(exist_ok=True)
    (payload / "config.json").write_text("{}", encoding="utf-8")
    (payload / "model.safetensors").write_bytes(b"weights")
    resolved = resolve_path(_entry())
    assert resolved == str(root)
    report = inspect_artifact(resolved)
    prefix = "weights/" if nested else ""
    assert report.kind == FRAMEWORK_WEIGHTS
    assert {record["key"] for record in report.keys} == {
        prefix + "config.json", prefix + "model.safetensors",
    }


def test_empty_snapshot_requires_a_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Creating empty directories does not make an HF artifact available."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    root = Path(cache.entry_dir(_entry().id))
    (root / "weights").mkdir()
    with pytest.raises(HubArtifactError, match="download first"):
        resolve_path(_entry())


def test_direct_url_still_resolves_to_its_cached_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A direct URL continues to resolve to a file rather than a snapshot."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    path = Path(cache.entry_dir(_entry().id)) / "model.safetensors"
    path.write_bytes(b"weights")
    assert resolve_path(_entry("url")) == str(path)

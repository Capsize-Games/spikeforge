"""Offline cache path resolution and size accounting."""

import os
from pathlib import Path

import pytest

from snn_interpreter import config
from snn_interpreter.hub import cache

_ROOT = Path(__file__).resolve().parent.parent


def test_cache_root_is_configurable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cache root follows the module's configured directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    assert cache.cache_root() == str(tmp_path)


def test_slug_flattens_a_catalog_id() -> None:
    """A ``framework/name`` id maps to a single safe directory name."""
    assert cache.slug("nir/conv_net") == "nir__conv_net"


def test_entry_dir_uses_a_slug_and_creates_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``entry_dir`` creates and returns the per-entry cache directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    path = cache.entry_dir("nir/conv_net")
    assert path == os.path.join(str(tmp_path), "nir__conv_net")
    assert os.path.isdir(path)


def test_entry_bytes_counts_cached_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cached artifact bytes are summed for progress reporting."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    path = cache.entry_dir("nir/conv_net")
    (Path(path) / "artifact.bin").write_bytes(b"x" * 10)
    assert cache.entry_bytes("nir/conv_net") == 10
    assert cache.cached("nir/conv_net") is True


def test_entry_bytes_is_zero_for_a_missing_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An uncached entry reports zero bytes and stays un-cached."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path))
    assert cache.entry_bytes("nir/absent") == 0
    assert cache.cached("nir/absent") is False


def test_dir_bytes_walks_nested_files(tmp_path: Path) -> None:
    """``dir_bytes`` sums every file under a nested directory."""
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "f.bin").write_bytes(b"12345")
    assert cache.dir_bytes(str(tmp_path)) == 5


def test_config_declares_the_hub_cache_dir() -> None:
    """``config`` exposes the hub cache dir and its env override."""
    source = (_ROOT / "snn_interpreter" / "config.py").read_text("utf-8")
    assert "SNN_HUB_DIR" in source
    assert "HUB_CACHE_DIR" in source


def test_config_hub_cache_defaults_under_data_dir() -> None:
    """Without an override the cache lives at ``DATA_DIR/hub``."""
    if os.environ.get("SNN_HUB_DIR"):
        pytest.skip("SNN_HUB_DIR is overridden in this environment")
    default = os.path.join(config.DATA_DIR, "hub")
    assert default == config.HUB_CACHE_DIR

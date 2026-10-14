"""The HSD suite fails loudly when its transitive h5py is gone (#121).

``h5py`` arrives only transitively through ``tonic``: if an upstream
release drops that dependency, ``test_hsd_reader.py`` would skip its
whole file and CI would stay green. This guard fails exactly there.
"""

import importlib

import pytest

from spikeforge.events import hsd_reader


def test_tonic_without_h5py_fails_instead_of_skipping() -> None:
    """With the events extra installed, h5py must be importable too."""
    pytest.importorskip("tonic", reason="needs the events extra")
    try:
        importlib.import_module("h5py")
    except ImportError:
        pytest.fail(
            "tonic is installed but h5py is missing: the HSD reader "
            "tests would skip silently (see issue #121)"
        )
    assert hsd_reader.available() is True

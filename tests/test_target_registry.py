"""Registry availability for the built-in deployment targets."""

import importlib
import json
import sys

import pytest

import spikeforge_targets as targets
from spikeforge_targets import (
    EMITTED_PRIMITIVES,
    capability_matrix,
    catalog,
    probe,
    registry,
    report,
)

_OPTIONAL_MODULES = ("lava", "spinnaker2", "sinabs", "rockpool", "norse")
_OPTIONAL_TARGETS = (
    "lava_loihi2",
    "spinnaker2",
    "speck",
    "xylo",
    "norse",
)


def _hide(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every optional target SDK appear absent."""
    for name in _OPTIONAL_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def test_reference_target_is_always_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference interpreter needs no SDK and is always available."""
    _hide(monkeypatch)
    assert registry.available("reference") is True
    assert "reference" in registry.available_names()


def test_reference_declares_every_emitted_primitive() -> None:
    """The reference target supports each primitive the mapper can emit."""
    spec = registry.get_target("reference")
    assert spec.kind == "reference"
    assert spec.extra is None
    assert spec.supported == EMITTED_PRIMITIVES


def test_optional_targets_unavailable_without_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional targets report unavailable while their SDK is absent."""
    _hide(monkeypatch)
    for name in _OPTIONAL_TARGETS:
        assert registry.available(name) is False
    assert registry.available_names() == ["reference"]
    assert set(registry.unavailable_names()) == set(_OPTIONAL_TARGETS)


def test_optional_targets_name_a_known_extra() -> None:
    """Every non-reference target points at a declared pip extra."""
    for name in registry.target_names():
        spec = registry.get_target(name)
        if spec.kind == "reference":
            assert spec.extra is None
        else:
            assert spec.extra in probe.EXTRA_MODULES


def test_importing_targets_does_not_probe_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-importing with every SDK hidden must not raise or probe."""
    _hide(monkeypatch)
    for module in (catalog, registry, capability_matrix, report, targets):
        importlib.reload(module)
    assert registry.available("reference") is True
    for name in _OPTIONAL_TARGETS:
        assert registry.available(name) is False


def test_probe_reports_unknown_extra_unavailable() -> None:
    """An extra the probe does not know is reported unavailable."""
    assert probe.extra_available("does_not_exist") is False
    assert probe.extra_available(None) is True


def test_unknown_target_raises_value_error() -> None:
    """Looking up an unregistered target raises ``ValueError``."""
    with pytest.raises(ValueError, match="unknown target"):
        registry.get_target("no_such_target")


def test_target_spec_dict_is_json_serialisable() -> None:
    """Every built-in target serialises to a plain JSON dict."""
    for name in registry.target_names():
        payload = registry.get_target(name).to_dict()
        assert json.dumps(payload)
        assert isinstance(payload["supported"], list)
        assert set(payload["constraints"]) == {
            "dtype",
            "timestep_ms",
            "quantization",
        }

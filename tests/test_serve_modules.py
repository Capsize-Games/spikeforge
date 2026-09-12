"""Install/list/uninstall a .spkf as a named module, and run it one-shot."""

import io
import json
from pathlib import Path
from typing import Any, List

import pytest

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.serving.errors import BundleFormatError
from spikeforge.topology import registry
from spikeforge_serve import modules
from spikeforge_serve.module_runner import resolve_bundle
from spikeforge_serve.module_runner import run as run_module

_TOPOLOGY = "fc_small"
_PARAMS = {"hidden": 5, "num_classes": 3}
_NUM_STEPS = 4


def _write_bundle(path: Path) -> None:
    """Write a small, deterministic ``.spkf`` bundle for the tests."""
    spec, module = registry.build_topology(_TOPOLOGY, dict(_PARAMS))
    module.eval()
    encode = EncodeSpec(coding="latency", num_steps=_NUM_STEPS)
    bundle = DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
            "topology": _TOPOLOGY,
            "topology_params": dict(_PARAMS),
            "encode_spec_version": ENCODE_SPEC_VERSION,
            "num_classes": _PARAMS["num_classes"],
            "label_map": {"0": "a", "1": "b", "2": "c"},
            "expected_metrics": {"test_accuracy": 0.9},
        },
        weights=module.state_dict(),
        encode_config=encode.to_dict(),
    )
    bundle.save(str(path))


def _sample() -> List[Any]:
    """Return one already-encoded frame shaped for ``fc_small``'s input."""
    return [0.0] * 784


def _request_file(tmp_path: Path, name: str = "request.json") -> str:
    """Write a one-frame, pre-encoded predict request; return its path."""
    path = tmp_path / name
    path.write_text(json.dumps({"frames": [_sample()], "encoded": True}))
    return str(path)


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point installs at a throwaway directory instead of the real home."""
    monkeypatch.setattr(modules, "MODULES_HOME", tmp_path / "modules")
    monkeypatch.setattr(modules, "BIN_HOME", tmp_path / "bin")


def test_install_writes_descriptor_and_wrapper(bundle_path: str) -> None:
    """Installing copies the bundle, writes module.json, and a wrapper."""
    target = modules.install(bundle_path, name="digits")

    assert (target / "model.spkf").exists()
    descriptor = json.loads((target / "module.json").read_text())
    assert descriptor["name"] == "digits"
    assert descriptor["topology"] == _TOPOLOGY
    assert descriptor["num_classes"] == _PARAMS["num_classes"]
    assert descriptor["label_map"] == {"0": "a", "1": "b", "2": "c"}

    wrapper = modules.BIN_HOME / "digits"
    assert wrapper.exists()
    assert 'spikeforge_serve run "digits"' in wrapper.read_text()
    assert wrapper.stat().st_mode & 0o111  # executable bit set


def test_install_defaults_name_to_the_bundle_stem(bundle_path: str) -> None:
    """An omitted --name falls back to the bundle file's stem."""
    target = modules.install(bundle_path)
    assert target.name == "model"


def test_install_rejects_unsafe_characters_in_name(bundle_path: str) -> None:
    """A name is sanitised to filesystem/shell-safe characters."""
    target = modules.install(bundle_path, name="a b/c")
    assert target.name == "a_b_c"


def test_install_refuses_a_corrupt_bundle(tmp_path: Path) -> None:
    """A bundle failing strict loading is refused before writing anything."""
    bad = tmp_path / "bad.spkf"
    bad.write_bytes(b"not a zip")
    with pytest.raises(BundleFormatError):
        modules.install(str(bad), name="bad")
    assert not modules.module_dir("bad").exists()


def test_list_modules_is_empty_then_reflects_installs(
    bundle_path: str,
) -> None:
    """An empty install tree lists nothing; installed modules show up."""
    assert modules.list_modules() == []
    modules.install(bundle_path, name="digits")
    listed = modules.list_modules()
    assert len(listed) == 1
    assert listed[0]["name"] == "digits"


def test_uninstall_removes_directory_and_wrapper(bundle_path: str) -> None:
    """Uninstalling deletes the module dir and its wrapper script."""
    modules.install(bundle_path, name="digits")
    assert modules.uninstall("digits") is True
    assert not modules.module_dir("digits").exists()
    assert not (modules.BIN_HOME / "digits").exists()


def test_uninstall_a_missing_module_reports_false() -> None:
    """Uninstalling a name never installed is a no-op, not an error."""
    assert modules.uninstall("nope") is False


def test_resolve_bundle_prefers_an_installed_module(bundle_path: str) -> None:
    """A name that matches an installed module resolves to its copy."""
    target = modules.install(bundle_path, name="digits")
    assert resolve_bundle("digits") == str(target / "model.spkf")


def test_resolve_bundle_falls_back_to_a_literal_path(bundle_path: str) -> None:
    """A name with no matching install is returned unchanged, as a path."""
    assert resolve_bundle(bundle_path) == bundle_path


def test_run_module_reads_from_file(tmp_path: Path, bundle_path: str) -> None:
    """``--file`` reads the request instead of stdin."""
    request_path = _request_file(tmp_path)
    out = io.StringIO()

    exit_code = run_module(bundle_path, input_file=request_path, out=out)

    assert exit_code == 0
    result = json.loads(out.getvalue())
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert "predicted" in prediction
    assert prediction["steps"] == 1


def test_run_module_resolves_an_installed_name(
    tmp_path: Path, bundle_path: str
) -> None:
    """Running by installed name works the same as running by path."""
    modules.install(bundle_path, name="digits")
    request_path = _request_file(tmp_path)
    out = io.StringIO()

    exit_code = run_module("digits", input_file=request_path, out=out)

    assert exit_code == 0
    assert json.loads(out.getvalue())["predictions"]


def test_run_module_rejects_empty_input(
    tmp_path: Path, bundle_path: str
) -> None:
    """An empty request file is a clear error, not a JSON parse traceback."""
    request_path = tmp_path / "empty.json"
    request_path.write_text("")
    with pytest.raises(ValueError):
        run_module(
            bundle_path, input_file=str(request_path), out=io.StringIO()
        )

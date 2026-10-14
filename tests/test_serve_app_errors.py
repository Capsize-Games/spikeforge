"""Error mapping for the headless ``spikeforge-serve`` service.

Bad bundles map to 404/422/409 with typed error payloads, and bad
request bodies are 400s -- asserted through the same shared ASGI
driver as the happy-path app tests.
"""

import json
import platform
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict

import pytest
from _fakes import _get_json, _post_json
from _serve_helpers import _write_bundle

from spikeforge.serving import bundle_manifest as bm
from spikeforge_serve import create_app


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


def _rewrite_manifest(
    path: Path, mutate: Callable[[Dict[str, Any]], None]
) -> None:
    """Rewrite a bundle's manifest, then re-sign its checksums."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries[bm.MANIFEST_NAME].decode("utf-8"))
    mutate(manifest)
    entries[bm.MANIFEST_NAME] = bm.dump_json(manifest)
    payloads = {
        name: value
        for name, value in entries.items()
        if name != bm.SUMS_NAME
    }
    entries[bm.SUMS_NAME] = bm.sums_text(payloads).encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)


def test_missing_bundle_maps_to_404(tmp_path: Path) -> None:
    """A missing bundle is a 404 and leaves the service unready."""
    app = create_app(str(tmp_path / "absent.spkf"))
    status, payload = _get_json(app, "/v1/bundle")
    assert status == 404
    assert payload["error"]["type"] == "bundle_not_found"
    assert _get_json(app, "/health")[0] == 200
    assert _get_json(app, "/ready")[0] == 503


def test_malformed_bundle_maps_to_422(tmp_path: Path) -> None:
    """A file that is not a zip bundle is an unprocessable entity."""
    path = tmp_path / "broken.spkf"
    path.write_bytes(b"not a zip archive")
    status, payload = _get_json(create_app(str(path)), "/v1/bundle")
    assert status == 422
    assert payload["error"]["type"] == "bundle_format"


def test_tampered_bundle_maps_to_422_integrity(tmp_path: Path) -> None:
    """A bundle whose contents do not match its checksums is refused."""
    path = tmp_path / "tampered.spkf"
    _write_bundle(path)
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries[bm.WEIGHTS_NAME] = entries[bm.WEIGHTS_NAME] + b"tampered"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    status, payload = _get_json(create_app(str(path)), "/v1/bundle")
    assert status == 422
    assert payload["error"]["type"] == "bundle_integrity"


def test_incompatible_bundle_maps_to_409(tmp_path: Path) -> None:
    """A bundle built by an incompatible runtime is a conflict."""
    path = tmp_path / "incompatible.spkf"
    _write_bundle(path)

    def _downgrade(manifest: Dict[str, Any]) -> None:
        """Record an impossible torch version against the manifest."""
        manifest["library_versions"] = {
            "torch": "0.0",
            "python": platform.python_version(),
        }

    _rewrite_manifest(path, _downgrade)
    status, payload = _get_json(create_app(str(path)), "/v1/bundle")
    assert status == 409
    assert payload["error"]["type"] == "bundle_compatibility"


def test_predict_rejects_a_bad_body(bundle_path: str) -> None:
    """A request without a non-empty frame list is a 400."""
    app = create_app(bundle_path)
    assert _post_json(app, "/v1/predict", {})[0] == 400
    assert _post_json(app, "/v1/predict", {"frames": []})[0] == 400
    assert _post_json(app, "/v1/predict", {"frames": "nope"})[0] == 400

"""The service, runner, CLI, and app thread the backend selection.

The serving layers add no backend logic of their own: they hand the
``(backend, c_lib)`` pair to the session, which validates it. These
tests pin that threading — including that ``run --backend nir-c``
reaches the export gate — without executing the C library.
"""

import io
from pathlib import Path

import pytest
import torch
from _c_backend_helpers import (
    _fc_bundle,
    _linear_bundle,
    _write_bundle,
    _write_request,
)

from spikeforge.serving.c_runtime_env import LIB_ENV_VAR
from spikeforge.serving.errors import (
    BackendError,
    CExportError,
    CRuntimeLibraryError,
)
from spikeforge.serving.inference_backend import NIR_C_BACKEND, TORCH_BACKEND
from spikeforge_serve.__main__ import main as serve_main
from spikeforge_serve.app import create_app
from spikeforge_serve.module_runner import run as run_module
from spikeforge_serve.service import ServingService


def test_service_default_backend_is_torch() -> None:
    """A service steps through torch unless asked otherwise."""
    assert ServingService(_fc_bundle()).backend == TORCH_BACKEND


def test_service_with_unknown_backend_refuses_on_first_use() -> None:
    """A misconfigured service fails when its first session opens."""
    service = ServingService(_fc_bundle(), backend="bogus")
    with pytest.raises(BackendError):
        service.predict([[0.0]])


def test_service_threads_backend_and_c_lib() -> None:
    """The service hands its backend pair to every session it opens."""
    pytest.importorskip("nir")
    service = ServingService(
        _fc_bundle(), backend=NIR_C_BACKEND, c_lib="/bogus/x.so"
    )
    assert service.session().backend_name == NIR_C_BACKEND
    frame = torch.zeros(1, 784).tolist()
    with pytest.raises(CRuntimeLibraryError) as caught:
        service.predict([frame], encoded=True)
    assert caught.value.path == "/bogus/x.so"


def test_run_passes_backend_selection_through(tmp_path: Path) -> None:
    """``run`` with an unknown backend fails before touching the graph."""
    bundle_path = _write_bundle(tmp_path, _fc_bundle())
    request = _write_request(tmp_path, [[0.0] * 784])
    with pytest.raises(BackendError):
        run_module(bundle_path, input_file=request, backend="bogus")


def test_run_reaches_the_export_gate(tmp_path: Path) -> None:
    """``run --backend nir-c`` refuses a C-rejected bundle by kind."""
    pytest.importorskip("nir")
    bundle_path = _write_bundle(tmp_path, _linear_bundle())
    request = _write_request(tmp_path, [[[0.0] * 4]])
    with pytest.raises(CExportError) as caught:
        run_module(
            bundle_path, input_file=request, backend=NIR_C_BACKEND,
            out=io.StringIO(),
        )
    assert caught.value.kinds == ("Linear",)


def test_cli_run_accepts_a_backend_flag(tmp_path: Path) -> None:
    """The CLI serves the default backend and rejects an unknown choice.

    ``argparse`` guards the flag's choices, so an unknown backend name
    can never reach the service from the command line.
    """
    bundle_path = _write_bundle(tmp_path, _fc_bundle())
    request = _write_request(tmp_path, [[0.0] * 784])
    assert serve_main(["run", bundle_path, "--file", request]) == 0
    with pytest.raises(SystemExit):
        serve_main(
            ["run", bundle_path, "--file", request, "--backend", "wat"]
        )


def test_cli_run_reaches_the_export_gate(tmp_path: Path) -> None:
    """``run --backend nir-c`` refuses a C-rejected bundle by kind."""
    pytest.importorskip("nir")
    bundle_path = _write_bundle(tmp_path, _linear_bundle())
    request = _write_request(tmp_path, [[[0.0] * 4]])
    with pytest.raises(CExportError) as caught:
        serve_main(
            ["run", bundle_path, "--file", request, "--backend", "nir-c"]
        )
    assert caught.value.kinds == ("Linear",)


def test_create_app_threads_the_backend() -> None:
    """The HTTP app serves through the backend it was given."""
    app = create_app(_fc_bundle(), backend=NIR_C_BACKEND)
    assert app.state.serving.backend == NIR_C_BACKEND


@pytest.fixture(autouse=True)
def _clean_library_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run without an ambient library path, whatever the machine has."""
    monkeypatch.delenv(LIB_ENV_VAR, raising=False)

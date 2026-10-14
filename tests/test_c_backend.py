"""Backend selection, capability gating, and typed session errors.

Nothing here executes the C library: backend misuse is validated in
pure Python before the library loads, and unsupported topologies are
refused by the committed export gate. Tests that do execute C live in
``test_c_backend_parity.py`` and skip without the library, and the
service/CLI threading lives in ``test_c_backend_serve.py``. Tests
that export a graph need the ``nir`` extra and skip without it; the
pure selection tests run on torch alone.
"""

import pytest
import torch
from _c_backend_helpers import (
    _conv_bundle,
    _fc_bundle,
    _linear_bundle,
    _spike_frames,
)

from spikeforge.serving.c_runtime_env import LIB_ENV_VAR
from spikeforge.serving.c_runtime_lib import NirCLibrary
from spikeforge.serving.errors import (
    BackendError,
    CExportError,
    CRuntimeLibraryError,
    StateError,
)
from spikeforge.serving.inference_backend import (
    NIR_C_BACKEND,
    TORCH_BACKEND,
    resolve_backend,
)
from spikeforge.serving.session import InferenceSession


def _nir_session() -> InferenceSession:
    """Return a ``nir-c`` session with the library lookup disabled.

    The module autouse fixture clears ``NIR_C_RUNTIME_LIB``, so these
    sessions never find a library even where one is installed.
    """
    pytest.importorskip("nir")
    return InferenceSession.load(_conv_bundle(), backend=NIR_C_BACKEND)


def test_default_backend_is_torch() -> None:
    """A session steps through torch unless asked otherwise."""
    assert InferenceSession.load(_fc_bundle()).backend_name == TORCH_BACKEND


def test_resolve_backend_names_both_backends() -> None:
    """Both backend names resolve; anything else names them in refusal."""
    assert resolve_backend(TORCH_BACKEND) == TORCH_BACKEND
    assert resolve_backend(NIR_C_BACKEND) == NIR_C_BACKEND
    with pytest.raises(BackendError) as caught:
        resolve_backend("bogus")
    assert TORCH_BACKEND in str(caught.value)
    assert NIR_C_BACKEND in str(caught.value)


def test_unknown_backend_is_refused_before_any_export() -> None:
    """An unknown backend fails without needing the ``nir`` extra."""
    with pytest.raises(BackendError):
        InferenceSession.load(_conv_bundle(), backend="bogus")


def test_nir_c_backend_is_selected_by_name() -> None:
    """``backend="nir-c"`` opens a C session over the bundle's spec."""
    pytest.importorskip("nir")
    bundle = _conv_bundle()
    session = InferenceSession.load(bundle, backend=NIR_C_BACKEND)
    assert session.backend_name == NIR_C_BACKEND
    assert session.spec.to_dict() == bundle.spec.to_dict()
    assert session.steps == 0


def test_nir_c_rejects_a_torch_post_step_hook() -> None:
    """A torch step transform cannot apply to C execution."""
    with pytest.raises(BackendError, match="post_step"):
        InferenceSession.load(
            _conv_bundle(), backend=NIR_C_BACKEND,
            post_step=lambda outputs, state: (outputs, state),
        )


def test_nir_c_rejects_a_passed_module() -> None:
    """The C backend builds its own graph, so a module contradicts it."""
    bundle = _conv_bundle()
    with pytest.raises(BackendError, match="no module"):
        InferenceSession(
            bundle, bundle.build_module(), backend=NIR_C_BACKEND
        )


def test_torch_backend_needs_its_module() -> None:
    """The torch backend cannot step without the bundle's module."""
    with pytest.raises(BackendError, match="module"):
        InferenceSession(_conv_bundle(), backend=TORCH_BACKEND)


def test_unsupported_topology_names_its_kinds() -> None:
    """A C-rejected topology fails typed, naming ``Linear`` precisely."""
    pytest.importorskip("nir")
    with pytest.raises(CExportError) as caught:
        InferenceSession.load(_linear_bundle(), backend=NIR_C_BACKEND)
    assert caught.value.kinds == ("Linear",)


def test_first_step_without_a_library_names_the_build() -> None:
    """A missing library fails typed, explaining how to build it."""
    pytest.importorskip("nir")
    session = InferenceSession.load(_conv_bundle(), backend=NIR_C_BACKEND)
    with pytest.raises(CRuntimeLibraryError) as caught:
        session.step(_spike_frames()[0])
    assert caught.value.path is None
    assert LIB_ENV_VAR in str(caught.value)
    assert "cc -shared" in str(caught.value)


def test_missing_library_path_is_refused_with_its_path() -> None:
    """A library path naming nothing is refused before any C call."""
    with pytest.raises(CRuntimeLibraryError) as caught:
        NirCLibrary("/nonexistent/libnir_c_runtime.so")
    assert caught.value.path == "/nonexistent/libnir_c_runtime.so"


def test_batch_frames_are_refused_before_the_library_loads() -> None:
    """A batch of 2 is refused as a value, not a C shape failure."""
    session = _nir_session()
    frame = torch.zeros(2, 1, 8, 8)
    with pytest.raises(ValueError, match="batch of 2"):
        session.step(frame)


def test_flat_frame_without_a_batch_dim_is_refused() -> None:
    """The C path requires the leading batch dim the torch path needs."""
    session = _nir_session()
    with pytest.raises(ValueError, match="B, ..."):
        session.step(torch.zeros(8))


def test_float64_frame_is_refused() -> None:
    """The C runtime consumes float32, so anything else is refused."""
    session = _nir_session()
    with pytest.raises(ValueError, match="float32"):
        session.step(torch.zeros(1, 1, 8, 8, dtype=torch.float64))


def test_four_data_dims_are_refused() -> None:
    """A frame wider than 3 data dims cannot enter a C tensor."""
    session = _nir_session()
    with pytest.raises(ValueError, match="1 to 3 data dims"):
        session.step(torch.zeros(1, 2, 2, 2, 2))


def test_c_state_capture_and_restore_are_refused() -> None:
    """Native C state has no serializable form, so both directions refuse."""
    session = _nir_session()
    with pytest.raises(StateError):
        session.state()
    with pytest.raises(StateError):
        session.load_state({})


def test_c_close_is_terminal() -> None:
    """A closed C backend refuses work but tolerates another close."""
    session = _nir_session()
    session.close()
    session.close()
    with pytest.raises(BackendError, match="closed"):
        session.step(_spike_frames()[0])
    with pytest.raises(BackendError, match="closed"):
        session.reset()


def test_torch_close_is_terminal() -> None:
    """Closing a torch session retires it exactly like a C session."""
    session = InferenceSession.load(_fc_bundle())
    session.close()
    with pytest.raises(BackendError, match="closed"):
        session.step(torch.zeros(1, 784))
    with pytest.raises(BackendError, match="closed"):
        session.reset()


def test_session_context_manager_closes_on_exit() -> None:
    """Leaving a ``with`` session block closes the backend."""
    with InferenceSession.load(_fc_bundle()) as session:
        session.step(torch.zeros(1, 784))
    with pytest.raises(BackendError, match="closed"):
        session.step(torch.zeros(1, 784))


@pytest.fixture(autouse=True)
def _clean_library_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run library-missing tests without an ambient library path."""
    monkeypatch.delenv(LIB_ENV_VAR, raising=False)

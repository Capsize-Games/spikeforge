"""The ctypes binding against a stub shared library.

``tests/stub_nir_c_runtime.c`` implements the eight C functions the
binding calls with the real header layouts, so these tests exercise
the actual marshaling — struct fields, the error-string protocol,
and the load/run/reset/close lifecycle — with no ``nir-c-runtime``
checkout. The stub is compiled with the system ``cc``; the tests
skip where no compiler exists.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import pytest
import torch

from spikeforge.serving.c_runtime_env import (
    LIB_ENV_VAR,
    resolve_lib_path,
)
from spikeforge.serving.c_runtime_lib import NirCLibrary
from spikeforge.serving.errors import (
    CRuntimeExecutionError,
    CRuntimeLibraryError,
)
from spikeforge.serving.nir_c_backend import NirCBackend
from spikeforge.topology import registry

#: The stub sources, next to this file.
_STUB_C = str(Path(__file__).resolve().parent / "stub_nir_c_runtime.c")


def _stub_lib(tmp_path: Path) -> str:
    """Compile the stub library; skip the test without a compiler."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler builds the stub library")
    target = str(tmp_path / "libstub.so")
    subprocess.run(
        [compiler, "-shared", "-fPIC", "-O2", "-o", target, _STUB_C],
        check=True,
        capture_output=True,
    )
    return target


def _loaded(tmp_path: Path, graph: str = "{}") -> NirCLibrary:
    """Return a stub-backed library with ``graph`` loaded."""
    library = NirCLibrary(_stub_lib(tmp_path))
    library.load_graph(graph)
    return library


def _frame(*values: float) -> Tuple[int, Tuple[int, ...], object]:
    """Return ``(address, shape, keeper)`` for ``values`` as float32.

    The keeper holds the buffer alive; the caller keeps it until the
    C call using the address returns.
    """
    buffer = (ctypes.c_float * len(values))(*values)
    return ctypes.addressof(buffer), (len(values),), buffer


def test_run_frame_copies_readout_and_carries_state(
    tmp_path: Path,
) -> None:
    """Steps return the stub readout and advance its carried state."""
    library = _loaded(tmp_path)
    data, shape, _keep = _frame(0.5, 1.5)
    assert library.run_frame(data, shape) == ([1.0, 2.0], (2,))
    data, shape, _keep = _frame(0.0, 0.0)
    assert library.run_frame(data, shape) == ([2.0, 3.0], (2,))


def test_run_frame_refuses_a_bad_shape(tmp_path: Path) -> None:
    """A 0-D or 4-D shape is refused before any C call."""
    library = _loaded(tmp_path)
    data, _, _keep = _frame(1.0)
    with pytest.raises(ValueError, match="dense dims"):
        library.run_frame(data, ())
    with pytest.raises(ValueError, match="dense dims"):
        library.run_frame(data, (1, 1, 1, 1))


def test_parse_failure_reports_the_c_message(tmp_path: Path) -> None:
    """Invalid JSON fails typed with the runtime's own message."""
    library = NirCLibrary(_stub_lib(tmp_path))
    with pytest.raises(CRuntimeExecutionError) as caught:
        library.load_graph('{"x": STUB_PARSE_FAIL}')
    assert caught.value.operation == "parse"
    assert caught.value.detail == "C runtime parse failed: stub: bad json"
    assert library.loaded is False


def test_load_failure_names_the_rejected_kind(tmp_path: Path) -> None:
    """A rejected graph fails typed and leaves nothing live."""
    library = NirCLibrary(_stub_lib(tmp_path))
    with pytest.raises(CRuntimeExecutionError) as caught:
        library.load_graph("STUB_LOAD_FAIL")
    assert caught.value.operation == "load"
    assert "'Nope'" in caught.value.detail
    assert library.loaded is False


def test_init_failure_leaves_nothing_live(tmp_path: Path) -> None:
    """A failed init releases the graph it had just loaded."""
    library = NirCLibrary(_stub_lib(tmp_path))
    with pytest.raises(CRuntimeExecutionError) as caught:
        library.load_graph("STUB_INIT_FAIL")
    assert caught.value.operation == "init"
    assert library.loaded is False


def test_run_failure_reports_the_c_message(tmp_path: Path) -> None:
    """A failed step fails typed with the runtime's own message."""
    library = _loaded(tmp_path)
    buffer = (ctypes.c_float * 1)(float("nan"))
    with pytest.raises(CRuntimeExecutionError) as caught:
        library.run_frame(ctypes.addressof(buffer), (1,))
    assert caught.value.operation == "run"
    assert caught.value.detail == "C runtime run failed: stub: nan input"


def test_reset_reinitializes_the_carried_state(tmp_path: Path) -> None:
    """Resetting restarts the stub's step counter; reset early is free."""
    fresh = NirCLibrary(_stub_lib(tmp_path))
    fresh.reset()
    assert fresh.loaded is False
    library = _loaded(tmp_path)
    data, shape, _keep = _frame(0.0)
    library.run_frame(data, shape)
    library.reset()
    assert library.loaded is True
    data, shape, _keep = _frame(0.0)
    assert library.run_frame(data, shape) == ([1.0, 2.0], (2,))


def test_close_releases_and_refuses_further_steps(
    tmp_path: Path,
) -> None:
    """Closing twice is safe; stepping after needs a reload."""
    library = _loaded(tmp_path)
    assert library.loaded is True
    library.close()
    library.close()
    assert library.loaded is False
    data, shape, _keep = _frame(0.0)
    with pytest.raises(CRuntimeExecutionError, match="no graph"):
        library.run_frame(data, shape)


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    """Leaving a ``with`` block releases the graph and runtime."""
    with NirCLibrary(_stub_lib(tmp_path)) as library:
        library.load_graph("{}")
        assert library.loaded is True
    assert library.loaded is False


def test_reload_releases_the_previous_graph(tmp_path: Path) -> None:
    """Loading twice reloads cleanly, restarting the carried state."""
    library = _loaded(tmp_path)
    data, shape, _keep = _frame(0.0)
    library.run_frame(data, shape)
    library.load_graph("{}")
    data, shape, _keep = _frame(0.0)
    assert library.run_frame(data, shape) == ([1.0, 2.0], (2,))


def test_missing_libc_reads_no_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a libc free the calls pass NULL and stay typed."""
    monkeypatch.setattr(
        "ctypes.util.find_library", lambda name: None
    )
    library = NirCLibrary(_stub_lib(tmp_path))
    with pytest.raises(CRuntimeExecutionError) as caught:
        library.load_graph('{"x": STUB_PARSE_FAIL}')
    assert caught.value.operation == "parse"
    assert caught.value.detail == "C runtime parse failed: unknown error"


def test_resolve_lib_path_prefers_the_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit path wins; else the env var; else the empty probe."""
    monkeypatch.setenv(LIB_ENV_VAR, "/env/lib.so")
    assert resolve_lib_path("/x/lib.so") == "/x/lib.so"
    assert resolve_lib_path() == "/env/lib.so"
    monkeypatch.delenv(LIB_ENV_VAR)
    assert resolve_lib_path() == ""


def test_open_refuses_a_file_the_loader_rejects(tmp_path: Path) -> None:
    """A path naming a non-library fails with the loader's refusal."""
    bad = tmp_path / "bad.so"
    bad.write_text("not a shared library")
    with pytest.raises(CRuntimeLibraryError) as caught:
        NirCLibrary(str(bad))
    assert caught.value.path == str(bad)


def _backend(tmp_path: Path, graph: str = "{}") -> NirCBackend:
    """Return a stub-backed backend over a conv_net spec, unloaded."""
    params = {"in_channels": 1, "channels": 2, "input_size": 8}
    spec, _ = registry.build_topology("conv_net", params)
    return NirCBackend(graph, spec, lib=_stub_lib(tmp_path))


def test_backend_steps_through_the_stub(tmp_path: Path) -> None:
    """The backend's tensor boundary works end to end against C calls."""
    backend = _backend(tmp_path)
    first, _ = backend.step(torch.zeros(1, 2))
    assert first.tolist() == [[1.0, 2.0]]
    second, spikes = backend.step(torch.zeros(1, 2))
    assert second.tolist() == [[2.0, 3.0]]
    assert spikes == {}
    backend.reset()
    third, _ = backend.step(torch.zeros(1, 2))
    assert third.tolist() == [[1.0, 2.0]]
    backend.close()


def test_backend_reports_a_c_load_failure(tmp_path: Path) -> None:
    """A graph C rejects fails the first step typed, staying unloaded."""
    backend = _backend(tmp_path, "STUB_LOAD_FAIL")
    with pytest.raises(CRuntimeExecutionError) as caught:
        backend.step(torch.zeros(1, 2))
    assert caught.value.operation == "load"
    backend.close()

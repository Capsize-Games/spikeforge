"""C-vs-torch readout parity on the deterministic conv_net fixture.

Every test executes the real shared library and skips without it: set
``NIR_C_RUNTIME_LIB`` to a built ``libnir_c_runtime.so`` (one ``cc``
command; see ``documentation/model-deployment.md``). The tolerance is
the documented acceptance bar, not a fit: ``1e-3`` readout max-abs,
the same bar ``nir-c-runtime`` uses in its own end-to-end test (see
``plans/c_package_contract.md`` under "Tolerances").
"""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from _c_backend_helpers import _STEPS, _conv_bundle, _spike_frames

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.serving.c_runtime_env import LIB_ENV_VAR
from spikeforge.serving.inference_backend import NIR_C_BACKEND
from spikeforge.serving.session import InferenceSession
from spikeforge_serve.module_runner import run as run_module

pytest.importorskip("nir")

#: Readout max-abs bar both backends must meet; see the module docstring.
READOUT_TOL = 1e-3


def _lib() -> str:
    """Return the C library path, skipping the test without one."""
    path = os.environ.get(LIB_ENV_VAR, "")
    if not path or not os.path.isfile(path):
        pytest.skip(f"set {LIB_ENV_VAR} to libnir_c_runtime.so for C parity")
    return path


def _close(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Return True when two readouts agree within the readout bar."""
    return bool((((got - want).abs().max()) <= READOUT_TOL).item())


def _tensor(payload: Dict[str, Any]) -> torch.Tensor:
    """Return a ``tensor_json`` payload as a float tensor."""
    return torch.tensor(payload["values"], dtype=torch.float32)


def test_c_readout_matches_torch_per_step() -> None:
    """Per-step logits, totals, and labels agree within tolerance."""
    lib = _lib()
    frames = _spike_frames()
    bundle = _conv_bundle()
    with InferenceSession.load(bundle) as torch_session:
        expected = [torch_session.step(frame) for frame in frames]
    with InferenceSession.load(
        bundle, backend=NIR_C_BACKEND, c_lib=lib
    ) as actual_session:
        actual = [actual_session.step(frame) for frame in frames]
    peak = 0.0
    for want, got in zip(expected, actual):
        assert got.steps == want.steps
        assert got.label == want.label
        assert _close(got.logits, want.logits)
        assert _close(got.class_totals, want.class_totals)
        peak = max(peak, float(want.logits.abs().max()))
    assert peak > 0.0


def test_c_matches_the_reference_interpreter() -> None:
    """The C mean readout matches ``NirInterpreter`` on the same graph."""
    lib = _lib()
    bundle = _conv_bundle()
    module = bundle.build_module()
    frames = _spike_frames()
    want = NirInterpreter(to_nir(bundle.spec, module)).run(frames).readout
    with InferenceSession.load(
        bundle, backend=NIR_C_BACKEND, c_lib=lib
    ) as session:
        final = None
        for frame in frames:
            final = session.step(frame)
        assert final is not None
        assert _close(final.mean_logits, want)


def test_c_reset_restarts_the_stream() -> None:
    """Resetting clears C temporal state, not just the counters."""
    lib = _lib()
    frames = _spike_frames()
    bundle = _conv_bundle()
    with InferenceSession.load(
        bundle, backend=NIR_C_BACKEND, c_lib=lib
    ) as session:
        for frame in frames:
            session.step(frame)
        session.reset()
        assert session.steps == 0
        replayed = [session.step(frame).logits for frame in frames[:3]]
    with InferenceSession.load(
        bundle, backend=NIR_C_BACKEND, c_lib=lib
    ) as fresh:
        expected = [fresh.step(frame).logits for frame in frames[:3]]
    for want, got in zip(expected, replayed):
        assert _close(got, want)


def _run_json(
    bundle_path: str, request: str, backend: str, lib: str
) -> List[Dict[str, Any]]:
    """Serve one request file through ``backend``; return predictions."""
    out = io.StringIO()
    run_module(
        bundle_path, input_file=request, out=out, backend=backend,
        c_lib=lib,
    )
    return json.loads(out.getvalue())["predictions"]


def _payload_peak(payload: List[Dict[str, Any]]) -> float:
    """Return the largest readout magnitude across ``payload``."""
    peak = 0.0
    for prediction in payload:
        for row in prediction["logits"]["values"]:
            peak = max(peak, max(abs(value) for value in row))
    return peak


def _assert_payloads_close(
    got: List[Dict[str, Any]], want: List[Dict[str, Any]]
) -> None:
    """Assert two prediction payloads agree within the readout bar."""
    assert len(got) == len(want) == _STEPS
    for actual, expected in zip(got, want):
        assert actual["steps"] == expected["steps"]
        assert actual["label"] == expected["label"]
        assert actual["spikes"] == {}
        logits = _tensor(actual["logits"])
        assert _close(logits, _tensor(expected["logits"]))
        totals = _tensor(actual["class_totals"])
        assert _close(totals, _tensor(expected["class_totals"]))


def test_run_serves_a_bundle_through_c_end_to_end(tmp_path: Path) -> None:
    """``run --backend nir-c`` answers the same JSON as torch, from C."""
    lib = _lib()
    bundle_path = str(tmp_path / "model.spkf")
    _conv_bundle().save(bundle_path)
    request_path = tmp_path / "request.json"
    frames = [frame.tolist() for frame in _spike_frames()]
    request_path.write_text(json.dumps({"frames": frames, "encoded": True}))
    request = str(request_path)
    torch_payload = _run_json(bundle_path, request, "torch", lib)
    c_payload = _run_json(bundle_path, request, NIR_C_BACKEND, lib)
    _assert_payloads_close(c_payload, torch_payload)
    assert _payload_peak(c_payload) > 0.0


def test_encoded_samples_agree_across_backends() -> None:
    """A raw sample's final prediction agrees on both backends."""
    lib = _lib()
    bundle = _conv_bundle()
    generator = torch.Generator().manual_seed(7)
    sample = (torch.rand(1, 8, 8, generator=generator) < 0.5).float()
    with InferenceSession.load(bundle) as torch_session:
        expected = torch_session.step_sample(sample)
    with InferenceSession.load(
        bundle, backend=NIR_C_BACKEND, c_lib=lib
    ) as actual_session:
        actual = actual_session.step_sample(sample)
    assert actual.steps == expected.steps
    assert actual.label == expected.label
    assert _close(actual.logits, expected.logits)
    assert _close(actual.class_totals, expected.class_totals)

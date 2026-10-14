"""The adapter path drives a stateful serving session end to end.

A recorded stream, windowed and encoded through the adapter path,
drives an ``InferenceSession`` -- the same path ``/v1/stream`` uses.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

import spikeforge_io as sio
from spikeforge.network import model_store
from spikeforge.streaming.recipe import (
    StreamTrainConfig,
    StreamTrainingResult,
    build_model,
)
from spikeforge.streaming.serving import (
    build_bundle,
    load_session,
    save_checkpoint,
)
from spikeforge.streaming.window_spec import fit_window_spec


def _stream(rows: int = 12, channels: int = 2, seed: int = 0) -> torch.Tensor:
    """Return a deterministic ``[rows, channels]`` float stream."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, channels, generator=generator)

@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads and writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _session_bundle(tmp_path: Path) -> Any:
    """Build an (untrained) bundle whose window contract is frozen."""
    config = StreamTrainConfig(
        seq_length=4,
        features=2,
        hidden=8,
        num_classes=2,
        num_steps=3,
        epochs=1,
        seed=0,
    )
    spec, module = build_model(config)
    result = StreamTrainingResult(
        spec=spec,
        module=module,
        config=config,
        encode_spec=config.encode_spec(),
        history=(),
    )
    window_spec = fit_window_spec(_stream(rows=32), 4, 2)
    save_checkpoint("io_session_model", result, window_spec)
    out = str(tmp_path / "io.spkf")
    build_bundle("io_session_model", out=out)
    return load_session(out), window_spec, result


def test_adapter_stream_drives_an_inference_session(tmp_path: Path) -> None:
    """A recorded stream, encoded by the adapter path, drives a session."""
    session, window_spec, result = _session_bundle(tmp_path)
    adapter = sio.InMemoryAdapter(_stream(rows=32))
    windows = adapter.windows(window_spec)
    spikes = sio.encode_windows(windows[:1], result.encode_spec)
    session.reset()
    prediction = None
    for step in range(int(spikes.size(0))):
        prediction = session.step(spikes[step])
    assert prediction is not None
    assert session.steps == int(spikes.size(0))
    assert prediction.label in (0, 1)

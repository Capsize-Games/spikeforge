"""PT-W7 acceptance: I/O adapters feed the frozen encode/serving contract.

Every adapter round-trips a numeric stream into the expected ``[N, D]`` shape,
windowing produces the frozen ``[W, L, D]`` windows, and a recorded stream
drives ``encode_windows`` and a stateful
:class:`~spikeforge.serving.session.InferenceSession` — the same path
``/v1/stream`` uses. Replay emits the exact ``{"frame": ...}`` message shape.
"""

import json
from pathlib import Path
from typing import Any, List, Sequence

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
from spikeforge_io.errors import (
    AdapterFormatError,
    AdapterShapeError,
    AdapterUnsupportedError,
)


def _stream(rows: int = 12, channels: int = 2, seed: int = 0) -> torch.Tensor:
    """Return a deterministic ``[rows, channels]`` float stream."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, channels, generator=generator)


def test_as_stream_shapes() -> None:
    """1-D payloads gain a channel and 2-D payloads are kept."""
    assert tuple(sio.as_stream([1, 2, 3], "m").shape) == (3, 1)
    assert tuple(sio.as_stream([[1, 2], [3, 4]], "m").shape) == (2, 2)


def test_as_stream_refuses_non_numeric_and_wrong_rank() -> None:
    """A non-numeric payload and a 3-D payload are both refused."""
    with pytest.raises(AdapterFormatError):
        sio.as_stream([["a", "b"]], "m")
    with pytest.raises(AdapterShapeError):
        sio.as_stream(torch.zeros(2, 2, 2), "m")


def test_csv_round_trip(tmp_path: Path) -> None:
    """A CSV with a header parses to the original ``[N, D]`` rows."""
    stream = _stream()
    path = tmp_path / "recording.csv"
    path.write_text(
        "c0,c1\n"
        + "\n".join(f"{a},{b}" for a, b in stream.tolist())
        + "\n",
        encoding="utf-8",
    )
    adapter = sio.CsvAdapter(str(path))
    assert torch.allclose(adapter.stream(), stream)
    assert len(adapter) == 12


def test_csv_ragged_row_is_refused(tmp_path: Path) -> None:
    """A row with the wrong width is a typed format error."""
    path = tmp_path / "ragged.csv"
    path.write_text("c0,c1\n1,2\n3\n", encoding="utf-8")
    with pytest.raises(AdapterFormatError):
        sio.CsvAdapter(str(path)).stream()


def test_csv_non_numeric_field_is_refused(tmp_path: Path) -> None:
    """A non-numeric cell names its row in the typed error."""
    path = tmp_path / "bad.csv"
    path.write_text("c0,c1\n1,oops\n", encoding="utf-8")
    with pytest.raises(AdapterFormatError) as excinfo:
        sio.CsvAdapter(str(path)).stream()
    assert excinfo.value.position == 0


def test_json_variants_parse(tmp_path: Path) -> None:
    """A list of rows, a scalar list, and an object key all parse."""
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps([[1, 2], [3, 4]]), encoding="utf-8")
    assert tuple(sio.JsonAdapter(str(rows)).stream().shape) == (2, 2)

    scalars = tmp_path / "scalars.json"
    scalars.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert tuple(sio.JsonAdapter(str(scalars)).stream().shape) == (3, 1)

    obj = tmp_path / "obj.json"
    obj.write_text(json.dumps({"data": [1, 2]}), encoding="utf-8")
    assert tuple(sio.JsonAdapter(str(obj)).stream().shape) == (2, 1)


def test_json_missing_key_is_refused(tmp_path: Path) -> None:
    """A JSON object without a stream key is a typed format error."""
    path = tmp_path / "nokey.json"
    path.write_text(json.dumps({"other": 1}), encoding="utf-8")
    with pytest.raises(AdapterFormatError):
        sio.JsonAdapter(str(path)).stream()


def test_npy_round_trip(tmp_path: Path) -> None:
    """A ``.npy`` array parses to the same values through numpy."""
    np = pytest.importorskip("numpy")
    stream = _stream(rows=6, channels=3)
    path = tmp_path / "stream.npy"
    np.save(str(path), stream.numpy())
    adapter = sio.NpyAdapter(str(path))
    assert torch.allclose(adapter.stream(), stream)


def test_in_memory_adapter() -> None:
    """An in-memory tensor adapts without a file."""
    stream = _stream()
    adapter = sio.InMemoryAdapter(stream)
    assert torch.allclose(adapter.stream(), stream)


def test_dataset_adapter_hooks() -> None:
    """The dataset hook accepts ``to_tensor`` and the sequence protocol."""
    stream = _stream(rows=4)

    class TensorDataset:
        def to_tensor(self) -> torch.Tensor:
            return stream

    class RowDataset:
        def __len__(self) -> int:
            return stream.size(0)

        def __getitem__(self, index: int) -> Sequence[float]:
            return stream[index].tolist()

    assert torch.allclose(
        sio.DatasetAdapter(TensorDataset()).stream(), stream
    )
    assert torch.allclose(sio.DatasetAdapter(RowDataset()).stream(), stream)


def test_dataset_adapter_refuses_a_plain_object() -> None:
    """An object without the hook is a typed format error."""
    with pytest.raises(AdapterFormatError):
        sio.DatasetAdapter(object()).stream()


def test_adapter_for_dispatches_and_refuses_unknown() -> None:
    """The factory picks by suffix and refuses an unknown one."""
    assert isinstance(sio.adapter_for("a.csv"), sio.CsvAdapter)
    assert isinstance(sio.adapter_for("a.json"), sio.JsonAdapter)
    assert isinstance(sio.adapter_for("a.npy"), sio.NpyAdapter)
    with pytest.raises(AdapterUnsupportedError):
        sio.adapter_for("a.parquet")


def test_adapter_windows_match_the_core_contract(tmp_path: Path) -> None:
    """Adapter windows equal the frozen core ``WindowSpec.windows``."""
    stream = _stream(rows=20)
    spec = fit_window_spec(stream, length=4, stride=2)
    adapter = sio.InMemoryAdapter(stream)
    assert torch.allclose(adapter.windows(spec), spec.windows(stream))
    assert adapter.windows(spec).shape == (9, 4, 2)


def test_window_spec_from_bundle_and_windows() -> None:
    """The bundle's frozen window contract round-trips through the adapter."""
    stream = _stream(rows=20)
    spec = fit_window_spec(stream, length=4, stride=2)

    class FakeBundle:
        preprocessing = {"window": spec.to_dict()}

    assert sio.window_spec_from_bundle(FakeBundle()).digest() == spec.digest()
    assert torch.allclose(
        sio.windows(FakeBundle(), stream), spec.windows(stream)
    )


def test_replay_feeds_every_frame_to_a_sink() -> None:
    """Replay passes one frame per sample and reports the count."""
    adapter = sio.InMemoryAdapter(_stream(rows=7))
    seen: List[torch.Tensor] = []
    report = sio.replay(adapter, seen.append)
    assert report.frames == 7
    assert report.windows is False
    assert len(seen) == 7
    assert seen[0].shape == (2,)


def test_stream_messages_match_the_v1_stream_envelope() -> None:
    """The dry-run messages use the ``/v1/stream`` ``frame`` shape."""
    adapter = sio.InMemoryAdapter(_stream(rows=3))
    messages = list(sio.stream_messages(adapter, reset=True, session_id="s1"))
    assert messages[0] == {"reset": True, "session_id": "s1"}
    assert messages[1]["encoded"] is False
    assert messages[1]["session_id"] == "s1"
    assert len(messages[1]["frame"]) == 2
    assert len(messages) == 4


def test_replay_to_stream_collects_replies() -> None:
    """Replaying into a send callable collects one reply per message."""
    adapter = sio.InMemoryAdapter(_stream(rows=3))

    def send(message: dict) -> dict:
        return {"type": "prediction" if "frame" in message else "reset"}

    report = sio.replay_to_stream(adapter, send, reset=True)
    assert report.frames == 3
    assert list(report.replies)[0] == {"type": "reset"}
    assert list(report.replies)[-1] == {"type": "prediction"}


def test_encoded_returns_a_spike_tensor() -> None:
    """Adapter windows encode to ``[T, W, L, D]`` via the shared contract."""
    stream = _stream(rows=20)
    spec = fit_window_spec(stream, length=4, stride=2)
    adapter = sio.InMemoryAdapter(stream)
    spikes = sio.encode_windows(adapter.windows(spec), None)
    assert spikes.shape == (100, 9, 4, 2)


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

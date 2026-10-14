"""Numpy/torch stream-view agreement for the I/O adapters (issue #161).

Every adapter offers the same ``[N, D]`` float32 stream twice: torch via
:meth:`StreamAdapter.stream` and numpy via
:meth:`StreamAdapter.stream_numpy`. The two views must agree exactly on a
shared corpus, and refuse bad payloads with the same typed errors.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import pytest
import torch

import spikeforge_io as sio
from spikeforge_io.adapters import StreamAdapter
from spikeforge_io.errors import (
    AdapterDependencyError,
    AdapterFormatError,
    AdapterShapeError,
)

#: A stream coercion: ``(payload, source)`` to an ``[N, D]`` stream.
Coerce = Callable[[Any, str], Any]


def _rows() -> List[List[float]]:
    """Return a small deterministic ``[4, 2]`` corpus."""
    return [[1.0, 2.0], [3.0, 4.0], [-0.5, 12.25], [7.0, -8.0]]


def _csv(path: Path, rows: List[List[float]], header: bool) -> str:
    """Write ``rows`` as CSV and return the path."""
    lines = ["c0,c1"] if header else []
    lines += [",".join(repr(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _json_file(path: Path, payload: Any) -> str:
    """Write ``payload`` as JSON and return the path."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _npy(path: Path, array: np.ndarray) -> str:
    """Write ``array`` as ``.npy`` and return the path."""
    np.save(str(path), array)
    return str(path)


def _assert_views_match(adapter: StreamAdapter) -> None:
    """Assert an adapter's torch and numpy views agree exactly."""
    tensor = adapter.stream()
    array = adapter.stream_numpy()
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32
    assert tensor.dtype == torch.float32
    assert tuple(array.shape) == tuple(tensor.shape)
    assert bool(np.array_equal(array, tensor.numpy()))
    assert len(adapter) == int(tensor.size(0))


def test_as_stream_numpy_shapes() -> None:
    """Scalars, 1-D, and 2-D payloads shape like the torch twin."""
    assert sio.as_stream_numpy(5, "m").shape == (1, 1)
    assert sio.as_stream_numpy([1, 2, 3], "m").shape == (3, 1)
    assert sio.as_stream_numpy([[1, 2], [3, 4]], "m").shape == (2, 2)


@pytest.mark.parametrize("coerce", [sio.as_stream, sio.as_stream_numpy])
def test_coercions_refuse_bad_payloads(coerce: Coerce) -> None:
    """Both coercions refuse non-numeric, None, and 3-D payloads alike."""
    for bad in ([["a", "b"]], None, {"a": 1}):
        with pytest.raises(AdapterFormatError):
            coerce(bad, "m")
    with pytest.raises(AdapterShapeError):
        coerce(np.zeros((2, 2, 2)), "m")


def test_csv_views_match(tmp_path: Path) -> None:
    """CSV rows with and without a header agree across both views."""
    rows = _rows()
    headed = sio.CsvAdapter(_csv(tmp_path / "h.csv", rows, True))
    plain = sio.CsvAdapter(
        _csv(tmp_path / "p.csv", rows, False), has_header=False
    )
    _assert_views_match(headed)
    _assert_views_match(plain)
    assert headed.stream_numpy().tolist() == rows


def test_tsv_views_match(tmp_path: Path) -> None:
    """The ``.tsv`` factory route agrees across both views."""
    path = tmp_path / "r.tsv"
    path.write_text(
        "c0\tc1\n" + "\n".join(f"{a}\t{b}" for a, b in _rows()) + "\n",
        encoding="utf-8",
    )
    _assert_views_match(sio.adapter_for(str(path)))


def test_json_variants_views_match(tmp_path: Path) -> None:
    """JSON rows, scalars, objects, and explicit keys agree in both."""
    rows = _rows()
    cases = [
        sio.JsonAdapter(_json_file(tmp_path / "r.json", rows)),
        sio.JsonAdapter(_json_file(tmp_path / "s.json", [1, 2, 3])),
        sio.JsonAdapter(_json_file(tmp_path / "o.json", {"data": rows})),
        sio.JsonAdapter(
            _json_file(tmp_path / "k.json", {"trace": rows}), key="trace"
        ),
    ]
    for adapter in cases:
        _assert_views_match(adapter)


def test_in_memory_views_match() -> None:
    """Nested lists, tensors, and arrays agree across both views."""
    rows = _rows()
    payloads = [
        rows,
        [1, 2, 3],
        torch.tensor(rows),
        torch.tensor([1.5, 2.5]),
        np.asarray(rows, dtype=np.float64),
        np.float32(2.5),
    ]
    for payload in payloads:
        _assert_views_match(sio.InMemoryAdapter(payload))


def test_npy_views_match(tmp_path: Path) -> None:
    """A ``.npy`` array agrees across both views."""
    array = np.asarray(_rows(), dtype=np.float32)
    _assert_views_match(sio.NpyAdapter(_npy(tmp_path / "s.npy", array)))


def test_dataset_method_hooks_views_match() -> None:
    """``to_tensor``/``array`` datasets agree across both views."""
    rows = _rows()

    class TensorDataset:
        def to_tensor(self) -> torch.Tensor:
            return torch.tensor(rows)

    class ArrayDataset:
        def array(self) -> List[List[float]]:
            return rows

    _assert_views_match(sio.DatasetAdapter(TensorDataset()))
    _assert_views_match(sio.DatasetAdapter(ArrayDataset()))


def test_dataset_sequence_views_match() -> None:
    """Sequence-protocol datasets agree across both views."""
    rows = _rows()

    class RowDataset:
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> List[float]:
            return rows[index]

    _assert_views_match(sio.DatasetAdapter(RowDataset()))


def test_default_numpy_view_converts_the_torch_stream() -> None:
    """A subclass without an override converts its torch stream."""

    class OnlyTorch(sio.StreamAdapter):
        def stream(self) -> torch.Tensor:
            return torch.tensor([[1.0, 2.0]])

    _assert_views_match(OnlyTorch())


def test_seeded_corpus_views_match(tmp_path: Path) -> None:
    """A seeded random corpus agrees exactly through every file adapter."""
    random = np.random.RandomState(0)
    array = random.randn(50, 3).astype(np.float32)
    rows = array.tolist()
    adapters = [
        sio.CsvAdapter(_csv(tmp_path / "r.csv", rows, True)),
        sio.JsonAdapter(_json_file(tmp_path / "r.json", rows)),
        sio.NpyAdapter(_npy(tmp_path / "r.npy", array)),
        sio.InMemoryAdapter(rows),
    ]
    for adapter in adapters:
        _assert_views_match(adapter)


def test_channel_mismatch_hits_both_views(tmp_path: Path) -> None:
    """A wrong channel count is a shape error in both views."""
    adapter = sio.CsvAdapter(
        _csv(tmp_path / "c.csv", _rows(), True), channels=3
    )
    with pytest.raises(AdapterShapeError):
        adapter.stream()
    with pytest.raises(AdapterShapeError):
        adapter.stream_numpy()


def test_ragged_json_hits_both_views(tmp_path: Path) -> None:
    """Ragged JSON rows blame the shape in both views alike."""
    adapter = sio.JsonAdapter(_json_file(tmp_path / "r.json", [[1, 2], [3]]))
    with pytest.raises(AdapterFormatError, match="not rectangular"):
        adapter.stream()
    with pytest.raises(AdapterFormatError, match="not rectangular"):
        adapter.stream_numpy()


def test_torch_view_reports_torch_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without torch, numpy ingestion works and torch reports itself."""
    monkeypatch.setitem(sys.modules, "torch", None)
    adapter = sio.InMemoryAdapter(_rows())
    assert adapter.stream_numpy().shape == (4, 2)
    assert len(adapter) == 4
    with pytest.raises(AdapterDependencyError) as excinfo:
        adapter.stream()
    assert excinfo.value.dependency == "torch"

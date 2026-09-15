"""Reading Heidelberg spiking-audio splits without tonic's broken decode.

Tonic converts these files' timestamps with ``times * 1e6``, but they are
stored as ``float16`` and NumPy 2 keeps the multiply in ``float16`` -- where
``1e6`` does not even fit, so the scale factor becomes ``inf`` and every
product is ``inf`` or ``NaN``, casting to ``INT64_MIN``. Every timestamp in
every sample is lost, and the pipeline would have binned the whole recording
into one time step and trained happily on it.

These tests build a file in the real layout with real ``float16`` timestamps,
assert the arithmetic that broke it *does* break, and then assert our reader
recovers the timestamps exactly.
"""

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

from spikeforge.data import event_loader
from spikeforge.data.datasets import dataset_spec
from spikeforge.events import hsd_reader

h5py = pytest.importorskip("h5py")

#: Event times in seconds, spanning a plausible ~1.4 s utterance.
_SECONDS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.4)
_CHANNELS: Tuple[int, ...] = (0, 17, 300, 699, 42, 128)
_SENSOR = (700, 1, 1)


class _StubTonic:
    """The part of a tonic HSD dataset our reader actually consults."""

    sensor_size = _SENSOR

    def __init__(self, directory: Path, filename: str) -> None:
        """Record where the extracted HDF5 lives, as tonic would."""
        self.location_on_system = str(directory)
        self.data_filename = filename


def _write_split(path: Path, samples: int = 3) -> None:
    """Write an HDF5 file in the Heidelberg layout, times as ``float16``."""
    with h5py.File(path, "w") as handle:
        spikes = handle.create_group("spikes")
        ragged_float = h5py.special_dtype(vlen=np.dtype("float16"))
        ragged_int = h5py.special_dtype(vlen=np.dtype("int64"))
        times = spikes.create_dataset(
            "times", (samples,), dtype=ragged_float
        )
        units = spikes.create_dataset("units", (samples,), dtype=ragged_int)
        for index in range(samples):
            times[index] = np.array(_SECONDS, dtype=np.float16)
            units[index] = np.array(_CHANNELS, dtype=np.int64)
        handle.create_dataset(
            "labels", data=np.arange(samples, dtype=np.int64)
        )


@pytest.fixture
def split(tmp_path: Path) -> hsd_reader.HsdSplit:
    """Return a reader over a freshly written Heidelberg-shaped split."""
    _write_split(tmp_path / "ssc_train.h5")
    return hsd_reader.open_split(_StubTonic(tmp_path, "ssc_train.h5"))


def _expected_microseconds() -> List[int]:
    """Return the microsecond timestamps a correct conversion produces."""
    stored = np.array(_SECONDS, dtype=np.float16).astype(np.float64)
    return [int(value) for value in np.rint(stored * 1e6)]


# --- the bug this module exists to route around -------------------------


def test_tonics_own_arithmetic_really_does_destroy_the_timestamps() -> None:
    """The premise, asserted rather than assumed.

    If NumPy ever changes this promotion, this test fails and the workaround
    can be reconsidered instead of quietly outliving its reason.
    """
    stored = np.array(_SECONDS, dtype=np.float16)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = stored * 1e6
        cast = scaled.astype(np.int64)
    if scaled.dtype != np.float16:
        # NumPy 1.x value-based promotion widens this to float32 and the
        # timestamps survive. The corruption needs NEP 50's weak scalars, so
        # on a legacy NumPy there is no premise to assert -- and this project
        # still supports numpy>=1.26. The reader is asserted correct either
        # way by the tests below; only the upstream tripwire is version-bound.
        pytest.skip(
            f"needs NEP 50 weak-scalar promotion; NumPy "
            f"{np.__version__} widened the multiply to {scaled.dtype}"
        )
    # 1e6 itself does not fit in float16 (max 65504), so it becomes inf. The
    # multiply is then inf, or NaN wherever the timestamp is exactly 0.
    assert scaled.dtype == np.float16
    assert not np.isfinite(scaled).any()
    assert bool(np.isnan(scaled[0]))
    assert bool(np.isinf(scaled[1:]).all())
    assert set(cast.tolist()) == {np.iinfo(np.int64).min}


# --- what our reader produces instead -----------------------------------


def test_the_reader_recovers_every_timestamp(
    split: hsd_reader.HsdSplit,
) -> None:
    """Widening to float64 before scaling makes the arithmetic exact."""
    events, label = split[0]
    assert events["t"].tolist() == _expected_microseconds()
    assert sorted(set(events["t"].tolist())) == events["t"].tolist()
    assert label == 0


def test_the_reader_keeps_tonics_stream_layout(
    split: hsd_reader.HsdSplit,
) -> None:
    """Downstream code must not be able to tell the two paths apart."""
    events, _label = split[0]
    assert events.dtype == hsd_reader.DTYPE
    assert events.dtype.names == ("t", "x", "p")
    assert events["x"].tolist() == list(_CHANNELS)
    assert set(events["p"].tolist()) == {1}


def test_the_reader_reports_the_splits_own_length_and_sensor(
    split: hsd_reader.HsdSplit,
) -> None:
    """Length and geometry come from the file and from tonic, not from us."""
    assert len(split) == 3
    assert split.sensor_size == _SENSOR


def test_labels_track_the_index(split: hsd_reader.HsdSplit) -> None:
    """Each sample carries its own label, not the first one's."""
    assert [split[i][1] for i in range(3)] == [0, 1, 2]


def test_the_file_is_opened_once_not_per_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tonic reopens the HDF5 per ``__getitem__``; this must not.

    That reopen is why the source's open-once caching bought nothing for these
    datasets.
    """
    _write_split(tmp_path / "ssc_train.h5")
    opens = []
    real_file = h5py.File

    def counting_file(*args: Any, **kwargs: Any) -> Any:
        opens.append(args[0])
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", counting_file)
    reader = hsd_reader.open_split(_StubTonic(tmp_path, "ssc_train.h5"))
    for index in range(3):
        reader[index]
    assert len(opens) == 1


# --- and end to end through the conversion ------------------------------


def test_a_read_sample_survives_the_timestamp_guard(
    split: hsd_reader.HsdSplit,
) -> None:
    """The guard that refuses destroyed timing must pass real timing.

    Binned across 25 steps the utterance occupies more than one bin, which is
    the whole difference between this and what tonic delivered.
    """
    events, _label = split[0]
    sample = event_loader.events_to_sample(events, (1, 700), 25)
    assert sample.num_events == len(_SECONDS)
    assert sample.shape == (1, 700)
    assert len(set(sample.t.tolist())) > 1
    assert sample.y.tolist() == [0] * len(_SECONDS)


def test_binning_spreads_the_utterance_across_the_window(
    split: hsd_reader.HsdSplit,
) -> None:
    """A 1.4 s utterance in 25 bins lands at the proportional positions."""
    events, _label = split[0]
    sample = event_loader.events_to_sample(events, (1, 700), 25)
    assert sample.t.tolist() == [0, 4, 8, 13, 17, 24]


# --- the registry routes the dataset here -------------------------------


def test_the_registry_routes_ssc_to_the_native_reader() -> None:
    """Routing is declared on the dataset, not hard-coded in the loader."""
    assert dataset_spec("ssc").native_reader == hsd_reader.HSD


def test_dvs_datasets_keep_tonics_own_decode() -> None:
    """Only the datasets tonic decodes wrongly are routed around it."""
    for name in ("n_mnist", "dvs128_gesture", "cifar10_dvs"):
        assert dataset_spec(name).native_reader is None


def test_the_reader_reports_its_availability() -> None:
    """``h5py`` ships with tonic, but the probe is defensive either way."""
    assert hsd_reader.available() is True

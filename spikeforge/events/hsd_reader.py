"""Read Heidelberg spiking-audio splits (SHD, SSC) from their HDF5 directly.

Tonic downloads these datasets correctly and then decodes them wrongly. Its
reader converts the file's timestamps from seconds to microseconds with
``times * 1e6``, but the Heidelberg files store them as ``float16``, whose
maximum is 65504. Under NumPy 2's NEP 50 promotion a ``float16`` array times a
Python float stays ``float16``, and ``1e6`` does not fit: the scale factor
itself becomes ``inf``. Every product is then ``inf`` -- or ``NaN`` wherever
the timestamp is exactly 0 -- and casting either to ``int64`` yields
``INT64_MIN``, for every timestamp in every sample. Binning then sees a
zero-width span and collapses the whole recording into one time step, so a
25-step spiking network trains on a single static frame and still reports a
plausible accuracy.

So this module replaces that one conversion. It does **not** replace tonic:
the download, extraction, cache layout, and sensor geometry all still come
from the tonic dataset object, which is passed in. Only ``__getitem__`` is
ours, and it scales in ``float64`` where the arithmetic is exact.

Two incidental wins. Tonic reopens the HDF5 file on every ``__getitem__``;
this opens it once, which is what makes
:class:`~spikeforge.events.event_source.EventSampleSource`'s open-once caching
actually pay off here. And the emitted stream keeps tonic's own
``(t, x, p)`` structured layout, so nothing downstream can tell the
difference -- a cochlea has channels rather than pixel rows, and
:func:`~spikeforge.data.event_loader.events_to_sample` already places a
one-row sensor's events on row 0.

``h5py`` arrives with ``tonic`` itself, so the ``events`` extra covers both,
but the import is defensive here exactly as it is in
:mod:`spikeforge.events.tonic_api`: this is the only module in the project
that imports ``h5py``.
"""

import os
from importlib import import_module
from typing import Any, Optional, Tuple

import numpy as np

#: Name a registry entry sets in ``DatasetSpec.native_reader`` to route here.
HSD = "hsd"
#: The structured layout tonic's own HSD reader emits, kept verbatim so
#: nothing downstream has to special-case this path.
DTYPE = np.dtype([("t", int), ("x", int), ("p", int)])
#: Seconds-to-microseconds scale, applied in float64 where it is exact.
MICROSECONDS = 1e6


def _h5py() -> Optional[Any]:
    """Return the ``h5py`` module, or ``None`` when it is unavailable."""
    try:
        return import_module("h5py")
    except ImportError:
        return None


def available() -> bool:
    """Return True when the HDF5 reader can run in this environment."""
    return _h5py() is not None


class HsdSplit:
    """One SHD/SSC split, read from its HDF5 file.

    Presents the same surface the rest of the event path uses of a tonic
    dataset -- ``len()``, ``[index]``, and ``sensor_size`` -- so it drops in
    wherever the tonic object went.
    """

    def __init__(self, dataset: Any) -> None:
        """Open the split ``dataset`` points at, reading its layout from it.

        ``dataset`` is the constructed tonic dataset: it has already done the
        download and extraction, and it knows the cache layout and the sensor
        geometry. Nothing here second-guesses any of that.
        """
        module = _h5py()
        if module is None:
            raise RuntimeError(
                "h5py is required to read SHD/SSC and is normally installed "
                'with tonic; reinstall the `events` extra: pip install -e '
                '".[events]"'
            )
        self._path = os.path.join(
            dataset.location_on_system, dataset.data_filename
        )
        self.sensor_size: Tuple[int, int, int] = dataset.sensor_size
        self._file = module.File(self._path, "r")
        self._times = self._file["spikes/times"]
        self._units = self._file["spikes/units"]
        self._labels = self._file["labels"]

    def __len__(self) -> int:
        """Return the split's sample count."""
        return int(len(self._labels))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Return one ``((t, x, p) array, label)`` pair.

        The timestamps are widened to ``float64`` *before* being scaled, which
        is the whole point of this module: doing it in the file's own
        ``float16`` overflows and loses every timestamp.
        """
        position = int(index)
        seconds = np.asarray(self._times[position], dtype=np.float64)
        units = np.asarray(self._units[position], dtype=np.int64)
        events = np.empty(seconds.shape[0], dtype=DTYPE)
        events["t"] = np.rint(seconds * MICROSECONDS).astype(np.int64)
        events["x"] = units
        # The Heidelberg recordings carry no polarity; tonic supplies a
        # constant 1 and we match it rather than inventing a second channel.
        events["p"] = 1
        return events, int(np.asarray(self._labels[position]))

    def close(self) -> None:
        """Close the underlying HDF5 file."""
        self._file.close()

    @property
    def path(self) -> str:
        """Return the HDF5 file this split reads."""
        return self._path


def open_split(dataset: Any) -> HsdSplit:
    """Return an :class:`HsdSplit` reading the split ``dataset`` located."""
    return HsdSplit(dataset)

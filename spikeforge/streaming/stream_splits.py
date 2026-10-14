"""The train/validation/test split plus the frozen windowing contract."""

from dataclasses import dataclass

from spikeforge.streaming.stream_dataset import StreamDataset
from spikeforge.streaming.window_spec import WindowSpec


@dataclass(frozen=True)
class StreamSplits:
    """The train/validation/test split plus the frozen windowing contract."""

    train: StreamDataset
    val: StreamDataset
    test: StreamDataset
    window_spec: WindowSpec

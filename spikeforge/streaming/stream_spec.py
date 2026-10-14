"""The geometry and signal shape of one synthetic stream."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StreamSpec:
    """The geometry and signal shape of one synthetic stream."""

    #: Samples per window (the ``L`` axis) and the window stride.
    length: int = 16
    stride: int = 8
    #: Number of numeric channels (the ``D`` axis).
    channels: int = 4
    #: Number of class labels the waveform generator separates.
    classes: int = 3
    #: Timesteps per class-labelled segment and how many segments to stitch.
    segment_length: int = 32
    segments: int = 12
    #: Probability that a segment carries an injected anomaly.
    anomaly_rate: float = 0.25
    #: Standard deviation of the additive Gaussian sensor noise.
    noise: float = 0.15
    #: Base seed; validation/test derive from it deterministically.
    seed: int = 0

    def validate(self) -> None:
        """Raise :class:`ValueError` when the spec is unusable."""
        if self.length < 1 or self.stride < 1:
            raise ValueError("length and stride must be >= 1")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.classes < 2:
            raise ValueError("classes must be >= 2")
        if self.segment_length < 1 or self.segments < 1:
            raise ValueError("segment_length and segments must be >= 1")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError("anomaly_rate must be in [0, 1]")
        if self.noise < 0.0:
            raise ValueError("noise must be >= 0")

    @property
    def channel_names(self) -> Tuple[str, ...]:
        """Return the stable channel order ``c0, c1, ...``."""
        return tuple(f"c{index}" for index in range(self.channels))

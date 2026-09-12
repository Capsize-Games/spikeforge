"""P0: the frozen windowing + z-score contract for the streaming use case.

A numeric stream becomes fixed-shape windows determined by a *frozen*
:class:`WindowSpec`: the window ``length``, the ``stride``, the channel order,
and the per-channel mean/std used to z-score. The statistics are fitted on the
**train** stream only and then pinned; validation and test windows are
normalised with the train statistics, and the same spec is stored in the
serving bundle's ``preprocessing.json`` so a served window can never be
normalised differently from a training one.

The module is pure torch and stdlib.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

#: Version of the windowing contract; a mismatch is refused by a reader.
WINDOW_SPEC_VERSION: int = 1

#: Floor applied to a fitted standard deviation so a constant channel cannot
#: divide by zero.
_STD_FLOOR: float = 1e-6


def _as_stream(stream: Any) -> torch.Tensor:
    """Return ``stream`` as a ``[N, D]`` float32 tensor."""
    tensor = torch.as_tensor(stream, dtype=torch.float32)
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError("a stream must be a 1-D or 2-D array")
    return tensor


def window_stream(
    stream: Any, length: int, stride: int = 1
) -> torch.Tensor:
    """Return every ``[length, D]`` window of ``stream`` as ``[W, L, D]``.

    A stride of 1 is the streaming default (one window per new sample). Fewer
    samples than ``length`` yields an empty ``[0, L, D]`` tensor rather than an
    error, so a short stream degrades cleanly.
    """
    tensor = _as_stream(stream)
    steps = int(length)
    step = int(stride)
    if steps < 1 or step < 1:
        raise ValueError("length and stride must be >= 1")
    samples = tensor.size(0)
    if samples < steps:
        return tensor.new_zeros((0, steps, tensor.size(1)))
    count = (samples - steps) // step + 1
    index = torch.arange(count, device=tensor.device) * step
    offsets = torch.arange(steps, device=tensor.device)
    gather = index.unsqueeze(1) + offsets.unsqueeze(0)
    return tensor[gather]


def fit_window_spec(
    stream: Any,
    length: int,
    stride: int = 1,
    channel_names: Optional[Sequence[str]] = None,
) -> "WindowSpec":
    """Fit a :class:`WindowSpec` on the **train** ``stream`` only.

    The per-channel mean and (biased) standard deviation are computed over the
    raw train samples and floored at :data:`_STD_FLOOR`.
    """
    tensor = _as_stream(stream)
    channels = int(tensor.size(1))
    names = tuple(channel_names or (f"c{i}" for i in range(channels)))
    if len(names) != channels:
        raise ValueError("channel_names must name every channel")
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False).clamp_min(_STD_FLOOR)
    return WindowSpec(
        length=int(length),
        stride=int(stride),
        channels=channels,
        channel_names=names,
        mean=tuple(float(value) for value in mean.tolist()),
        std=tuple(float(value) for value in std.tolist()),
    )


@dataclass(frozen=True)
class WindowSpec:
    """The frozen window geometry and the train-fitted z-score statistics."""

    length: int
    stride: int
    channels: int
    channel_names: Tuple[str, ...]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    spec_version: int = WINDOW_SPEC_VERSION

    def validate(self) -> None:
        """Raise :class:`ValueError` when the spec is unusable."""
        if self.length < 1 or self.stride < 1:
            raise ValueError("length and stride must be >= 1")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if len(self.channel_names) != self.channels:
            raise ValueError("channel_names must name every channel")
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError("mean and std must cover every channel")
        if any(value <= 0 for value in self.std):
            raise ValueError("every channel std must be positive")
        if self.spec_version != WINDOW_SPEC_VERSION:
            raise ValueError(
                f"unsupported window spec version {self.spec_version!r}; "
                f"this runtime speaks {WINDOW_SPEC_VERSION}"
            )

    def normalize(self, windows: Any) -> torch.Tensor:
        """Return ``windows`` z-scored with the frozen train statistics.

        ``windows`` is ``[..., D]`` (a ``[L, D]`` window or an ``[W, L, D]``
        batch); the ``mean``/``std`` broadcast over the leading axes.
        """
        tensor = torch.as_tensor(windows, dtype=torch.float32)
        if tensor.size(-1) != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {tensor.size(-1)}"
            )
        mean = torch.tensor(self.mean, dtype=tensor.dtype)
        std = torch.tensor(self.std, dtype=tensor.dtype)
        return (tensor - mean) / std

    def windows(self, stream: Any) -> torch.Tensor:
        """Return the z-scored ``[W, L, D]`` windows of ``stream``."""
        return self.normalize(window_stream(stream, self.length, self.stride))

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical, JSON-ready windowing contract."""
        return {
            "length": int(self.length),
            "stride": int(self.stride),
            "channels": int(self.channels),
            "channel_names": list(self.channel_names),
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "spec_version": int(self.spec_version),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WindowSpec":
        """Return a spec from a mapping, coercing every field."""
        if not isinstance(data, Mapping):
            raise ValueError("window spec must be a mapping")
        names = tuple(str(name) for name in data.get("channel_names", ()))
        mean = tuple(float(value) for value in data.get("mean", ()))
        std = tuple(float(value) for value in data.get("std", ()))
        return cls(
            length=int(data["length"]),
            stride=int(data.get("stride", 1)),
            channels=int(data["channels"]),
            channel_names=names,
            mean=mean,
            std=std,
            spec_version=int(data.get("spec_version", WINDOW_SPEC_VERSION)),
        )

    def digest(self) -> str:
        """Return the hex SHA-256 of the canonical (sorted-key) spec JSON."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_windows(
    windows: Any, spec: "WindowSpec"
) -> torch.Tensor:
    """Return ``windows`` z-scored with ``spec``'s frozen statistics."""
    spec.validate()
    return spec.normalize(windows)

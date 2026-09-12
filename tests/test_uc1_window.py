"""UC-1 P0 acceptance: the frozen windowing + z-score contract.

Every check is about the contract itself: windows are fixed-shape and
reproducible, and normalisation is pinned to the train-fitted statistics so a
served window can never be normalised differently from a training one.
"""

import pytest
import torch

from spikeforge.streaming import window_spec as ws


def _stream(rows: int = 20, channels: int = 2, seed: int = 0) -> torch.Tensor:
    """Return a deterministic ``[rows, channels]`` float stream."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, channels, generator=generator) + 1.0


def test_window_stream_shapes_and_stride() -> None:
    """A strided window stream returns ``[W, L, D]`` in sample order."""
    stream = torch.arange(40, dtype=torch.float32).reshape(20, 2)
    windows = ws.window_stream(stream, 5, 5)
    assert tuple(windows.shape) == (4, 5, 2)
    assert torch.equal(windows[0], stream[0:5])
    assert torch.equal(windows[1], stream[5:10])


def test_window_stream_default_stride_is_one() -> None:
    """The streaming default yields one window per new sample."""
    stream = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    assert ws.window_stream(stream, 4).shape == (7, 4, 2)


def test_window_stream_short_stream_is_empty() -> None:
    """Fewer samples than the window length degrades to an empty batch."""
    windows = ws.window_stream(torch.rand(3, 2), 5, 1)
    assert tuple(windows.shape) == (0, 5, 2)


def test_window_stream_rejects_bad_geometry() -> None:
    """A non-positive length or stride is refused."""
    with pytest.raises(ValueError):
        ws.window_stream(torch.rand(4, 2), 0, 1)
    with pytest.raises(ValueError):
        ws.window_stream(torch.rand(4, 2), 2, 0)


def test_fit_is_deterministic_and_hashable() -> None:
    """Fitting twice on the same train stream gives the same frozen spec."""
    stream = _stream()
    first = ws.fit_window_spec(stream, length=6, stride=3)
    second = ws.fit_window_spec(stream, length=6, stride=3)
    assert first.to_dict() == second.to_dict()
    assert first.digest() == second.digest()
    assert first.channel_names == ("c0", "c1")


def test_normalize_uses_frozen_train_statistics() -> None:
    """Normalisation divides by the train stats, not the window's own."""
    train = _stream(seed=0)
    spec = ws.fit_window_spec(train, length=6, stride=3)
    normalized = spec.normalize(train)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(
        normalized.std(dim=0, unbiased=False), torch.ones(2), atol=1e-4
    )
    shifted = train + 5.0
    assert not torch.allclose(
        spec.normalize(shifted).mean(dim=0), torch.zeros(2), atol=0.1
    )
    assert tuple(spec.windows(train).shape)[-1] == 2


def test_spec_round_trips_through_dict() -> None:
    """A spec survives ``to_dict``/``from_dict`` unchanged."""
    spec = ws.fit_window_spec(_stream(), length=4, stride=2)
    restored = ws.WindowSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    assert restored.digest() == spec.digest()
    assert restored.spec_version == ws.WINDOW_SPEC_VERSION


def test_validate_rejects_unusable_spec() -> None:
    """A non-positive std or an unknown version is refused."""
    spec = ws.fit_window_spec(_stream(), length=4, stride=2)
    broken = ws.WindowSpec(
        length=spec.length,
        stride=spec.stride,
        channels=spec.channels,
        channel_names=spec.channel_names,
        mean=spec.mean,
        std=(0.0, 1.0),
    )
    with pytest.raises(ValueError):
        broken.validate()
    future = ws.WindowSpec(
        length=4,
        stride=2,
        channels=2,
        channel_names=("a", "b"),
        mean=(0.0, 0.0),
        std=(1.0, 1.0),
        spec_version=ws.WINDOW_SPEC_VERSION + 1,
    )
    with pytest.raises(ValueError):
        future.validate()


def test_normalize_windows_helpers_agree() -> None:
    """The module helper and the spec method return identical tensors."""
    spec = ws.fit_window_spec(_stream(), length=5, stride=1)
    windows = ws.window_stream(_stream(seed=2), 5, 1)
    assert torch.equal(
        ws.normalize_windows(windows, spec), spec.normalize(windows)
    )

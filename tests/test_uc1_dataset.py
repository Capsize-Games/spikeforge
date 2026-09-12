"""UC-1 P1 acceptance: the deterministic synthetic streaming dataset."""

import torch

from spikeforge.streaming import stream_source


def _spec(**overrides: object) -> stream_source.StreamSpec:
    """Return a small spec, overriding any field the test needs."""
    base = {
        "length": 8,
        "stride": 4,
        "channels": 2,
        "classes": 2,
        "segment_length": 16,
        "segments": 8,
        "anomaly_rate": 0.4,
        "seed": 3,
    }
    base.update(overrides)
    return stream_source.StreamSpec(**base)


def test_generate_stream_shapes_and_labels() -> None:
    """The raw stream, per-sample labels, and anomaly mask are consistent."""
    spec = _spec()
    stream, labels, anomaly = stream_source.generate_stream(spec)
    assert tuple(stream.shape) == (spec.segments * spec.segment_length, 2)
    assert labels.shape == (stream.size(0),)
    assert anomaly.shape == (stream.size(0),)
    assert labels.dtype == torch.long
    assert anomaly.dtype == torch.bool
    assert int(labels.min()) >= 0
    assert int(labels.max()) < spec.classes
    assert set(labels.tolist()) == {0, 1}


def test_datasets_are_deterministic() -> None:
    """Two builds of the same spec give byte-identical splits."""
    first = stream_source.build_datasets(_spec())
    second = stream_source.build_datasets(_spec())
    for left, right in (
        (first.train, second.train),
        (first.val, second.val),
        (first.test, second.test),
    ):
        assert torch.equal(left.windows, right.windows)
        assert torch.equal(left.labels, right.labels)
        assert torch.equal(left.anomaly, right.anomaly)
    assert first.window_spec.digest() == second.window_spec.digest()


def test_splits_have_expected_shapes() -> None:
    """Every split windows to ``[W, L, D]`` with valid labels."""
    spec = _spec()
    splits = stream_source.build_datasets(spec)
    for split in (splits.train, splits.val, splits.test):
        assert len(split) > 0
        assert tuple(split.windows.shape)[1:] == (spec.length, spec.channels)
        assert split.labels.shape == (len(split),)
        assert split.anomaly.shape == (len(split),)


def test_splits_are_independent_streams() -> None:
    """Train and test come from different seeds, so their streams differ."""
    spec = _spec()
    train_stream, _, _ = stream_source.generate_stream(spec, spec.seed)
    test_stream, _, _ = stream_source.generate_stream(
        spec, spec.seed + stream_source._TEST_SEED_OFFSET
    )
    assert not torch.allclose(train_stream, test_stream)


def test_anomaly_rate_is_honoured() -> None:
    """A rate of 1.0 flags every window; 0.0 flags none."""
    always = stream_source.build_datasets(_spec(anomaly_rate=1.0))
    assert bool(always.train.anomaly.all())
    never = stream_source.build_datasets(_spec(anomaly_rate=0.0))
    assert not bool(never.train.anomaly.any())


def test_window_statistics_come_from_train_only() -> None:
    """The frozen spec's stats match the train stream, not test."""
    spec = _spec()
    splits = stream_source.build_datasets(spec)
    train_stream, _, _ = stream_source.generate_stream(spec, spec.seed)
    mean = train_stream.mean(dim=0)
    assert torch.allclose(
        torch.tensor(splits.window_spec.mean), mean, atol=1e-5
    )


def test_different_seeds_give_different_windows() -> None:
    """A different base seed changes the generated windows."""
    left = stream_source.build_datasets(_spec(seed=3))
    right = stream_source.build_datasets(_spec(seed=9))
    assert not torch.allclose(left.train.windows, right.train.windows)

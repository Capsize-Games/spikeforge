"""The download opt-out fails on a cold cache instead of fetching (#118).

``download=False`` must raise the dataset's own not-found error without
touching the network; a transient mirror failure with downloading enabled
is retried with backoff instead of failing the caller (issue #87).
"""

from typing import Any, List

import pytest
import torchvision.datasets.utils as tv_utils

from spikeforge.data import datasets
from spikeforge.data.data_loader import build_loader
from spikeforge.data.sample_source import SampleSource
from spikeforge.training.training_engine import TrainingEngine


@pytest.fixture
def _cold_cache(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the dataset directory at an empty directory for one test."""
    monkeypatch.setattr(datasets, "DATA_DIR", str(tmp_path))


@pytest.fixture
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if torchvision tries to download anything."""

    def _refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must not touch the network")

    monkeypatch.setattr(tv_utils, "download_url", _refuse)
    monkeypatch.setattr(tv_utils, "download_and_extract_archive", _refuse)


def test_build_dataset_without_download_raises_on_a_cold_cache(
    _cold_cache: None, _no_network: None,
) -> None:
    """No-download mode reports the missing files instead of fetching."""
    with pytest.raises(RuntimeError, match="download"):
        datasets.build_dataset("mnist", download=False)


def test_sample_source_honours_its_download_flag(
    _cold_cache: None, _no_network: None,
) -> None:
    """The source passes ``download`` through instead of ignoring it."""
    with pytest.raises(RuntimeError, match="download"):
        SampleSource("mnist", download=False)


def test_build_loader_without_download_raises_on_a_cold_cache(
    _cold_cache: None, _no_network: None,
) -> None:
    """The shared loader offers the same no-download behavior."""
    with pytest.raises(RuntimeError, match="download"):
        build_loader("mnist", download=False)


def test_engine_without_download_fails_before_its_first_batch(
    _cold_cache: None, _no_network: None,
) -> None:
    """An offline engine raises from training, not from a download."""
    engine = TrainingEngine(
        dataset="mnist", num_steps=2, device="cpu", subset=2, epochs=1,
        download=False,
    )
    with pytest.raises(RuntimeError, match="download"):
        list(engine.train())


class _Flaky:
    """A torchvision-shaped class that fails ``failures`` times first."""

    calls: List[str] = []
    failures: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Record the attempt, then fail until the budget is spent."""
        type(self).calls.append("attempt")
        if len(type(self).calls) <= type(self).failures:
            raise RuntimeError("Error downloading train-images-idx3-ubyte")


@pytest.fixture
def _flaky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the flaky class and skip real backoff sleeps for one test."""
    _Flaky.calls = []
    _Flaky.failures = 0
    monkeypatch.setattr(datasets.time, "sleep", lambda seconds: None)


def test_transient_mirror_failures_are_retried(_flaky: None) -> None:
    """Two flakes then a success builds the dataset on attempt three."""
    _Flaky.failures = 2
    built = datasets._instantiate(
        _Flaky, {}, train=True, download=True, size=(28, 28)
    )
    assert isinstance(built, _Flaky)
    assert len(_Flaky.calls) == 3


def test_persistent_mirror_failures_raise_after_three_attempts(
    _flaky: None,
) -> None:
    """A mirror that never recovers still fails, after three tries."""
    _Flaky.failures = 10
    with pytest.raises(RuntimeError, match="downloading"):
        datasets._instantiate(
            _Flaky, {}, train=True, download=True, size=(28, 28)
        )
    assert len(_Flaky.calls) == 3


def test_the_opt_out_does_not_retry(_flaky: None) -> None:
    """``download=False`` raises at once; there is nothing to retry."""
    _Flaky.failures = 10
    with pytest.raises(RuntimeError, match="downloading"):
        datasets._instantiate(
            _Flaky, {}, train=True, download=False, size=(28, 28)
        )
    assert len(_Flaky.calls) == 1


def test_non_download_errors_are_not_retried(_flaky: None) -> None:
    """A programming error (not a mirror flake) fails immediately."""

    class _Broken:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _Flaky.calls.append("attempt")
            raise ValueError("bad constructor argument")

    with pytest.raises(ValueError, match="bad constructor"):
        datasets._instantiate(
            _Broken, {}, train=True, download=True, size=(28, 28)
        )
    assert len(_Flaky.calls) == 1

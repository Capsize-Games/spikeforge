"""Download worker routing between the image and event loaders."""

import pytest

from spikeforge.data import download_cli


def test_event_dataset_routes_to_event_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event modality is prepared by the tonic loader, not torchvision.

    Both declared splits are warmed here in the cancellable child, so the
    held-out score taken during training never reaches the network.
    """
    calls = []
    monkeypatch.setattr(
        download_cli,
        "ensure_event_dataset",
        lambda name, split: calls.append((name, split)),
    )
    monkeypatch.setattr(
        download_cli, "build_dataset", lambda *a, **k: calls.append("image")
    )
    download_cli._ensure("n_mnist", True)
    assert calls == [("n_mnist", "train"), ("n_mnist", "test")]


def test_event_dataset_warms_only_the_splits_it_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dataset shipping one pool is warmed once, not twice."""
    calls = []
    monkeypatch.setattr(
        download_cli,
        "ensure_event_dataset",
        lambda name, split: calls.append((name, split)),
    )
    download_cli._ensure("cifar10_dvs", True)
    assert calls == [("cifar10_dvs", "train")]


def test_image_dataset_keeps_the_torchvision_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image modality keeps using the torchvision builder unchanged."""
    calls = []
    monkeypatch.setattr(
        download_cli,
        "build_dataset",
        lambda name, train, download: calls.append((name, train, download)),
    )
    download_cli._ensure("mnist", False)
    assert calls == [("mnist", False, True)]

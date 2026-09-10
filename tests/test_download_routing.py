"""Download worker routing between the image and event loaders."""

import pytest

from snn_interpreter.data import download_cli


def test_event_dataset_routes_to_event_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event modality is prepared by the tonic loader, not torchvision."""
    calls = []
    monkeypatch.setattr(
        download_cli, "ensure_event_dataset", calls.append
    )
    monkeypatch.setattr(
        download_cli, "build_dataset", lambda *a, **k: calls.append("image")
    )
    download_cli._ensure("n_mnist", True)
    assert calls == ["n_mnist"]


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

"""UC-1 P2/P3 acceptance: CPU training smoke and evaluation metrics."""

import torch

from spikeforge.streaming import delta_over_window, recipe, stream_source


def _splits() -> stream_source.StreamSplits:
    """Return a small deterministic split bundle for the tests."""
    spec = stream_source.StreamSpec(
        length=8,
        stride=4,
        channels=2,
        classes=2,
        segment_length=16,
        segments=10,
        anomaly_rate=0.4,
        seed=1,
    )
    return stream_source.build_datasets(spec)


def _config(epochs: int = 30) -> recipe.StreamTrainConfig:
    """Return a tiny CPU recipe config that trains in well under a second."""
    return recipe.StreamTrainConfig(
        seq_length=8,
        features=2,
        hidden=16,
        num_classes=2,
        num_steps=4,
        epochs=epochs,
        lr=5e-2,
        seed=0,
    )


def test_encode_windows_shape_and_determinism() -> None:
    """Delta coding is deterministic and keeps the ``[T, B, L, D]`` layout."""
    splits = _splits()
    config = _config()
    first = recipe.encode_windows(splits.train.windows, config.encode_spec())
    second = recipe.encode_windows(splits.train.windows, config.encode_spec())
    assert tuple(first.shape)[1:] == tuple(splits.train.windows.shape)
    assert first.shape[0] == config.num_steps
    assert torch.equal(first, second)


def test_delta_over_window_removes_the_first_sample() -> None:
    """The temporal-change transform zeroes each window's first sample."""
    windows = torch.rand(3, 5, 2)
    deltas = delta_over_window(windows)
    assert deltas.shape == windows.shape
    assert torch.allclose(deltas[:, 0, :], torch.zeros(3, 2))
    assert torch.allclose(
        deltas[:, 1:, :], windows[:, 1:, :] - windows[:, :-1, :]
    )


def test_train_smoke_history_and_weight_change() -> None:
    """Training records one finite loss per epoch and moves the weights."""
    splits = _splits()
    config = _config()
    fresh_spec, fresh = recipe.build_model(config)
    fresh_weights = {
        key: value.clone() for key, value in fresh.state_dict().items()
    }
    result = recipe.train_classifier(splits.train, config)
    assert len(result.history) == config.epochs
    assert all(
        float(entry["loss"]) == float(entry["loss"])
        for entry in result.history
    )
    assert result.spec.to_dict() == fresh_spec.to_dict()
    changed = any(
        not torch.allclose(fresh_weights[key], value)
        for key, value in result.module.state_dict().items()
    )
    assert changed


def test_train_classifier_is_deterministic() -> None:
    """Two runs of the same config produce the same loss history."""
    splits = _splits()
    config = _config(epochs=5)
    first = recipe.train_classifier(splits.train, config)
    second = recipe.train_classifier(splits.train, config)
    assert first.history == second.history


def test_evaluate_metrics_are_sane() -> None:
    """F1, balanced accuracy, and anomaly AUROC land in valid ranges."""
    splits = _splits()
    config = _config()
    result = recipe.train_classifier(splits.train, config)
    threshold = recipe.fit_threshold(result.module, splits.train, config)
    report = recipe.evaluate(
        result.module, splits.test, config, threshold=threshold
    )
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert 0.0 <= report["balanced_accuracy"] <= 1.0
    assert len(report["per_class_recall"]) == config.num_classes
    assert 0.0 <= report["anomaly_auroc"] <= 1.0
    assert report["accuracy"] > 1.0 / config.num_classes
    assert report["macro_f1"] > 0.0
    assert report["anomaly_auroc"] >= 0.5


def test_anomaly_threshold_rule_flags_the_tail() -> None:
    """The threshold rule flags exactly the scores above the threshold."""
    scores = torch.arange(100, dtype=torch.float32)
    threshold = recipe.anomaly_threshold(scores, percentile=90.0)
    flags = recipe.anomaly_rule(scores, threshold)
    assert int(flags.sum()) == 10
    assert bool(flags[99]) and not bool(flags[0])


def test_class_metrics_perfect_and_auroc_endpoints() -> None:
    """Perfect predictions score 1.0 and AUROC handles both extremes."""
    truth = torch.tensor([0, 1, 0, 1])
    perfect = recipe.class_metrics(truth, truth.clone(), 2)
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == 1.0
    assert perfect["per_class_recall"] == [1.0, 1.0]
    labels = torch.tensor([0, 0, 1, 1])
    assert recipe.auroc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert recipe.auroc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert recipe.auroc(torch.tensor([0.5, 0.6]), torch.tensor([1, 1])) == 0.5


def test_anomaly_scores_are_bounded() -> None:
    """The one-class score is ``1 - max softmax`` and lies in ``[0, 1]``."""
    logits = torch.randn(6, 3) * 3.0
    scores = recipe.anomaly_scores(logits)
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= 1.0)
    confident = recipe.anomaly_scores(torch.tensor([[10.0, 0.0, 0.0]]))
    assert float(confident.item()) < 0.01

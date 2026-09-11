"""Train an image model headlessly with the Python training API.

This runs the fully-connected LIF network on MNIST for a couple of batches
and prints the final metrics dict. It is the headless equivalent of the
dashboard's training panel.

Run from the repository root::

    venv/bin/python examples/01_train_image_model.py

The first run downloads MNIST into ``SPIKEFORGE_DATA_DIR`` (default
``build/``); later runs are offline. Note that ``train_accuracy`` is
``-1.0`` until the
metric is first computed, and ``test_accuracy`` is a percentage (``null``
until an evaluation step runs).
"""

from typing import Any, Dict

from spikeforge.training.training_engine import TrainingEngine


def train_once() -> Dict[str, Any]:
    """Train a tiny MNIST run and return the last metrics dict."""
    engine = TrainingEngine(
        dataset="mnist",
        hidden=32,
        epochs=1,
        num_steps=5,
        subset=64,
        batch_size=32,
        device="cpu",
        seed=0,
    )
    last: Dict[str, Any] = {}
    for metrics in engine.train():
        last = metrics
    return last


def main() -> int:
    """Run the headless training example and print its final metrics."""
    metrics = train_once()
    print("metric keys:", sorted(metrics))
    print(
        f"loss={metrics['loss']:.4f} "
        f"train_accuracy={metrics['train_accuracy']:.3f} "
        f"test_accuracy={metrics['test_accuracy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

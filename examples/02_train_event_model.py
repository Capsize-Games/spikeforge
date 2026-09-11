"""Train an event model on the labelled synthetic event path.

The event modality is shown end to end without a network: the
``EventSampleSource`` serves explicitly-labelled synthetic N-MNIST streams,
the ``EventSpikeBridge`` lays them out for the ``fc_legacy`` topology, and
``EventTrainingEngine`` runs the shared training loop on them.

Run from the repository root::

    venv/bin/python examples/02_train_event_model.py

On a synthetic stream the accuracy is near chance (about 10% for ten
classes); it is a wiring demo, not a benchmark. Without ``synthetic_only``
a missing ``tonic`` raises the typed ``EventsExtraMissingError`` instead of
passing a synthetic stream off as a real recording.
"""

from typing import Any, Dict

from spikeforge.events.event_bridge import EventSpikeBridge
from spikeforge.events.event_source import EventSampleSource
from spikeforge.topology.registry import build_topology
from spikeforge.training.event_engine import EventTrainingEngine


def inspect_source() -> None:
    """Print the synthetic source's provenance and a bridged layout."""
    source = EventSampleSource("n_mnist", synthetic_only=True)
    sample, label = source.load(0)
    spec, _module = build_topology("fc_legacy", {"num_classes": 10})
    spikes, meta = EventSpikeBridge().encode(sample, spec)
    print("origin:", source.origin, "| label:", label)
    print("description:", source.description)
    print("sensor:", tuple(sample.shape), "bridged:", tuple(spikes.shape))
    print("bridge sparsity:", round(float(meta["sparsity"]), 4))


def train_once() -> Dict[str, Any]:
    """Train one epoch on synthetic events and return the last metrics."""
    engine = EventTrainingEngine(
        dataset="n_mnist",
        synthetic_only=True,
        topology="fc_legacy",
        hidden=32,
        epochs=1,
        num_steps=6,
        subset=16,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    last: Dict[str, Any] = {}
    for metrics in engine.train():
        last = metrics
    return last


def main() -> int:
    """Run the synthetic event training example and report its metrics."""
    inspect_source()
    metrics = train_once()
    print(
        f"loss={metrics['loss']:.4f} "
        f"test_accuracy={metrics['test_accuracy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

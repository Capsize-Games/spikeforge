"""Guarded smoke test for the real tonic event pipeline (needs tonic)."""

import pytest

from snn_interpreter.data import event_loader
from snn_interpreter.events.event_bridge import EventSpikeBridge
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module


def test_tonic_sample_converts_and_drives_simulator() -> None:
    """A tiny tonic-dtype sample converts to simulator-ready spikes.

    Skips when the optional ``events`` extra (``tonic``) is absent. The
    sample is built from tonic's own event dtype, so the test exercises the
    real ``(x, y, t, p)`` layout without any network download.
    """
    pytest.importorskip("tonic")
    from tonic.io import make_structured_array

    events = make_structured_array(
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 10, 20, 30],
        [True, False, True, False],
    )
    sample = event_loader.events_to_sample(events, (8, 8), num_steps=4)
    assert sample.num_events == 4
    assert set(sample.p.tolist()) == {1, -1}
    spec = presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=64)
    spikes, meta = EventSpikeBridge().encode(sample, spec)
    assert list(spikes.shape) == [4, 1, 64]
    assert meta["modality"] == "event"
    result = run(build_module(spec), spikes)
    assert list(result.logits.shape) == [1, 3]

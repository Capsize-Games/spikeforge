"""The stateful :class:`InferenceSession` lifecycle and its typed errors."""

from typing import Any, Dict, Tuple

import pytest
import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving import bundle_manifest
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import BundleFormatError, StateError
from spikeforge.serving.session import InferenceSession
from spikeforge.simulator import input_shape
from spikeforge.topology import registry


def _bundle(spec: Any, module: Any) -> DeploymentBundle:
    """Build an in-memory bundle around a live module."""
    return DeploymentBundle(
        manifest={
            "format": bundle_manifest.BUNDLE_FORMAT,
            "version": bundle_manifest.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=module.state_dict(),
    )


def _spikes(spec: Any, steps: int = 5, batch: int = 2) -> torch.Tensor:
    """Return a seeded spike train shaped for ``spec``'s input stage."""
    torch.manual_seed(0)
    flat = torch.rand(steps, batch, 28 * 28)
    return input_shape.to_input_shape(flat, spec)


def _session(
    name: str, params: Dict[str, Any]
) -> Tuple[Any, InferenceSession]:
    """Return ``(spec, session)`` for one registry topology."""
    spec, module = registry.build_topology(name, params)
    return spec, InferenceSession.load(_bundle(spec, module))


def test_load_builds_module_from_the_bundle_spec() -> None:
    """A loaded session exposes the bundle's spec and starts unstepped."""
    spec, session = _session("fc_small", {"hidden": 5, "num_classes": 3})
    assert session.spec.to_dict() == spec.to_dict()
    assert session.steps == 0
    assert session.mode is ExecutionMode.PRODUCTION


def test_step_returns_cumulative_class_totals() -> None:
    """``class_totals`` sums each step's readout across the stream."""
    spec, session = _session("fc_small", {"hidden": 5, "num_classes": 3})
    frame = _spikes(spec)[0]
    first = session.step(frame)
    second = session.step(frame)
    assert second.steps == 2
    assert torch.allclose(
        second.class_totals, first.logits + second.logits
    )
    assert 0 <= second.label < 3


def test_state_round_trip_restores_the_stream() -> None:
    """A saved state resumes the stream exactly where it stopped."""
    spec, module = registry.build_topology(
        "fc_small", {"hidden": 5, "num_classes": 3}
    )
    frames = _spikes(spec)
    session = InferenceSession.load(_bundle(spec, module))
    for frame in frames[:3]:
        session.step(frame)
    saved = session.state()
    expected = session.step(frames[3])
    resumed = InferenceSession.load(_bundle(spec, module))
    resumed.load_state(saved)
    actual = resumed.step(frames[3])
    assert actual.steps == expected.steps
    assert torch.allclose(actual.class_totals, expected.class_totals)


def test_build_module_rejects_mismatched_state_dict() -> None:
    """Weights that do not fit the manifest's spec are a format error."""
    spec, _ = registry.build_topology(
        "conv_net", {"channels": 2, "num_classes": 4}
    )
    _, other = registry.build_topology(
        "fc_small", {"hidden": 5, "num_classes": 3}
    )
    bundle = DeploymentBundle(
        manifest={
            "format": bundle_manifest.BUNDLE_FORMAT,
            "version": bundle_manifest.BUNDLE_VERSION,
            "spec": spec.to_dict(),
        },
        weights=other.state_dict(),
    )
    with pytest.raises(BundleFormatError):
        bundle.build_module()


def test_load_state_rejects_a_foreign_spec() -> None:
    """A state from a differently shaped topology is refused."""
    spec_a, session_a = _session("fc_small", {"hidden": 5, "num_classes": 3})
    session_a.step(_spikes(spec_a)[0])
    spec_b, session_b = _session("conv_net", {"channels": 2, "num_classes": 4})
    session_b.step(_spikes(spec_b)[0])
    with pytest.raises(StateError):
        session_b.load_state(session_a.state())


def test_rejects_a_non_object_state() -> None:
    """A state payload that is not a mapping is a typed error."""
    _, session = _session("fc_small", {"hidden": 5, "num_classes": 3})
    with pytest.raises(StateError):
        session.load_state(["not", "a", "mapping"])


def test_streaming_does_not_accumulate_an_autograd_graph() -> None:
    """Serving is inference-only, so no prediction carries a grad_fn."""
    spec, session = _session("fc_small", {"hidden": 5, "num_classes": 3})
    prediction = None
    for frame in _spikes(spec, steps=6):
        prediction = session.step(frame)
    assert prediction is not None
    assert prediction.logits.requires_grad is False
    assert prediction.class_totals.grad_fn is None

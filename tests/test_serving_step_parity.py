"""Streaming inference must agree with the closed-loop simulator."""

from typing import Any, Dict, List, Tuple

import pytest
import torch

from spikeforge.serving import bundle_manifest
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.session import InferenceSession
from spikeforge.simulator import input_shape
from spikeforge.simulator.frames import normalise_frame
from spikeforge.simulator.runner import run
from spikeforge.simulator.step_stages import step_stages
from spikeforge.topology import registry

_CASES: List[Tuple[str, Dict[str, Any]]] = [
    ("fc_legacy", {"hidden": 6, "beta": 0.5, "num_classes": 4}),
    ("fc_small", {"hidden": 5, "beta": 0.9, "num_classes": 3}),
    ("conv_net", {"channels": 2, "num_classes": 4}),
    ("recurrent_net", {"hidden": 6, "beta": 0.9, "num_classes": 4}),
]


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


@pytest.mark.parametrize("name,params", _CASES)
def test_step_stages_matches_stage_module_step(
    name: str, params: Dict[str, Any]
) -> None:
    """The shared helper is exactly the normalised ``module.step``."""
    spec, module = registry.build_topology(name, params)
    frame = _spikes(spec)[0]
    shared, _ = step_stages(module, frame)
    kind = spec.stage(spec.input).kind
    direct, _ = module.step(normalise_frame(frame, kind))
    assert set(shared) == set(direct)
    for key, value in shared.items():
        assert torch.allclose(value, direct[key])


@pytest.mark.parametrize("name,params", _CASES)
def test_streaming_matches_closed_loop_run(
    name: str, params: Dict[str, Any]
) -> None:
    """A streamed sequence reproduces the whole-tensor ``run`` readout."""
    spec, module = registry.build_topology(name, params)
    module.eval()
    spikes = _spikes(spec)
    session = InferenceSession.load(_bundle(spec, module))
    last = None
    for frame in spikes:
        last = session.step(frame)
    trajectory = run(module, spikes)
    assert last is not None
    assert last.steps == spikes.size(0)
    assert torch.allclose(
        last.class_totals / last.steps, trajectory.logits, atol=1e-6
    )


@pytest.mark.parametrize("name,params", _CASES)
def test_run_stream_preserves_state_across_frames(
    name: str, params: Dict[str, Any]
) -> None:
    """``run_stream`` yields one prediction per frame, carrying state."""
    spec, module = registry.build_topology(name, params)
    spikes = _spikes(spec)
    streamed = list(
        InferenceSession.load(_bundle(spec, module)).run_stream(spikes)
    )
    looped = []
    session = InferenceSession.load(_bundle(spec, module))
    for frame in spikes:
        looped.append(session.step(frame))
    assert len(streamed) == len(looped) == spikes.size(0)
    assert torch.allclose(
        streamed[-1].class_totals, looped[-1].class_totals
    )


@pytest.mark.parametrize("name,params", _CASES)
def test_reset_between_streams_reproduces_a_fresh_run(
    name: str, params: Dict[str, Any]
) -> None:
    """After reset the session behaves as if it had never seen a frame."""
    spec, module = registry.build_topology(name, params)
    frame = _spikes(spec)[0]
    session = InferenceSession.load(_bundle(spec, module))
    first = session.step(frame)
    session.reset()
    again = session.step(frame)
    assert session.steps == 1
    assert torch.allclose(first.class_totals, again.class_totals)

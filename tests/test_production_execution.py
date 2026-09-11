"""Production vs educational logits parity for every shipped preset."""

from typing import Any, Dict, List, Tuple

import pytest
import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator import input_shape
from spikeforge.simulator.production import run_production
from spikeforge.simulator.runner import run
from spikeforge.topology import registry
from spikeforge.topology.spec import TopologySpec

_CASES: List[Tuple[str, Dict[str, Any]]] = [
    ("fc_legacy", {"hidden": 6, "beta": 0.5, "num_classes": 4}),
    ("conv_net", {"channels": 2, "num_classes": 4}),
    ("recurrent_net", {"hidden": 6, "beta": 0.9, "num_classes": 4}),
]


def _spikes(spec: TopologySpec) -> torch.Tensor:
    """Build a seeded spike train shaped for ``spec``'s input stage."""
    torch.manual_seed(0)
    flat = torch.rand(4, 2, 28 * 28)
    return input_shape.to_input_shape(flat, spec)


@pytest.mark.parametrize("name,params", _CASES)
def test_production_matches_educational(
    name: str, params: Dict[str, Any]
) -> None:
    """Production logits match educational and production records nothing."""
    spec, module = registry.build_topology(name, params)
    spikes = _spikes(spec)
    quiet = run(module, spikes, mode=ExecutionMode.PRODUCTION)
    loud = run(module, spikes, mode=ExecutionMode.EDUCATIONAL)
    assert torch.allclose(quiet.logits, loud.logits, atol=1e-6)
    assert not quiet.recorded
    assert loud.recorded
    assert loud.spikes and loud.membranes and loud.currents


def test_run_production_reports_eager_default() -> None:
    """The production helper reports the eager path and records nothing."""
    spec, module = registry.build_topology(
        "fc_legacy", {"hidden": 6, "num_classes": 4}
    )
    spikes = _spikes(spec)
    result = run_production(module, spikes)
    educational = run(module, spikes, mode=ExecutionMode.EDUCATIONAL)
    assert result.compiled is False
    assert result.status == "eager"
    assert torch.allclose(result.logits, educational.logits, atol=1e-6)
    assert not result.trajectory.recorded


def test_run_defaults_stay_production_and_silent() -> None:
    """The bare ``run`` call keeps its production/no-recording defaults."""
    spec, module = registry.build_topology(
        "fc_small", {"hidden": 5, "beta": 0.9, "num_classes": 3}
    )
    trajectory = run(module, _spikes(spec))
    assert not trajectory.recorded
    assert trajectory.spikes == {}
    assert trajectory.membranes == {}
    assert trajectory.currents == {}

"""Server mode gating plus the extended ``model_list`` payload."""

import asyncio
from typing import Any, Dict, List, Optional

import torch

from server.handlers import dispatch
from server.introspection_handlers import EDUCATIONAL_ONLY
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from spikeforge.neurons.registry import neuron_kinds
from spikeforge.topology import registry
from spikeforge.topology.registry import topology_names

_STEPS = 4
_SMALL = {"hidden": 4, "beta": 0.5, "num_classes": 3}


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


class FakeSample:
    """A minimal stand-in for the configured encoder engine."""

    def __init__(self, steps: int = _STEPS) -> None:
        """Hold a fixed ``[T,1,784]`` spike volume."""
        self._spikes = torch.rand(steps, 1, 784)

    def spike_input(self) -> torch.Tensor:
        """Return the ``[T,1,784]`` spike volume."""
        return self._spikes


class FakeModel:
    """A minimal stand-in for the active training engine."""

    def __init__(self, spec: Any, net: Any, mode: str) -> None:
        """Store the topology spec, module, and declared execution mode."""
        self.spec = spec
        self.net = net
        self.mode = mode


def _model(mode: str = "educational") -> FakeModel:
    """Build a small deterministic model holder with a mode."""
    torch.manual_seed(0)
    spec, net = registry.build_topology("fc_small", dict(_SMALL))
    return FakeModel(spec, net, mode)


async def _dispatch(
    message: ClientMessage,
    sample: Optional[FakeSample] = None,
    model: Optional[FakeModel] = None,
) -> List[Dict[str, Any]]:
    """Dispatch one message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    if sample is not None:
        session.set_engine(sample)
    if model is not None:
        session.training._engine = model
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


def test_model_list_includes_topologies_and_neurons() -> None:
    """model_list advertises the registry topologies and neuron kinds."""
    sent = asyncio.run(_dispatch(ClientMessage(type="list_models")))
    assert [message["type"] for message in sent] == ["model_list"]
    payload = sent[0]["payload"]
    assert payload["topologies"] == topology_names()
    assert payload["neurons"] == list(neuron_kinds())
    assert "models" in payload and "datasets" in payload


def test_train_config_accepts_mode() -> None:
    """TrainConfig defaults to production and accepts educational."""
    assert TrainConfig().mode == "production"
    assert TrainConfig(mode="educational").mode == "educational"


def test_trajectory_succeeds_in_educational_mode() -> None:
    """An educational engine lets the trajectory action record traces."""
    sent = asyncio.run(
        _dispatch(ClientMessage(type="trajectory"), FakeSample(), _model())
    )
    assert [message["type"] for message in sent] == ["trajectory"]


def test_metrics_succeeds_in_educational_mode() -> None:
    """An educational engine lets the metrics action aggregate traces."""
    sent = asyncio.run(
        _dispatch(ClientMessage(type="metrics"), FakeSample(), _model())
    )
    assert [message["type"] for message in sent] == ["metrics"]


def test_trajectory_errors_in_production_mode() -> None:
    """A production engine rejects trajectory capture with a clear error."""
    sent = asyncio.run(
        _dispatch(
            ClientMessage(type="trajectory"), FakeSample(),
            _model("production"),
        )
    )
    assert [message["type"] for message in sent] == ["error"]
    assert sent[0]["payload"] == EDUCATIONAL_ONLY


def test_metrics_errors_in_production_mode() -> None:
    """A production engine rejects metrics capture with the same error."""
    sent = asyncio.run(
        _dispatch(
            ClientMessage(type="metrics"), FakeSample(), _model("production")
        )
    )
    assert [message["type"] for message in sent] == ["error"]
    assert sent[0]["payload"] == EDUCATIONAL_ONLY

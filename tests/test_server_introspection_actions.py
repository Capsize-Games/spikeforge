"""Dispatch routing and payload shapes for the introspection WS actions."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import torch

from server.handlers import dispatch
from server.introspection_payloads import MAX_NEURONS, MAX_STAGES
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from snn_interpreter.encoding.spike_encoder import SpikeEncoder
from snn_interpreter.introspection.surrogate import list_surrogates
from snn_interpreter.topology import registry

_SMALL = {"hidden": 4, "beta": 0.5, "num_classes": 3}
_WIDE = {"hidden": 100, "beta": 0.5, "num_classes": 3}
_STEPS = 4


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
        """Hold a fixed spike volume, image, and encoder."""
        self._spikes = torch.rand(steps, 1, 784)
        self._image = torch.rand(1, 28, 28)
        self._encoder = SpikeEncoder(coding="rate", num_steps=steps)

    def spike_input(self) -> torch.Tensor:
        """Return the ``[T,1,784]`` spike volume."""
        return self._spikes

    def sample_tensor(self) -> torch.Tensor:
        """Return the raw ``[1,28,28]`` image."""
        return self._image

    @property
    def encoder(self) -> SpikeEncoder:
        """Return the encoder that produced the spikes."""
        return self._encoder


class FakeModel:
    """A minimal stand-in for the active training engine."""

    def __init__(self, spec: Any, net: Any) -> None:
        """Store the topology spec and module."""
        self.spec = spec
        self.net = net


def _model(params: Optional[Dict[str, Any]] = None) -> FakeModel:
    """Build a small, deterministic model holder."""
    torch.manual_seed(0)
    spec, net = registry.build_topology("fc_small", dict(params or _SMALL))
    return FakeModel(spec, net)


async def _dispatch(
    message: ClientMessage,
    sample: Optional[FakeSample] = None,
    model: Optional[FakeModel] = None,
) -> List[Dict[str, Any]]:
    """Dispatch one client message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    if sample is not None:
        session.set_engine(sample)
    if model is not None:
        session.training._engine = model
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


def _message(kind: str, name: Optional[str] = None) -> ClientMessage:
    """Return an introspection message for the small FC topology."""
    return ClientMessage(type=kind, name=name, train=TrainConfig())


def test_dispatch_routes_trajectory_with_caps() -> None:
    """The trajectory action replies with bounded per-stage traces."""
    sent = asyncio.run(
        _dispatch(_message("trajectory"), FakeSample(), _model())
    )
    assert [m["type"] for m in sent] == ["trajectory"]
    payload = sent[0]["payload"]
    assert payload["caps"] == {
        "max_neurons": MAX_NEURONS, "max_stages": MAX_STAGES,
    }
    assert payload["steps"] == _STEPS
    assert len(payload["stages"]) <= MAX_STAGES
    assert json.dumps(payload)


def test_trajectory_caps_wide_stage_features() -> None:
    """A hidden stage wider than MAX_NEURONS is truncated in the payload."""
    sent = asyncio.run(
        _dispatch(_message("trajectory"), FakeSample(), _model(_WIDE))
    )
    payload = sent[0]["payload"]
    traces = payload["traces"]
    assert set(traces) == set(payload["stages"])
    for stage in traces.values():
        assert stage["neurons"] <= MAX_NEURONS
        for row in stage["membrane"]:
            assert len(row) == stage["neurons"]
    assert traces["lif1"]["neurons"] == MAX_NEURONS


def test_trajectory_includes_all_three_quantities() -> None:
    """Every traced stage carries membrane, current, and spike rows."""
    sent = asyncio.run(
        _dispatch(_message("trajectory"), FakeSample(), _model())
    )
    stage = sent[0]["payload"]["traces"]["lif1"]
    for key in ("membrane", "current", "spikes"):
        assert len(stage[key]) == _STEPS
        assert len(stage[key][0]) == stage["neurons"]


def test_dispatch_routes_metrics() -> None:
    """The metrics action replies with per-stage aggregate metrics."""
    sent = asyncio.run(_dispatch(_message("metrics"), FakeSample(), _model()))
    assert [m["type"] for m in sent] == ["metrics"]
    payload = sent[0]["payload"]
    assert payload["steps"] == _STEPS
    stage = payload["stages"]["lif1"]
    assert {"firing_rate", "sparsity", "isi", "histogram"} <= set(stage)
    assert json.dumps(payload)


def test_dispatch_routes_encoding_report() -> None:
    """The encoding_report action replies with an encoded sample report."""
    sent = asyncio.run(_dispatch(_message("encoding_report"), FakeSample()))
    assert [m["type"] for m in sent] == ["encoding_report"]
    payload = sent[0]["payload"]
    assert payload["coding"] == "rate"
    assert payload["reconstruction_supported"] is True
    assert json.dumps(payload)


def test_dispatch_routes_surrogate_list() -> None:
    """The surrogates action replies with the selectable names."""
    sent = asyncio.run(_dispatch(_message("surrogates")))
    assert [m["type"] for m in sent] == ["surrogate_list"]
    assert sent[0]["payload"] == list_surrogates()
    assert json.dumps(sent[0]["payload"])


def test_dispatch_routes_surrogate_curve() -> None:
    """The surrogate_curve action replies with parallel x/y lists."""
    name = next(item for item in list_surrogates() if item != "LSO")
    sent = asyncio.run(_dispatch(_message("surrogate_curve", name)))
    assert [m["type"] for m in sent] == ["surrogate_curve"]
    payload = sent[0]["payload"]
    assert payload["name"] == name
    assert len(payload["x"]) == len(payload["y"]) == 101
    assert json.dumps(payload)


def test_dispatch_routes_benchmark() -> None:
    """The benchmark action replies with a tiny JSON report."""
    message = ClientMessage(
        type="benchmark", train=TrainConfig(topology="fc_small")
    )
    sent = asyncio.run(_dispatch(message))
    assert [m["type"] for m in sent] == ["benchmark"]
    payload = sent[0]["payload"]
    assert payload["results"]
    assert payload["results"][0]["topology"] == "fc_small"
    assert json.dumps(payload)


def test_trajectory_requires_sample() -> None:
    """Without a configured sample the trajectory action errors."""
    sent = asyncio.run(_dispatch(_message("trajectory")))
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no sample configured"


def test_trajectory_requires_model() -> None:
    """With a sample but no model the trajectory action errors."""
    sent = asyncio.run(_dispatch(_message("trajectory"), FakeSample()))
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no model loaded; train or load one"


def test_metrics_requires_model() -> None:
    """With a sample but no model the metrics action errors."""
    sent = asyncio.run(_dispatch(_message("metrics"), FakeSample()))
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no model loaded; train or load one"


def test_encoding_report_requires_sample() -> None:
    """Without a configured sample the encoding_report action errors."""
    sent = asyncio.run(_dispatch(_message("encoding_report")))
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no sample configured"


def test_surrogate_curve_requires_name() -> None:
    """A missing surrogate name yields an error, not an exception."""
    sent = asyncio.run(_dispatch(_message("surrogate_curve")))
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no surrogate selected"


def test_surrogate_curve_unknown_name_errors() -> None:
    """An unknown surrogate name is reported as an error payload."""
    sent = asyncio.run(_dispatch(_message("surrogate_curve", "nope")))
    assert [m["type"] for m in sent] == ["error"]
    assert "unknown surrogate" in sent[0]["payload"]

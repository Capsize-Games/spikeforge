"""Server payloads and dispatch routing for the NIR WS actions."""

import asyncio
import json
from typing import Any, Dict, List, Tuple

import pytest
import torch

from server.handlers import dispatch
from server.payloads import nir_graph_payload, nir_validation_payload
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import registry

pytest.importorskip("nir")

_Case = Tuple[str, Dict[str, Any]]
_FC: _Case = ("fc_legacy", {"hidden": 4, "beta": 0.5, "num_classes": 3})
_CONV: _Case = ("conv_net", {"channels": 2, "num_classes": 3})
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

    def __init__(self, steps: int = 4) -> None:
        """Hold a fixed [T,1,784] spike volume."""
        self._spikes = torch.rand(steps, 1, 784)

    def spike_input(self) -> torch.Tensor:
        """Return the [T,1,784] spike volume."""
        return self._spikes


def _build(case: _Case) -> Tuple[Any, Any]:
    """Build a small deterministic module for a case."""
    torch.manual_seed(0)
    return registry.build_topology(case[0], case[1])


def _kinds(payload: Dict[str, Any]) -> set:
    """Return the set of NIR node kinds named in a graph payload."""
    return {node["kind"] for node in payload["nodes"]}


def test_graph_payload_lists_fc_node_kinds() -> None:
    """fc_legacy exports affine and LIF primitives in its summary."""
    spec, module = _build(_FC)
    payload = nir_graph_payload(spec, module)
    assert {"Affine", "LI", "Threshold", "Delay", "Scale"} <= _kinds(payload)
    assert json.dumps(payload)


def test_graph_payload_lists_conv_node_kinds() -> None:
    """conv_net exports conv/pool/flatten primitives in its summary."""
    spec, module = _build(_CONV)
    payload = nir_graph_payload(spec, module)
    expected = {"Conv2d", "AvgPool2d", "SumPool2d", "Flatten"}
    assert expected <= _kinds(payload)
    assert json.dumps(payload)


def test_validation_payload_passes_for_valid_model() -> None:
    """A freshly built topology validates within tolerance."""
    spec, module = _build(_FC)
    report = nir_validation_payload(spec, module, torch.rand(4, 1, 784))
    assert report["within_tolerance"] is True
    assert report["worst"] is not None
    assert json.dumps(report)


def test_validation_payload_reports_worst_when_perturbed() -> None:
    """A perturbed exported weight flips the verdict and names the worst."""
    spec, module = _build(_FC)
    graph = to_nir(spec, module)
    graph.nodes["_fc1"].weight = graph.nodes["_fc1"].weight + 0.5
    report = nir_validation_payload(
        spec, module, torch.rand(4, 1, 784), graph=graph
    )
    assert report["within_tolerance"] is False
    assert report["worst"]["layer"]


async def _dispatch(
    message: ClientMessage, sample: Any = None
) -> List[Dict[str, Any]]:
    """Dispatch one client message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    if sample is not None:
        session.set_engine(sample)
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


def _export_message() -> ClientMessage:
    """Return a nir_export message for the small FC topology."""
    return ClientMessage(
        type="nir_export", train=TrainConfig(topology_params=dict(_SMALL))
    )


def _validate_message() -> ClientMessage:
    """Return a nir_validate message for the small FC topology."""
    return ClientMessage(
        type="nir_validate", train=TrainConfig(topology_params=dict(_SMALL))
    )


def test_dispatch_routes_nir_export() -> None:
    """The nir_export action replies with a nir_graph message."""
    sent = asyncio.run(_dispatch(_export_message()))
    assert [message["type"] for message in sent] == ["nir_graph"]
    assert sent[0]["payload"]["nodes"]


def test_dispatch_nir_validate_requires_sample() -> None:
    """Without a configured sample nir_validate reports a locked error."""
    sent = asyncio.run(_dispatch(_validate_message()))
    assert [message["type"] for message in sent] == ["error"]
    assert sent[0]["payload"] == "no sample configured"


def test_dispatch_routes_nir_validate() -> None:
    """With a sample configured nir_validate replies with a report."""
    sent = asyncio.run(_dispatch(_validate_message(), sample=FakeSample()))
    assert [message["type"] for message in sent] == ["nir_validation"]
    assert sent[0]["payload"]["within_tolerance"] is True

"""Dispatch routing and payload shapes for the target WS actions."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
import torch

from server.handlers import dispatch
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from snn_interpreter.targets.registry import target_names
from snn_interpreter.topology import registry

pytest.importorskip("nir")

_CONV = {"channels": 2, "num_classes": 3}


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


class FakeModel:
    """A minimal stand-in for the active training engine."""

    def __init__(self, spec: Any, net: Any) -> None:
        """Store the topology spec and module."""
        self.spec = spec
        self.net = net


def _conv_model() -> FakeModel:
    """Build a small deterministic conv preset holder."""
    torch.manual_seed(0)
    spec, net = registry.build_topology("conv_net", dict(_CONV))
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


def _targets_message() -> ClientMessage:
    """Return a parameterless ``targets`` message."""
    return ClientMessage(type="targets")


def _report_message(
    target: Optional[str] = None,
    topology: str = "conv_net",
    params: Optional[Dict[str, Any]] = None,
) -> ClientMessage:
    """Return a ``deployment_report`` message for the given target."""
    return ClientMessage(
        type="deployment_report",
        name=target,
        train=TrainConfig(
            topology=topology, topology_params=dict(params or _CONV)
        ),
    )


def test_dispatch_routes_targets() -> None:
    """The targets action replies with the annotated registry."""
    sent = asyncio.run(_dispatch(_targets_message()))
    assert [message["type"] for message in sent] == ["target_list"]
    payload = sent[0]["payload"]
    names = [item["name"] for item in payload["targets"]]
    assert names == target_names()
    assert all("available" in item for item in payload["targets"])
    assert json.dumps(payload)


def test_dispatch_routes_deployment_report_reference() -> None:
    """The reference report for conv_net is deployable with validation."""
    sent = asyncio.run(
        _dispatch(
            _report_message("reference"), FakeSample(), _conv_model()
        )
    )
    assert [message["type"] for message in sent] == ["deployment_report"]
    payload = sent[0]["payload"]
    assert payload["deployable"] is True
    assert payload["nodes"]["unsupported"] == []
    assert payload["validation"] is not None
    assert payload["validation"]["within_tolerance"] is True
    assert json.dumps(payload)


def test_deployment_report_narrow_target_flags_unsupported() -> None:
    """A target without conv support reports the unsupported nodes."""
    sent = asyncio.run(
        _dispatch(_report_message("xylo"), FakeSample(), _conv_model())
    )
    payload = sent[0]["payload"]
    assert payload["nodes"]["unsupported"]
    assert payload["deployable"] is False
    assert json.dumps(payload)


def test_deployment_report_without_model_uses_configured_topology() -> None:
    """With no model the configured topology is classified, sans validation."""
    sent = asyncio.run(_dispatch(_report_message("reference")))
    payload = sent[0]["payload"]
    assert payload["nodes"]["counts"]["total"] > 0
    assert payload["validation"] is None
    assert json.dumps(payload)


def test_deployment_report_defaults_to_reference() -> None:
    """A message without a target name reports against the reference."""
    sent = asyncio.run(_dispatch(_report_message()))
    assert sent[0]["payload"]["target"]["name"] == "reference"


def test_deployment_report_unknown_target_errors() -> None:
    """An unknown target name becomes an error payload, not a raise."""
    sent = asyncio.run(_dispatch(_report_message("no_such_target")))
    assert [message["type"] for message in sent] == ["error"]
    assert "unknown target" in sent[0]["payload"]

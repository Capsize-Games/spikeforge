"""Dispatch routing and payload shapes for the backend WS action."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
import torch
from _fakes import FakeModel, FakeSample, FakeWS

from server.handlers import dispatch
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from spikeforge.topology import registry

pytest.importorskip("nir")

_CONV = {"channels": 2, "num_classes": 3}








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


def _run_message(target: Optional[str] = None) -> ClientMessage:
    """Return a ``deploy_run`` message for the given target."""
    return ClientMessage(
        type="deploy_run",
        name=target,
        train=TrainConfig(topology="conv_net", topology_params=dict(_CONV)),
    )


def _report_message(target: Optional[str] = None) -> ClientMessage:
    """Return a ``deployment_report`` message for the given target."""
    return ClientMessage(
        type="deployment_report",
        name=target,
        train=TrainConfig(topology="conv_net", topology_params=dict(_CONV)),
    )


def test_deploy_run_reference_returns_backend_run() -> None:
    """The reference backend run replies with an executed, compared result."""
    sent = asyncio.run(
        _dispatch(_run_message("reference"), FakeSample(), _conv_model())
    )
    assert [message["type"] for message in sent] == ["backend_run"]
    payload = sent[0]["payload"]
    assert payload["status"] == "ok"
    assert payload["target"] == "reference"
    assert payload["compare"]["readout"]["max_abs"] == 0.0
    assert payload["rewritten"]["ready"] is True
    assert json.dumps(payload)


def test_deploy_run_defaults_to_reference() -> None:
    """A message without a target name runs the reference backend."""
    sent = asyncio.run(
        _dispatch(_run_message(), FakeSample(), _conv_model())
    )
    assert sent[0]["payload"]["target"] == "reference"


def test_deploy_run_unknown_target_errors() -> None:
    """An unknown target name becomes an error payload, not a raise."""
    sent = asyncio.run(_dispatch(_run_message("no_such_target")))
    assert [message["type"] for message in sent] == ["error"]
    assert "unknown target" in sent[0]["payload"]


def test_deploy_run_without_sample_reports_error() -> None:
    """Without a sample the payload is an honest, named error."""
    sent = asyncio.run(_dispatch(_run_message("reference"), None, None))
    payload = sent[0]["payload"]
    assert payload["status"] == "error"
    assert payload["notes"]
    assert json.dumps(payload)


def test_deploy_run_norse_absent_is_honest() -> None:
    """A target whose SDK is absent reports unavailable with a reason."""
    sent = asyncio.run(
        _dispatch(_run_message("norse"), FakeSample(), _conv_model())
    )
    payload = sent[0]["payload"]
    assert payload["status"] == "unavailable"
    assert any("norse" in note for note in payload["notes"])
    assert json.dumps(payload)


def test_deployment_report_carries_the_rewrite_section() -> None:
    """The capability report gains an additive executed-substitution key."""
    sent = asyncio.run(
        _dispatch(_report_message("reference"), FakeSample(), _conv_model())
    )
    payload = sent[0]["payload"]
    assert "rewrite" in payload
    assert payload["rewrite"]["target"] == "reference"
    assert payload["rewrite"]["ready"] is True
    assert json.dumps(payload)

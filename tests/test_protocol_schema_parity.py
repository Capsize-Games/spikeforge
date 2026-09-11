"""Parity between the Python protocol models and the JSON Schema contract.

These four assertions are the protocol contract's regression net: they keep
the pydantic models, the checked-in schemas, the real server emissions, and
``protocol_version.txt`` from drifting apart.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, get_args

import jsonschema
from referencing import Registry, Resource

from server.messages import (
    emit_frame,
    send_download_state,
    send_event_frame,
    send_inference,
    send_introspection,
    send_reconstruction,
    send_run_state,
)
from server.schemas import ClientMessage, EncodeConfig, ServerMessage
from server.session import Session

PROTOCOL_DIR = Path(__file__).resolve().parent.parent / "protocol"


def _load(name: str) -> Dict[str, Any]:
    """Load one schema from the protocol directory."""
    return json.loads((PROTOCOL_DIR / name).read_text(encoding="utf-8"))


def _type_enum(schema: Dict[str, Any]) -> set:
    """Return the envelope schema's closed ``type`` enum as a set."""
    return set(schema["properties"]["type"]["enum"])


def _literal_set(model: Any) -> set:
    """Return a pydantic model's ``type`` Literal members as a set."""
    return set(get_args(model.model_fields["type"].annotation))


def _registry() -> Registry:
    """Register every schema by ``$id`` so relative refs resolve offline."""
    registry = Registry()
    for path in sorted(PROTOCOL_DIR.rglob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        registry = registry.with_resource(
            schema["$id"], Resource.from_contents(schema)
        )
    return registry


def test_client_message_types_match_schema() -> None:
    """The inbound pydantic literal set equals the schema's type enum."""
    schema = _load("client_message.schema.json")
    assert _literal_set(ClientMessage) == _type_enum(schema)


def test_server_message_types_match_schema() -> None:
    """The outbound pydantic literal set equals the schema's type enum."""
    schema = _load("server_message.schema.json")
    assert _literal_set(ServerMessage) == _type_enum(schema)


class _FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


class _FakeEngine:
    """A minimal encoder engine for the outbound helpers under test."""

    def sample_frame(self) -> Any:
        """Return one whole-sample event frame."""
        return [[1.0, 0.0], [0.0, 1.0]]

    def reconstruction(self) -> Dict[str, Any]:
        """Return the two rate reconstructions."""
        frame = [[1.0, 0.0]]
        return {"gain1": frame, "low": frame}

    def spike_frame(self, step: int) -> Any:
        """Return a one-row spike frame for ``step``."""
        return [[float(step)]]


def _capture() -> List[Dict[str, Any]]:
    """Return real ``server/messages.py`` emissions, with a ``kind`` one."""

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        session.set_engine(_FakeEngine(), EncodeConfig())
        ws = _FakeWS()
        await send_event_frame(ws, session)
        await send_reconstruction(ws, session)
        await send_inference(ws, session, {"predicted": 1, "_rasters": {}})
        await send_run_state(ws, session, True)
        await send_download_state(ws, session, {"dataset": "mnist"})
        await emit_frame(ws, session, session.engine, 0)
        await send_introspection(ws, session, "trajectory", {"steps": []})
        return ws.sent

    return asyncio.run(_run())


def test_emitted_server_messages_validate_against_schema() -> None:
    """Real emissions validate, including the previously undeclared kind."""
    schema = _load("server_message.schema.json")
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema, registry=_registry())
    messages = _capture()
    assert any("kind" in message for message in messages)
    for message in messages:
        validator.validate(message)


def test_protocol_version_file_matches_envelope_consts() -> None:
    """``protocol_version.txt`` equals the const in both envelopes."""
    version = (PROTOCOL_DIR / "protocol_version.txt").read_text(
        encoding="utf-8"
    ).strip()
    envelopes = ("client_message.schema.json", "server_message.schema.json")
    for name in envelopes:
        const = _load(name)["properties"]["protocol_version"]["const"]
        assert version == const

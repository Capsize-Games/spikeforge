"""Framework-free response shaping for ``spikeforge_serve.routes``.

Split out of ``routes`` so that module stays within the file-length limit.
Nothing here touches FastAPI directly except the ``HTTPException`` used to
translate a bad request into a 400 status; the actual route wiring lives in
``routes``.
"""

from typing import Any, Dict, Mapping, Tuple

from fastapi import HTTPException, Request

from spikeforge.serving.bundle import DeploymentBundle
from spikeforge_serve.payloads import (
    encoded_flag,
    frames_from,
    prediction_json,
    session_id_from,
)
from spikeforge_serve.service import ServingService


async def _body(request: Request) -> Any:
    """Return the request's JSON body, or raise a 400."""
    try:
        return await request.json()
    except Exception as error:  # any parse failure is a client error
        raise HTTPException(
            status_code=400,
            detail=f"request body is not valid JSON: {error}",
        ) from None


async def _optional_body(request: Request) -> Dict[str, Any]:
    """Return the JSON object body, treating an empty body as ``{}``."""
    raw = await request.body()
    if not raw:
        return {}
    return await _body(request)


def _predict_response(
    serving: ServingService, payload: Any
) -> Dict[str, Any]:
    """Run one predict call and shape its JSON response."""
    try:
        frames = frames_from(payload)
        encoded = encoded_flag(payload)
        session_id = session_id_from(payload)
        predictions = serving.predict(
            frames, session_id=session_id, encoded=encoded
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=str(error)
        ) from None
    return {
        "session_id": session_id,
        "steps": serving.session(session_id).steps,
        "predictions": [prediction_json(i) for i in predictions],
    }


def _reset_response(
    serving: ServingService, payload: Any
) -> Dict[str, Any]:
    """Run one reset call and shape its JSON response."""
    try:
        session_id = session_id_from(payload)
        steps = serving.reset(session_id)
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=400, detail=str(error)
        ) from None
    return {"session_id": session_id, "steps": steps}


def _stream_reply(
    serving: ServingService, session_id: str, message: Any
) -> Tuple[str, Dict[str, Any]]:
    """Return ``(session_id, reply)`` for one stream message."""
    if not isinstance(message, Mapping):
        return session_id, _stream_error(
            "stream message must be a JSON object"
        )
    try:
        return _dispatch_stream_message(serving, session_id, message)
    except (TypeError, ValueError) as error:
        return session_id, _stream_error(str(error))


def _dispatch_stream_message(
    serving: ServingService, session_id: str, message: Mapping
) -> Tuple[str, Dict[str, Any]]:
    """Route one validated stream message to its reply."""
    if "session_id" in message:
        session_id = session_id_from(message)
    if message.get("reset"):
        steps = serving.reset(session_id)
        return session_id, {
            "type": "reset",
            "payload": {"session_id": session_id, "steps": steps},
        }
    if "frame" not in message:
        return session_id, {
            "type": "session",
            "payload": {"session_id": session_id},
        }
    return session_id, _predict_stream_reply(serving, session_id, message)


def _predict_stream_reply(
    serving: ServingService, session_id: str, message: Mapping
) -> Dict[str, Any]:
    """Return the prediction reply for a ``frame`` stream message."""
    encoded = encoded_flag(message)
    prediction = serving.predict(
        [message["frame"]], session_id=session_id, encoded=encoded
    )[0]
    payload = prediction_json(prediction)
    payload["session_id"] = session_id
    payload["steps"] = serving.session(session_id).steps
    return {"type": "prediction", "payload": payload}


def _stream_error(detail: str) -> Dict[str, Any]:
    """Return a stream error envelope for ``detail``."""
    return {
        "type": "error",
        "payload": {"type": "bad_request", "message": detail},
    }


def _bundle_info(bundle: DeploymentBundle) -> Dict[str, Any]:
    """Return the self-describing metadata a client needs for the model."""
    manifest = bundle.manifest
    spec = bundle.encode_spec()
    info = _manifest_fields(manifest)
    info["path"] = bundle.path
    info.update(_encode_fields(spec))
    return info


def _manifest_fields(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the manifest-derived fields of the bundle-info payload."""
    return {
        "format": manifest.get("format"),
        "version": manifest.get("version"),
        "topology": manifest.get("topology"),
        "topology_params": dict(manifest.get("topology_params") or {}),
        "spec": manifest.get("spec"),
        "protocol_version": manifest.get("protocol_version"),
        "encode_spec_version": manifest.get("encode_spec_version"),
        "num_classes": manifest.get("num_classes"),
        "num_steps": manifest.get("num_steps"),
        "label_map": dict(manifest.get("label_map") or {}),
        "expected_metrics": dict(manifest.get("expected_metrics") or {}),
        "library_versions": dict(manifest.get("library_versions") or {}),
    }


def _encode_fields(spec: Any) -> Dict[str, Any]:
    """Return the encode-spec-derived fields of the bundle-info payload."""
    encode = spec.to_dict()
    return {
        "encode_spec": encode,
        "encode_digest": spec.digest(),
        "input_size": encode["input_size"],
    }

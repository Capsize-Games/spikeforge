"""P4: export a bundle, serve it, and prove streaming/batch parity.

The checkpoint the recipe wrote carries the frozen encode spec and the frozen
windowing contract in its metadata, so
:func:`~spikeforge.serving.bundle.build` freezes both into the ``.spkf``
artifact. :func:`predict_batch` runs the whole window batch through the shared
temporal loop, while :func:`stream_logits` drives a stateful
:class:`~spikeforge.serving.session.InferenceSession` one timestep at a time;
:func:`parity_report` asserts the two agree within tolerance. The same session
backs :class:`~spikeforge_serve.service.ServingService`, so the served path is
the parity path.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from spikeforge.network import model_store
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.bundle import build as build_deployment
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.serving.prediction import Prediction
from spikeforge.serving.session import InferenceSession
from spikeforge.simulator.runner import run
from spikeforge.streaming.encoding import encode_windows
from spikeforge.streaming.recipe import TOPOLOGY, StreamTrainingResult
from spikeforge.streaming.window_spec import WindowSpec

#: Bundle/session source accepted by the helpers.
BundleSource = Union[str, DeploymentBundle]
#: Default numerical tolerance for streaming-vs-batch parity.
PARITY_ATOL: float = 1e-5


def _checkpoint_spec_meta(
    result: StreamTrainingResult, window_spec: WindowSpec
) -> Dict[str, Any]:
    """Return the topology/encode/window fields of the checkpoint meta."""
    config = result.config
    return {
        "topology": TOPOLOGY,
        "topology_params": config.topology_params(),
        "spec": result.spec.to_dict(),
        "num_classes": int(config.num_classes),
        "num_steps": int(config.num_steps),
        "coding": str(config.coding),
        "encode_spec": result.encode_spec.to_dict(),
        "window": window_spec.to_dict(),
        "delta_along_window": bool(config.delta_along_window),
    }


def _checkpoint_meta(
    result: StreamTrainingResult,
    window_spec: WindowSpec,
    label_map: Optional[Mapping[int, str]],
    metrics: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return the metadata block a streaming checkpoint records."""
    meta = _checkpoint_spec_meta(result, window_spec)
    meta["labels"] = {
        int(key): str(value) for key, value in (label_map or {}).items()
    }
    meta["metrics"] = dict(metrics or {})
    return meta


def save_checkpoint(
    name: str,
    result: StreamTrainingResult,
    window_spec: WindowSpec,
    *,
    label_map: Optional[Mapping[int, str]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> str:
    """Persist a trained trunk plus its frozen contracts, and return its path.

    The metadata records the topology spec, the encode spec, the windowing
    contract, and the anomaly score distribution, so a bundle built from this
    checkpoint is self-describing.
    """
    meta = _checkpoint_meta(result, window_spec, label_map, metrics)
    history = [dict(entry) for entry in result.history]
    return model_store.save(name, result.module, meta, history, None)


def _resolved_encode(
    meta: Dict[str, Any], encode_spec: Optional[EncodeSpec]
) -> EncodeSpec:
    """Return ``encode_spec``, or the checkpoint's own frozen spec."""
    if encode_spec is not None:
        return encode_spec
    return EncodeSpec.from_mapping(meta.get("encode_spec"))


def _bundle_preprocessing(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return the frozen z-score/window preprocessing block."""
    return {
        "window": dict(meta.get("window") or {}),
        "delta_over_window": bool(meta.get("delta_along_window", False)),
    }


def build_bundle(
    checkpoint: str, out: Optional[str] = None, *,
    encode_spec: Optional[EncodeSpec] = None,
    label_map: Optional[Mapping[int, str]] = None,
    expected_metrics: Optional[Mapping[str, Any]] = None,
) -> DeploymentBundle:
    """Freeze a checkpoint into a :class:`DeploymentBundle`.

    Reads the windowing/encode contract from checkpoint metadata unless
    ``encode_spec`` overrides it.
    """
    meta = dict(model_store.load(checkpoint).get("meta") or {})
    return build_deployment(
        checkpoint,
        out=out,
        encode_config=_resolved_encode(meta, encode_spec).to_dict(),
        preprocessing=_bundle_preprocessing(meta),
        label_map=dict(label_map or meta.get("labels") or {}),
        expected_metrics=dict(expected_metrics or meta.get("metrics") or {}),
    )


def load_session(bundle: BundleSource) -> InferenceSession:
    """Return a stateful session over ``bundle`` (a path or an object)."""
    return InferenceSession.load(bundle)


def predict_batch(
    module: Any,
    windows: torch.Tensor,
    encode_spec: Any,
    *,
    delta_along_window: bool = False,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits for a whole batch (reference)."""
    module.eval()
    spikes = encode_windows(
        windows, encode_spec, delta_along_window=delta_along_window
    )
    with torch.no_grad():
        return run(module, spikes).logits.mean(dim=1)


def _stream_one_window(
    session: InferenceSession,
    window: torch.Tensor,
    encode_spec: Any,
    delta_along_window: bool,
) -> torch.Tensor:
    """Reset the session, then stream one window's spikes step by step."""
    spikes = encode_windows(
        window, encode_spec, delta_along_window=delta_along_window
    )
    session.reset()
    prediction = None
    for step in range(int(spikes.size(0))):
        prediction = session.step(spikes[step])
    if prediction is None:
        raise ValueError("an encoded window produced no timesteps")
    return prediction.mean_logits.mean(dim=1).reshape(-1)


def stream_logits(
    session: InferenceSession, windows: torch.Tensor, encode_spec: Any,
    *, delta_along_window: bool = False,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits streaming one window at a time.

    Reset per window through the stateful session -- exactly the serving
    path, so this loop and a whole-tensor ``run`` agree by construction.
    """
    rows = [
        _stream_one_window(
            session, windows[i:i + 1], encode_spec, delta_along_window
        )
        for i in range(int(windows.size(0)))
    ]
    if not rows:
        return torch.zeros(0, 0)
    return torch.stack(rows, dim=0)


def _argmax_labels(logits: torch.Tensor) -> List[int]:
    """Return the predicted class per row of ``logits``."""
    return [int(value) for value in logits.argmax(dim=-1).tolist()]


def _parity_fields(
    batch: torch.Tensor, stream: torch.Tensor, atol: float
) -> Dict[str, Any]:
    """Return the parity report fields for computed batch/stream logits."""
    difference = float((batch - stream).abs().max().item())
    return {
        "batch_logits": batch,
        "stream_logits": stream,
        "max_abs_diff": difference,
        "within_tolerance": difference <= float(atol),
        "batch_labels": _argmax_labels(batch),
        "stream_labels": _argmax_labels(stream),
    }


def parity_report(
    module: Any, session: InferenceSession, windows: torch.Tensor,
    encode_spec: Any, *, delta_along_window: bool = False,
    atol: float = PARITY_ATOL,
) -> Dict[str, Any]:
    """Return the streaming-vs-batch parity evidence for ``windows``."""
    batch = predict_batch(
        module, windows, encode_spec, delta_along_window=delta_along_window
    )
    stream = stream_logits(
        session, windows, encode_spec, delta_along_window=delta_along_window
    )
    return _parity_fields(batch, stream, atol)


@dataclass(frozen=True)
class ServeResult:
    """One served window: the final prediction and its mean logits."""

    prediction: Prediction
    mean_logits: torch.Tensor


def _predict_window(
    service: Any, window: torch.Tensor, encode_spec: Any,
    session_id: str, delta_along_window: bool,
) -> Prediction:
    """Encode ``window`` and predict its final frame through ``service``."""
    spikes = encode_windows(
        window, encode_spec, delta_along_window=delta_along_window
    )
    frames = [spikes[step] for step in range(int(spikes.size(0)))]
    predictions = service.predict(frames, session_id=session_id, encoded=True)
    if not predictions:
        raise ValueError("an encoded window produced no timesteps")
    return predictions[-1]


def serve_window(
    service: Any,
    window: torch.Tensor,
    encode_spec: Any,
    *,
    session_id: str = "default",
    delta_along_window: bool = False,
) -> ServeResult:
    """Drive one raw window through ``ServingService`` and return its readout.

    Encoded, then handed to the service as pre-encoded frames; the session
    is reset first so the result is a function of this window alone.
    """
    service.reset(session_id)
    final = _predict_window(
        service, window, encode_spec, session_id, delta_along_window
    )
    return ServeResult(
        prediction=final, mean_logits=final.mean_logits.mean(dim=1).reshape(-1)
    )

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
    config = result.config
    meta: Dict[str, Any] = {
        "topology": TOPOLOGY,
        "topology_params": config.topology_params(),
        "spec": result.spec.to_dict(),
        "num_classes": int(config.num_classes),
        "num_steps": int(config.num_steps),
        "coding": str(config.coding),
        "encode_spec": result.encode_spec.to_dict(),
        "window": window_spec.to_dict(),
        "delta_along_window": bool(config.delta_along_window),
        "labels": {
            int(key): str(value)
            for key, value in (label_map or {}).items()
        },
        "metrics": dict(metrics or {}),
    }
    history = [
        dict(entry) for entry in result.history
    ]
    return model_store.save(name, result.module, meta, history, None)


def build_bundle(
    checkpoint: str,
    out: Optional[str] = None,
    *,
    encode_spec: Optional[EncodeSpec] = None,
    label_map: Optional[Mapping[int, str]] = None,
    expected_metrics: Optional[Mapping[str, Any]] = None,
) -> DeploymentBundle:
    """Freeze a checkpoint into a :class:`DeploymentBundle`.

    The windowing contract and the encode spec are read from the checkpoint's
    metadata unless ``encode_spec`` overrides the latter. The preprocessing
    block carries the frozen z-score statistics so a served window is
    normalised exactly as a training window was.
    """
    stored = model_store.load(checkpoint)
    meta = dict(stored.get("meta") or {})
    resolved = (
        encode_spec
        if encode_spec is not None
        else EncodeSpec.from_mapping(meta.get("encode_spec"))
    )
    preprocessing = {
        "window": dict(meta.get("window") or {}),
        "delta_over_window": bool(meta.get("delta_along_window", False)),
    }
    return build_deployment(
        checkpoint,
        out=out,
        encode_config=resolved.to_dict(),
        preprocessing=preprocessing,
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


def stream_logits(
    session: InferenceSession,
    windows: torch.Tensor,
    encode_spec: Any,
    *,
    delta_along_window: bool = False,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits streaming one window at a time.

    Every window is presented as its own spike train through the stateful
    session (reset per window), which is exactly the path the serving service
    drives, so a whole-tensor ``run`` and this loop agree by construction.
    """
    rows: List[torch.Tensor] = []
    count = int(windows.size(0))
    for index in range(count):
        spikes = encode_windows(
            windows[index:index + 1],
            encode_spec,
            delta_along_window=delta_along_window,
        )
        session.reset()
        prediction = None
        for step in range(int(spikes.size(0))):
            prediction = session.step(spikes[step])
        if prediction is None:
            raise ValueError("an encoded window produced no timesteps")
        rows.append(prediction.mean_logits.mean(dim=1).reshape(-1))
    if not rows:
        return torch.zeros(0, 0)
    return torch.stack(rows, dim=0)


def parity_report(
    module: Any,
    session: InferenceSession,
    windows: torch.Tensor,
    encode_spec: Any,
    *,
    delta_along_window: bool = False,
    atol: float = PARITY_ATOL,
) -> Dict[str, Any]:
    """Return the streaming-vs-batch parity evidence for ``windows``."""
    batch = predict_batch(
        module, windows, encode_spec, delta_along_window=delta_along_window
    )
    stream = stream_logits(
        session,
        windows,
        encode_spec,
        delta_along_window=delta_along_window,
    )
    difference = float((batch - stream).abs().max().item())
    return {
        "batch_logits": batch,
        "stream_logits": stream,
        "max_abs_diff": difference,
        "within_tolerance": difference <= float(atol),
        "batch_labels": [
            int(value) for value in batch.argmax(dim=-1).tolist()
        ],
        "stream_labels": [
            int(value) for value in stream.argmax(dim=-1).tolist()
        ],
    }


@dataclass(frozen=True)
class ServeResult:
    """One served window: the final prediction and its mean logits."""

    prediction: Prediction
    mean_logits: torch.Tensor


def serve_window(
    service: Any,
    window: torch.Tensor,
    encode_spec: Any,
    *,
    session_id: str = "default",
    delta_along_window: bool = False,
) -> ServeResult:
    """Drive one raw window through ``ServingService`` and return its readout.

    The window is encoded with the shared encode function, then handed to the
    service as pre-encoded timestep frames. The session is reset first so the
    result is a function of this window alone.
    """
    service.reset(session_id)
    spikes = encode_windows(
        window, encode_spec, delta_along_window=delta_along_window
    )
    frames = [spikes[step] for step in range(int(spikes.size(0)))]
    predictions = service.predict(frames, session_id=session_id, encoded=True)
    if not predictions:
        raise ValueError("an encoded window produced no timesteps")
    final = predictions[-1]
    return ServeResult(
        prediction=final, mean_logits=final.mean_logits.mean(dim=1).reshape(-1)
    )

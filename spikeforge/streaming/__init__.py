"""UC-1: streaming time-series classification / anomaly detection (MVP P0-P4).

This subpackage is the thin offline/online glue for the first production use
case. It reuses the shared enablers rather than re-implementing them:

* the frozen encode contract (:class:`~spikeforge.serving.encode_spec.\
EncodeSpec`) and the single :class:`~spikeforge.encoding.spike_encoder.\
SpikeEncoder`, so training and serving encode identically;
* the :func:`~spikeforge.simulator.runner.run` temporal loop and the
  ``sequence_mlp`` topology preset, so the SNN trunk is the one the simulator
  already validates;
* the portable :class:`~spikeforge.serving.bundle.DeploymentBundle` and the
  stateful :class:`~spikeforge.serving.session.InferenceSession`, so a stream
  and a whole-tensor ``run`` agree by construction.

The modules are pure torch and stdlib; no pydantic, FastAPI, or optional SDK
is imported, so a headless core install can train and serve a model.
"""

from spikeforge.streaming.encoding import (
    delta_over_window,
    encode_from_bundle,
    encode_windows,
)
from spikeforge.streaming.recipe import (
    StreamTrainConfig,
    StreamTrainingResult,
    anomaly_rule,
    anomaly_scores,
    anomaly_threshold,
    auroc,
    build_model,
    class_metrics,
    evaluate,
    fit_threshold,
    predict_logits,
    set_seed,
    train_classifier,
)
from spikeforge.streaming.serving import (
    build_bundle,
    load_session,
    parity_report,
    predict_batch,
    save_checkpoint,
    serve_window,
    stream_logits,
)
from spikeforge.streaming.stream_source import (
    StreamDataset,
    StreamSpec,
    StreamSplits,
    build_datasets,
    generate_stream,
)
from spikeforge.streaming.window_spec import (
    WINDOW_SPEC_VERSION,
    WindowSpec,
    fit_window_spec,
    normalize_windows,
    window_stream,
)

__all__ = [
    "WINDOW_SPEC_VERSION",
    "StreamDataset",
    "StreamSpec",
    "StreamSplits",
    "StreamTrainConfig",
    "StreamTrainingResult",
    "WindowSpec",
    "anomaly_rule",
    "anomaly_scores",
    "anomaly_threshold",
    "auroc",
    "build_bundle",
    "build_datasets",
    "build_model",
    "class_metrics",
    "delta_over_window",
    "encode_from_bundle",
    "encode_windows",
    "evaluate",
    "fit_threshold",
    "fit_window_spec",
    "generate_stream",
    "load_session",
    "normalize_windows",
    "parity_report",
    "predict_batch",
    "predict_logits",
    "save_checkpoint",
    "serve_window",
    "set_seed",
    "stream_logits",
    "train_classifier",
    "window_stream",
]

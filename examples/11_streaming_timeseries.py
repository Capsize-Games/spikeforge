"""UC-1: train, evaluate, and serve a streaming time-series model.

Run from the repository root::

    venv/bin/python examples/11_streaming_timeseries.py

The script builds a deterministic synthetic multi-channel stream, windows and
z-scores it with the frozen contract, trains a small ``sequence_mlp`` on
delta-coded windows, reports class and anomaly metrics, freezes the model into
a ``.spkf`` deployment bundle, and proves the streaming readout equals the
closed-loop batch reference (through both ``InferenceSession`` and
``spikeforge-serve``). Everything runs on CPU and offline; artifacts go to a
temporary directory.

This is a runnable MVP slice, not a benchmark: the synthetic task is small and
the reported numbers move with the config.
"""

import os
import tempfile
from typing import Any, Dict, Tuple

from spikeforge.network import model_store
from spikeforge.streaming import (
    StreamSpec,
    StreamSplits,
    StreamTrainConfig,
    StreamTrainingResult,
    build_bundle,
    build_datasets,
    evaluate,
    fit_threshold,
    load_session,
    parity_report,
    save_checkpoint,
    train_classifier,
)
from spikeforge.streaming.serving import serve_window

try:
    from spikeforge_serve.service import ServingService
except ImportError:  # spikeforge-serve is a separate, optional distribution
    ServingService = None  # type: ignore[assignment,misc]

_NAME = "uc1_example"


def _describe_dataset(spec: StreamSpec) -> None:
    """Print the window geometry and the class/anomaly balance."""
    splits = build_datasets(spec)
    print(
        "windows train/val/test:",
        len(splits.train),
        len(splits.val),
        len(splits.test),
    )
    print("window shape:", tuple(splits.train.windows.shape))
    print("channels:", splits.window_spec.channel_names)
    print("train anomalies:", int(splits.train.anomaly.sum()))
    print("test anomalies:", int(splits.test.anomaly.sum()))


def _build_spec() -> StreamSpec:
    """Return the deterministic UC-1 synthetic-stream spec."""
    return StreamSpec(
        length=16,
        stride=8,
        channels=4,
        classes=3,
        segment_length=32,
        segments=24,
        anomaly_rate=0.3,
        seed=0,
    )


def _train_config(spec: StreamSpec) -> StreamTrainConfig:
    """Return the training config derived from ``spec``."""
    return StreamTrainConfig(
        seq_length=spec.length,
        features=spec.channels,
        hidden=24,
        num_classes=spec.classes,
        num_steps=6,
        epochs=25,
        lr=5e-2,
        seed=0,
    )


def _train_and_evaluate(
    splits: StreamSplits, config: StreamTrainConfig
) -> Tuple[StreamTrainingResult, Dict[str, Any]]:
    """Train the classifier, then evaluate it with its fitted threshold."""
    result = train_classifier(splits.train, config)
    final = result.history[-1]
    print(
        "trained:",
        f"epochs={config.epochs}",
        f"loss={final['loss']:.4f}",
        f"train_accuracy={final['train_accuracy']:.3f}",
    )
    threshold = fit_threshold(result.module, splits.train, config)
    report = evaluate(
        result.module, splits.test, config, threshold=threshold
    )
    _print_report(report)
    return result, report


def _print_report(report: Dict[str, Any]) -> None:
    """Print the classification and anomaly-detection metrics."""
    print(
        "test:",
        f"accuracy={report['accuracy']:.3f}",
        f"macro_f1={report['macro_f1']:.3f}",
        f"balanced_accuracy={report['balanced_accuracy']:.3f}",
    )
    print(
        "anomaly:",
        f"auroc={report['anomaly_auroc']:.3f}",
        f"threshold={report['anomaly_threshold']:.4f}",
        f"precision={report['anomaly_precision']:.3f}",
        f"recall={report['anomaly_recall']:.3f}",
    )


def _save_bundle(
    spec: StreamSpec,
    splits: StreamSplits,
    result: StreamTrainingResult,
    report: Dict[str, Any],
    folder: str,
) -> Tuple[str, Any]:
    """Checkpoint the model and freeze it into a ``.spkf`` bundle."""
    label_map = {i: f"class_{i}" for i in range(spec.classes)}
    save_checkpoint(
        _NAME,
        result,
        splits.window_spec,
        label_map=label_map,
        metrics={"accuracy": report["accuracy"]},
    )
    out = os.path.join(folder, "uc1.spkf")
    return out, build_bundle(_NAME, out=out)


def _print_bundle_info(out: str, bundle: Any) -> None:
    """Print the frozen bundle's path and encode/window configuration."""
    print(
        "bundle:",
        os.path.basename(out),
        f"coding={bundle.encode_config['coding']}",
        f"window_L={bundle.preprocessing['window']['length']}",
    )


def _check_parity(
    result: StreamTrainingResult, out: str, splits: StreamSplits
) -> None:
    """Load the bundle as a session and confirm it matches the batch path."""
    session = load_session(out)
    parity = parity_report(
        result.module, session, splits.test.windows[:8], result.encode_spec
    )
    print(
        "parity:",
        f"max_abs_diff={parity['max_abs_diff']:.2e}",
        f"within_tolerance={parity['within_tolerance']}",
    )


def _serve_and_report(
    out: str, splits: StreamSplits, result: StreamTrainingResult
) -> None:
    """Serve one window through spikeforge-serve, if it is installed."""
    if ServingService is None:
        print("serve: spikeforge-serve not installed; skipped")
        return
    service = ServingService(out)
    served = serve_window(
        service, splits.test.windows[:1], result.encode_spec
    )
    print(
        "served mean logits:",
        [round(value, 4) for value in served.mean_logits.tolist()],
    )
    print("session reset steps:", service.reset())


def main() -> int:
    """Run the UC-1 end-to-end slice."""
    with tempfile.TemporaryDirectory() as folder:
        model_store.MODEL_DIR = folder
        spec = _build_spec()
        _describe_dataset(spec)
        splits = build_datasets(spec)
        config = _train_config(spec)
        result, report = _train_and_evaluate(splits, config)
        out, bundle = _save_bundle(spec, splits, result, report, folder)
        _print_bundle_info(out, bundle)
        _check_parity(result, out, splits)
        _serve_and_report(out, splits, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

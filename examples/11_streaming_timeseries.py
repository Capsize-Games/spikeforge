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

from spikeforge.network import model_store
from spikeforge.streaming import (
    StreamSpec,
    StreamTrainConfig,
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


def main() -> int:
    """Run the UC-1 end-to-end slice."""
    with tempfile.TemporaryDirectory() as folder:
        model_store.MODEL_DIR = folder
        spec = StreamSpec(
            length=16,
            stride=8,
            channels=4,
            classes=3,
            segment_length=32,
            segments=24,
            anomaly_rate=0.3,
            seed=0,
        )
        _describe_dataset(spec)
        splits = build_datasets(spec)

        config = StreamTrainConfig(
            seq_length=spec.length,
            features=spec.channels,
            hidden=24,
            num_classes=spec.classes,
            num_steps=6,
            epochs=25,
            lr=5e-2,
            seed=0,
        )
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

        label_map = {index: f"class_{index}" for index in range(spec.classes)}
        save_checkpoint(
            _NAME,
            result,
            splits.window_spec,
            label_map=label_map,
            metrics={"accuracy": report["accuracy"]},
        )
        out = os.path.join(folder, "uc1.spkf")
        bundle = build_bundle(_NAME, out=out)
        print(
            "bundle:",
            os.path.basename(out),
            f"coding={bundle.encode_config['coding']}",
            f"window_L={bundle.preprocessing['window']['length']}",
        )

        session = load_session(out)
        parity = parity_report(
            result.module, session, splits.test.windows[:8], result.encode_spec
        )
        print(
            "parity:",
            f"max_abs_diff={parity['max_abs_diff']:.2e}",
            f"within_tolerance={parity['within_tolerance']}",
        )

        if ServingService is None:
            print("serve: spikeforge-serve not installed; skipped")
        else:
            service = ServingService(out)
            served = serve_window(
                service, splits.test.windows[:1], result.encode_spec
            )
            print(
                "served mean logits:",
                [round(value, 4) for value in served.mean_logits.tolist()],
            )
            print("session reset steps:", service.reset())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

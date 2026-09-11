"""Encode one image with every spike coding and inspect the result.

Uses a registry MNIST sample and the shared ``SpikeEncoder`` to build rate,
latency, delta, and random codes, then reports each coding's firing rate,
sparsity, and whether a reconstruction is supported.

Run from the repository root::

    venv/bin/python examples/03_encode_and_inspect.py

The ``rate`` and ``random`` codings are stochastic, so their exact numbers
vary run to run; the printed values are therefore illustrative. The
reconstructions are explicitly approximate and documented in the report's
``approximation`` field.
"""

from snn_interpreter.data.sample_source import SampleSource
from snn_interpreter.encoding.spike_encoder import SpikeEncoder
from snn_interpreter.introspection.encoding import encoding_report

#: Codings the shared encoder supports, in report order.
CODINGS = ("rate", "latency", "delta", "random")


def main() -> int:
    """Encode one sample per coding and print its introspection report."""
    image = SampleSource(dataset="mnist").image(0)
    for coding in CODINGS:
        encoder = SpikeEncoder(coding=coding, num_steps=10, seed=0)
        report = encoding_report(image, encoder)
        print(
            f"{coding:<7} firing_rate={report['firing_rate']:.4f} "
            f"sparsity={report['sparsity']:.4f} "
            f"supported={report['reconstruction_supported']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

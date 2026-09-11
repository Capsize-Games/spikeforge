"""Export a topology's single step to ONNX and re-import it.

Shows the ONNX bridge's structural journey: ``export_topology`` writes a
one-step ONNX graph stamped with the spec metadata, ``roundtrip`` re-imports
it and reports whether the spec survives exactly, and ``import_report``
reads a written file back.

Run from the repository root::

    venv/bin/python examples/09_onnx_roundtrip.py

The exported file is **not** a complete temporal SNN: the time loop stays in
the simulator, so another runtime will not reproduce multi-timestep
dynamics. Requires the ``onnx`` extra; without it the script degrades to a
clear message. The equivalent shell commands are ``spikeforge-verify
onnx-export``, ``spikeforge-verify onnx-import``, and
``spikeforge-verify onnx-roundtrip``.
"""

import tempfile
from pathlib import Path

from spikeforge.onnx_bridge import (
    available,
    export_topology,
    import_report,
    roundtrip,
)

#: Topology exported by this example.
TOPOLOGY = "conv_net"


def main() -> int:
    """Export, round-trip, and re-import a topology through ONNX."""
    if not available():
        print("the `onnx` extra is not installed; skipping.")
        print("install it with: pip install -e '.[onnx]'")
        return 0
    report = roundtrip(TOPOLOGY)
    print("topology:", report["topology"])
    print("source:", report["source"], "| identical:", report["identical"])
    print("ops:", report["ops"])
    print("temporal:", report["temporal"])
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.onnx")
        export_topology(TOPOLOGY, path)
        summary = import_report(path)
        print("re-import source:", summary["source"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

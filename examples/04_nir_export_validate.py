"""Export a topology to NIR and validate it against snnTorch.

Builds the ``conv_net`` preset, renders it into a ``nir.NIRGraph`` with
``to_nir``, prints the node inventory from ``graph_summary``, and runs the
independent interpreter through ``validate`` to report numerical drift.

Run from the repository root::

    venv/bin/python examples/04_nir_export_validate.py

Nothing here reads a dataset: the spike fixture is deterministic and
offline. The membrane residual you see is the documented Euler-versus-
zero-order-hold difference, reported rather than hidden.
"""

from snn_interpreter.cli import fixture
from snn_interpreter.nir_bridge import graph_summary, to_nir, validate

#: Topology exported by this example.
TOPOLOGY = "conv_net"


def main() -> int:
    """Export a topology, summarize it, and print its drift report."""
    spec, module, spikes = fixture.synthetic_input(TOPOLOGY, 4, 1, 0)
    graph = to_nir(spec, module)
    summary = graph_summary(graph)
    print("topology:", TOPOLOGY, "| input:", tuple(spikes.shape))
    print("nodes:", len(summary["nodes"]))
    print("kinds:", [node["kind"] for node in summary["nodes"]])
    report = validate(spec, module, spikes)
    print("within_tolerance:", report["within_tolerance"])
    print("worst:", report["worst"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Estimate energy for a topology, sparse versus dense.

Counts SOP/MAC/AC for the ``conv_net`` fixture and maps them onto the
``reference`` target's declared cost table, once on the event-driven
(sparse) path and once on the dense baseline.

Run from the repository root::

    venv/bin/python examples/07_energy_sparse_dense.py

Every number is an **estimate** from a ``"measured": false`` table. It is
sound for comparing models and for sparse-versus-dense trade-offs, but it is
not a power budget. A target with no declared table reports
``basis: "unavailable"`` and ``null`` numbers rather than a fabricated
figure.
"""

from typing import Any, Dict

from snn_interpreter.energy.accounting import measure_topology

#: Topology and target accounted by this example.
TOPOLOGY = "conv_net"
TARGET = "reference"


def account(sparse: bool) -> Dict[str, Any]:
    """Return the energy report for the sparse or dense fixture."""
    _result, report = measure_topology(
        TOPOLOGY, TARGET, steps=4, sparse=sparse
    )
    return report.to_dict()


def main() -> int:
    """Print the sparse and dense estimates and their ratio."""
    sparse = account(True)
    dense = account(False)
    print("basis:", sparse["basis"], "| estimate:", sparse["estimate"])
    print("sparse sop:", sparse["ops"]["sop"], "mac:", sparse["ops"]["mac"])
    print("dense  sop:", dense["ops"]["sop"], "mac:", dense["ops"]["mac"])
    print(
        f"sop_over_mac sparse={sparse['efficiency']['sop_over_mac']:.4f} "
        f"dense={dense['efficiency']['sop_over_mac']:.4f}"
    )
    print(
        f"energy total sparse={sparse['energy']['total_pj']:.1f} pJ "
        f"dense={sparse['energy']['dense_pj']:.1f} pJ"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

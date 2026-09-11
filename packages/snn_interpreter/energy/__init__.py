"""Energy/latency accounting built on the event-driven runtime.

Public API: :func:`account` maps operation counts to a target's declared cost
table (or reports ``basis: "unavailable"``), and :func:`measure_topology` runs
a topology's sparse path and accounts it in one call. Every report is an
estimate unless a real device reports its own timing.
"""

from snn_interpreter.energy.accounting import (
    BASIS_MEASURED,
    BASIS_TABLE,
    BASIS_UNAVAILABLE,
    account,
    measure_topology,
)
from snn_interpreter.energy.cost_table import REQUIRED_KEYS, CostTable
from snn_interpreter.energy.errors import EnergyCostError, EnergyError
from snn_interpreter.energy.report import EnergyReport
from snn_interpreter.energy.target_costs import declared, load, names

__all__ = [
    "BASIS_MEASURED",
    "BASIS_TABLE",
    "BASIS_UNAVAILABLE",
    "REQUIRED_KEYS",
    "CostTable",
    "EnergyCostError",
    "EnergyError",
    "EnergyReport",
    "account",
    "declared",
    "load",
    "measure_topology",
    "names",
]

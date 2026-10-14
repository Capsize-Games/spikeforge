#!/usr/bin/env python3
"""Packaging-topology guards for the ARCH-0001 multi-distribution layout.

Four release-blocking health metrics from
``plans/arch-0001-decision-metrics.md`` that the runtime blockers do not cover:

* **disjoint import roots** - no top-level import root may be shipped by two
  distributions (the PEP 420 / packaging-profile health metric);
* **console-script ownership** - no console-script name may appear in two
  distributions;
* **compatibility pin** - every non-core distribution's ``spikeforge~=X.Y.0``
  pin must equal the core version recorded in ``compatibility.json``. This
  mirrors the ``release.yml`` verify step exactly: a distribution listed in
  :data:`CORE_FREE_DISTRIBUTIONS` (currently ``spikeforge-clients``, which must
  never pull the torch-backed core) is exempt from the pin but must declare no
  ``spikeforge`` dependency at all, while every other satellite must declare
  exactly the matrix pin;
* **coordinated minor bump** - a core MINOR bump across the last two
  ``compatibility.json`` rows requires every pinned satellite to ship its
  own version bump in the new row. The 0.6.0 release forgot serve,
  targets, and io, and the recorded combination could not be installed.

Import roots and scripts are read from each distribution's ``pyproject.toml``.
When built wheels are passed with ``--core-wheel``, ``--server-wheel``,
``--targets-wheel``, and ``--hub-wheel`` the import-root check runs against the
actual archives instead, so CI proves the artifact and not just the intent.

The checks live in :mod:`scripts.packaging_layout`,
:mod:`scripts.packaging_check_topology`, :mod:`scripts.packaging_check_bump`,
and :mod:`scripts.packaging_check_pins`; this module keeps the CLI entry
point and the historical import surface (#149).

Usage::

    python scripts/check_packaging_guards.py \
        --core-wheel dist/spikeforge-*.whl \
        --server-wheel dist/spikeforge_server-*.whl \
        --serve-wheel dist/spikeforge_serve-*.whl \
        --clients-wheel dist/spikeforge_clients-*.whl \
        --targets-wheel dist/spikeforge_targets-*.whl \
        --hub-wheel dist/spikeforge_hub-*.whl \
        --io-wheel dist/spikeforge_io-*.whl
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

try:  # imported as part of the scripts package
    from scripts.packaging_check_bump import (
        _releases,
        check_coordinated_minor_bump,
    )
    from scripts.packaging_check_pins import check_core_pin, check_matrix_pins
    from scripts.packaging_check_topology import (
        check_disjoint_roots,
        check_script_ownership,
    )
    from scripts.packaging_layout import (
        CLIENTS_DISTRIBUTION,
        CLIENTS_PYPROJECT,
        CORE_DISTRIBUTION,
        CORE_FREE_DISTRIBUTIONS,
        CORE_PYPROJECT,
        HUB_DISTRIBUTION,
        HUB_PYPROJECT,
        IO_DISTRIBUTION,
        IO_PYPROJECT,
        SERVE_DISTRIBUTION,
        SERVE_PYPROJECT,
        SERVER_DISTRIBUTION,
        SERVER_PYPROJECT,
        TARGETS_DISTRIBUTION,
        TARGETS_PYPROJECT,
        roots_from_pyproject,
        roots_from_wheel,
    )
except ImportError:  # run directly as scripts/check_packaging_guards.py
    from packaging_check_bump import _releases, check_coordinated_minor_bump
    from packaging_check_pins import check_core_pin, check_matrix_pins
    from packaging_check_topology import (
        check_disjoint_roots,
        check_script_ownership,
    )
    from packaging_layout import (
        CLIENTS_DISTRIBUTION,
        CLIENTS_PYPROJECT,
        CORE_DISTRIBUTION,
        CORE_FREE_DISTRIBUTIONS,
        CORE_PYPROJECT,
        HUB_DISTRIBUTION,
        HUB_PYPROJECT,
        IO_DISTRIBUTION,
        IO_PYPROJECT,
        SERVE_DISTRIBUTION,
        SERVE_PYPROJECT,
        SERVER_DISTRIBUTION,
        SERVER_PYPROJECT,
        TARGETS_DISTRIBUTION,
        TARGETS_PYPROJECT,
        roots_from_pyproject,
        roots_from_wheel,
    )

__all__ = [
    "CLIENTS_DISTRIBUTION",
    "CORE_DISTRIBUTION",
    "CORE_FREE_DISTRIBUTIONS",
    "check_coordinated_minor_bump",
    "check_core_pin",
    "main",
]


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ARCH-0001 packaging-topology guards."
    )
    parser.add_argument("--core-wheel", type=Path, default=None)
    parser.add_argument("--server-wheel", type=Path, default=None)
    parser.add_argument("--serve-wheel", type=Path, default=None)
    parser.add_argument("--clients-wheel", type=Path, default=None)
    parser.add_argument("--targets-wheel", type=Path, default=None)
    parser.add_argument("--hub-wheel", type=Path, default=None)
    parser.add_argument("--io-wheel", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the guards and return a process exit code."""
    args = _parse_args(argv)
    pyprojects = {
        CORE_DISTRIBUTION: CORE_PYPROJECT,
        SERVER_DISTRIBUTION: SERVER_PYPROJECT,
        SERVE_DISTRIBUTION: SERVE_PYPROJECT,
        CLIENTS_DISTRIBUTION: CLIENTS_PYPROJECT,
        TARGETS_DISTRIBUTION: TARGETS_PYPROJECT,
        HUB_DISTRIBUTION: HUB_PYPROJECT,
        IO_DISTRIBUTION: IO_PYPROJECT,
    }
    wheels = {
        CORE_DISTRIBUTION: args.core_wheel,
        SERVER_DISTRIBUTION: args.server_wheel,
        SERVE_DISTRIBUTION: args.serve_wheel,
        CLIENTS_DISTRIBUTION: args.clients_wheel,
        TARGETS_DISTRIBUTION: args.targets_wheel,
        HUB_DISTRIBUTION: args.hub_wheel,
        IO_DISTRIBUTION: args.io_wheel,
    }
    if all(wheels.values()):
        roots_by_distribution = {
            name: roots_from_wheel(wheel)
            for name, wheel in wheels.items()
            if wheel is not None
        }
        source = "built wheels"
    else:
        roots_by_distribution = {
            name: roots_from_pyproject(path)
            for name, path in pyprojects.items()
        }
        source = "pyproject discovery"
    checks = (
        (
            "disjoint import roots",
            check_disjoint_roots(roots_by_distribution),
        ),
        ("console-script ownership", check_script_ownership(pyprojects)),
        ("compatibility pin", check_matrix_pins(pyprojects)),
        (
            "coordinated minor bump",
            check_coordinated_minor_bump(_releases()),
        ),
    )
    failures = [(name, message) for name, message in checks if message]
    if failures:
        for name, message in failures:
            print(f"FAIL {name}: {message}", file=sys.stderr)
        return 1
    print(
        "packaging-guards: OK - "
        + ", ".join(name for name, _ in checks)
        + f" ({source})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

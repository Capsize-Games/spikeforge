"""The typed, JSON-serialisable result of a simulator-backed test deploy.

A :class:`TestDeployMatrix` holds one :class:`DeployCell` per registered
target. Each cell is deliberately honest: ``available`` reports whether the
target's SDK could be imported, ``backend`` whether an executable simulator
backend is wired at all, and ``status`` is ``ok`` only for a completed run.
An absent SDK or a declared-only simulator lands in ``unavailable`` with a
named ``reason`` rather than being reported as a failure or silently dropped,
so the matrix is a capability report and not a success/failure verdict.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from spikeforge_targets.backends.result import (
    STATUS_ERROR,
    STATUS_OK,
)


@dataclass(frozen=True)
class DeployCell:
    """One target's row of a simulator-backed test-deploy matrix."""

    target: str
    kind: str
    available: bool
    backend: bool
    status: str
    path: Optional[str] = None
    reason: Optional[str] = None
    parity: Optional[Mapping[str, Any]] = None
    parity_ok: Optional[bool] = None
    capability: Optional[Mapping[str, Any]] = None
    notes: Tuple[str, ...] = ()

    def ok(self) -> bool:
        """Return True when this cell completed a run."""
        return self.status == STATUS_OK

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the cell."""
        return {
            "target": self.target,
            "kind": self.kind,
            "available": self.available,
            "backend": self.backend,
            "status": self.status,
            "path": self.path,
            "reason": self.reason,
            "parity_ok": self.parity_ok,
            "parity": None if self.parity is None else dict(self.parity),
            "capability": (
                None if self.capability is None else dict(self.capability)
            ),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TestDeployMatrix:
    """One test-deploy cell per target plus the matrix-level verdict."""

    topology: str
    steps: int
    reference: str
    estimate: bool
    cells: Tuple[DeployCell, ...]
    notes: Tuple[str, ...] = ()

    def by_target(self) -> Dict[str, DeployCell]:
        """Return the cells keyed by target name."""
        return {cell.target: cell for cell in self.cells}

    def reference_cell(self) -> Optional[DeployCell]:
        """Return the reference cell, or ``None`` when it was not run."""
        return self.by_target().get(self.reference)

    def parity_targets(self) -> List[str]:
        """Return the targets that completed with a reference comparison."""
        return [cell.target for cell in self.cells if cell.parity is not None]

    def ok(self) -> bool:
        """Return True when the matrix ran honestly and without a real gap.

        The reference must have completed, and no *available* backend may have
        errored. An ``unavailable`` cell (an absent SDK or a declared-only
        simulator) is honest reporting, not a failure, so it never flips the
        verdict.
        """
        reference = self.reference_cell()
        if reference is None or not reference.ok():
            return False
        return all(
            cell.status != STATUS_ERROR for cell in self.cells
            if cell.available
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the matrix."""
        return {
            "topology": self.topology,
            "steps": int(self.steps),
            "reference": self.reference,
            "estimate": bool(self.estimate),
            "ok": self.ok(),
            "parity_targets": self.parity_targets(),
            "cells": [cell.to_dict() for cell in self.cells],
            "notes": list(self.notes),
        }

"""Execution-mode selection for the Spikeforge.

``ExecutionMode`` chooses between capturing full per-timestep trajectories
(``EDUCATIONAL``) and running a lean, faster path without trajectory logging
(``PRODUCTION``). It is a flag on one shared code path, never a fork. Phase 0
only defines the scaffold; there are no consumers yet.
"""

from enum import Enum


class ExecutionMode(Enum):
    """Selects trajectory capture versus fast execution."""

    EDUCATIONAL = "educational"
    PRODUCTION = "production"

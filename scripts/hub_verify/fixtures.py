"""The one deterministic spike probe shared by the drift and energy checks.

Both checks need some input to run the module over; using the same seeded,
low-density probe for both means one fixture to reason about instead of
two, and matches the fixture :mod:`spikeforge_targets.energy.accounting`
already uses for its own topology fixtures.
"""

import torch

from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.event_runtime.spike_view import synthetic_spikes

#: Matches ``spikeforge_targets/energy/accounting.py``'s own defaults.
STEPS = 8
BATCH = 2
SEED = 0


def probe_spikes(spec: TopologySpec) -> torch.Tensor:
    """Return a seeded, low-density spike probe shaped for ``spec``."""
    return synthetic_spikes(spec, STEPS, BATCH, SEED)

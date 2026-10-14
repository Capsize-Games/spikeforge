"""Firing fixtures shared by the activation-quantization tests."""

from typing import Any

import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)
#: Factor the seeded init weights are scaled by so the readout fires.
GAIN = 8.0

def _graph(spec: TopologySpec, gain: float = GAIN) -> Any:
    """Return ``spec`` exported with seeded weights scaled by ``gain``."""
    torch.manual_seed(0)
    module = build_module(spec)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.mul_(gain)
    return to_nir(spec, module)


def _spikes(steps: int = 8) -> torch.Tensor:
    """Return a fixed-seed binary spike train shaped for the conv presets."""
    torch.manual_seed(1)
    return (torch.rand(steps, 2, 1, 8, 8) < 0.3).float()

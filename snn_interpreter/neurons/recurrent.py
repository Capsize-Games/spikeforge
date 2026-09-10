"""Recurrent leaky neuron handler (``snn.RLeaky``)."""

from typing import Any, Dict, Mapping, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from snn_interpreter.neurons import contract
from snn_interpreter.neurons.contract import NeuronState


def _topology_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the snnTorch recurrent-topology selector arguments."""
    kwargs: Dict[str, Any] = {}
    if "linear_features" in params:
        kwargs["linear_features"] = int(params["linear_features"])
    if "conv2d_channels" in params:
        kwargs["conv2d_channels"] = int(params["conv2d_channels"])
    if "kernel_size" in params:
        kwargs["kernel_size"] = params["kernel_size"]
    if not kwargs:
        raise ValueError(
            "recurrent neuron requires linear_features, conv2d_channels, "
            "or kernel_size"
        )
    return kwargs


class RecurrentNeuron:
    """Build an ``snn.RLeaky`` and describe it in NIR parameter terms."""

    kind: str = "recurrent"

    def build(self, params: Mapping[str, Any]) -> nn.Module:
        """Build an ``snn.RLeaky`` from ``params``."""
        return snn.RLeaky(
            beta=contract.get_float(params, "beta", contract.DEFAULT_BETA),
            V=contract.get_float(params, "V", 1.0),
            all_to_all=bool(params.get("all_to_all", True)),
            threshold=contract.get_float(
                params, "threshold", contract.DEFAULT_THRESHOLD
            ),
            reset_mechanism=contract.get_reset(params),
            **_topology_kwargs(params),
        )

    def nir_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Return NIR parameters including the one-step feedback delay."""
        base = contract.base_params(self.kind, params)
        base["feedback_delay"] = True
        return base

    def initial_state(self, device: torch.device) -> NeuronState:
        """Return the zero-length ``(spk, mem)`` state on ``device``."""
        return (torch.empty(0, device=device), torch.empty(0, device=device))

    def step(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, NeuronState]:
        """Run one step, returning ``(spikes, (spk_trace, mem))``."""
        if state is None:
            state = self.initial_state(x.device)
        spk, mem = module(x, state[0], state[1])
        return spk, (spk, mem)

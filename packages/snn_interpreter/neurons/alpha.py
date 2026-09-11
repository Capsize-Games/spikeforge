"""Alpha-function neuron handler (``snn.Alpha``).

``snn.Alpha`` is simulation/introspection-only for now. The installed
``nir`` exposes no alpha-function primitive: ``snntorch`` keeps three states
(``syn_exc``, ``syn_inh``, ``mem``) and derives the membrane as
``tau_alpha * (syn_exc + syn_inh)``, whereas ``nir.CubaLIF`` carries a single
synaptic current. Exporting an ``alpha`` stage therefore raises the typed
:class:`~snn_interpreter.nir_bridge.errors.UnsupportedStageError` rather than
inventing a lossy mapping.
"""

from typing import Any, Dict, Mapping, Optional, Tuple

import snntorch as snn
import torch
import torch.nn as nn

from snn_interpreter.neurons import contract
from snn_interpreter.neurons.contract import NeuronState
from snn_interpreter.neurons.spike_grad import spike_grad_from

#: ``snn.Alpha`` requires ``alpha > beta``; these defaults satisfy that.
DEFAULT_ALPHA = 0.9
DEFAULT_BETA = 0.8


class AlphaNeuron:
    """Build an ``snn.Alpha`` and describe it in NIR parameter terms."""

    kind: str = "alpha"

    def build(self, params: Mapping[str, Any]) -> nn.Module:
        """Build an ``snn.Alpha`` from ``params``."""
        return snn.Alpha(
            alpha=contract.get_float(params, "alpha", DEFAULT_ALPHA),
            beta=contract.get_float(params, "beta", DEFAULT_BETA),
            threshold=contract.get_float(
                params, "threshold", contract.DEFAULT_THRESHOLD
            ),
            reset_mechanism=contract.get_reset(params),
            spike_grad=spike_grad_from(params),
        )

    def nir_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Return NIR parameters with both synaptic and membrane constants."""
        resolved = dict(params)
        resolved.setdefault("beta", DEFAULT_BETA)
        alpha = contract.get_float(params, "alpha", DEFAULT_ALPHA)
        dt = contract.get_float(params, "dt", contract.DEFAULT_DT)
        base = contract.base_params(self.kind, resolved)
        base["tau_syn"] = contract.tau_from_decay(alpha, dt)
        return base

    def initial_state(self, device: torch.device) -> NeuronState:
        """Return the zero-length ``(syn_exc, syn_inh, mem)`` state."""
        return (
            torch.empty(0, device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device),
        )

    def step(
        self,
        module: nn.Module,
        x: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, NeuronState]:
        """Run one step, returning ``(spikes, (syn_exc, syn_inh, mem))``."""
        if state is None:
            state = self.initial_state(x.device)
        spk, syn_exc, syn_inh, mem = module(
            x, state[0], state[1], state[2]
        )
        return spk, (syn_exc, syn_inh, mem)

"""The serializable carried state of a stateful inference session."""

from typing import Any, Dict, Mapping, Optional, Union

import torch

from spikeforge.serving import tensor_codec


class StateTree:
    """One tensor-bearing state mapping that can be reset, saved, and moved.

    The mapping is the ``state`` returned by
    :meth:`~spikeforge.topology.stage_module.StageModule.step`: one entry per
    neuron stage (an ordered tuple of tensors) plus the reserved previous-step
    and input-current entries a delayed (recurrent) edge reads. :meth:`to_dict`
    and :meth:`from_dict` are exact inverses, so a session can be paused and
    resumed across processes.
    """

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        """Wrap ``state``, or start empty when it is absent."""
        self._state: Dict[str, Any] = dict(state) if state else {}

    def reset(self) -> None:
        """Drop every carried tensor so the next step starts clean."""
        self._state = {}

    def raw(self) -> Mapping[str, Any]:
        """Return the underlying mapping the simulator consumes."""
        return self._state

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe mapping with dtype and shape preserved."""
        return tensor_codec.encode(self._state)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StateTree":
        """Rebuild a tree from :meth:`to_dict` output."""
        decoded = tensor_codec.decode(dict(data))
        return cls(decoded if isinstance(decoded, dict) else {})

    def to(self, device: Union[str, torch.device]) -> "StateTree":
        """Return a copy with every tensor placed on ``device``."""
        return StateTree(tensor_codec.move(self._state, torch.device(device)))

"""Topology resolution shared with the training engine."""

from typing import Any, Dict, Optional

import torch

from spikeforge.simulator import input_shape
from spikeforge.topology.spec import TopologySpec


class TopologyMixin:
    """Resolve a topology name/params and expose its accessors."""

    _topology: str
    _topology_params: Dict[str, Any]
    _architecture: Dict[str, Any]
    _spec: TopologySpec
    _hidden: int
    _beta: float
    _num_classes: int

    def _adopt_topology(self, checkpoint: Optional[str]) -> None:
        """Override the topology from a checkpoint's stored meta, if any."""
        if not checkpoint:
            return
        meta = self._stored_meta(checkpoint)
        self._topology = str(meta.get("topology", self._topology))
        stored = meta.get("topology_params")
        if stored:
            self._topology_params = dict(stored)

    def _topology_arguments(self) -> Dict[str, Any]:
        """Return engine-wide args overlaid by the caller's topology params."""
        params: Dict[str, Any] = {
            "hidden": self._hidden,
            "beta": self._beta,
            "num_classes": self._num_classes,
        }
        params.update(self._topology_params)
        return params

    def _apply_effective(self) -> None:
        """Mirror resolved hidden/beta back onto the engine accessors."""
        if "hidden" in self._architecture:
            self._hidden = int(self._architecture["hidden"])
        if "beta" in self._architecture:
            self._beta = float(self._architecture["beta"])

    def _dummy_spikes(self) -> torch.Tensor:
        """Return a correctly-shaped zero spike train for CUDA warmup."""
        frames = torch.zeros(self.num_steps, 1, self._input_features())
        return self._to_input_shape(frames)

    def _input_features(self) -> int:
        """Return the flat feature count the input stage consumes."""
        value = self._architecture.get("input_size")
        shape = input_shape.spatial_shape(self._spec, value)
        if shape is not None:
            return shape[0] * shape[1]
        return int(value) if value is not None else 28 * 28

    @property
    def topology(self) -> str:
        """Return the active topology registry name."""
        return self._topology

    @property
    def spec(self) -> TopologySpec:
        """Return the topology spec the module was built from."""
        return self._spec

    @property
    def hidden(self) -> int:
        """Return the hidden-layer width (ignored by spatial topologies)."""
        return self._hidden

    @property
    def beta(self) -> float:
        """Return the LIF membrane decay rate."""
        return self._beta

"""Run settings, session configuration, and seeding for the engine."""

from typing import Any, Dict, Optional

from spikeforge.data.datasets import dataset_info
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.tracking.determinism import enable_deterministic
from spikeforge.tracking.seed import set_seed
from spikeforge.training.scaleup_mixin import scaleup_options


class SessionMixin:
    """Store the run's plain settings, session, and seeding policy."""

    _dataset: str
    _download: bool
    _num_classes: int
    _hidden: int
    _beta: float
    _lr: float
    _weight_decay: float
    _epochs: int
    _num_steps: int
    _subset: int
    _batch_size: int
    _mode: ExecutionMode
    _seed: Optional[int]
    _tracking: Optional[str]
    _deterministic: bool
    _determinism_report: Optional[Dict[str, Any]]
    _encode: Any
    _topology: str
    _topology_params: Dict[str, Any]
    _scaleups: Dict[str, Any]

    def _store_settings(
        self, dataset: str, hidden: int, beta: float, lr: float,
        weight_decay: float, epochs: int, num_steps: int, subset: int,
        batch_size: int, download: bool = True,
    ) -> None:
        """Store the plain training settings on the instance."""
        self._dataset = dataset
        self._download = bool(download)
        self._num_classes, _ = dataset_info(dataset)
        self._hidden = hidden
        self._beta = beta
        self._lr = lr
        self._weight_decay = weight_decay
        self._epochs = epochs
        self._num_steps = num_steps
        self._subset = subset
        self._batch_size = batch_size

    def _configure_session(
        self, mode: str, seed: Optional[int], tracking: Optional[str],
        deterministic: bool, encode: Any, topology: str,
        topology_params: Optional[Dict[str, Any]], amp: bool,
        grad_checkpoint: bool, bptt_steps: Optional[int], multi_gpu: bool,
    ) -> None:
        """Store the mode/seed/tracking/topology/scaleup settings."""
        self._mode = ExecutionMode(mode)
        self._seed = None if seed is None else int(seed)
        self._tracking = tracking
        self._deterministic = bool(deterministic)
        self._determinism_report = None
        self._encode = encode
        self._topology = topology
        self._topology_params = dict(topology_params or {})
        self._scaleups = scaleup_options(
            amp, grad_checkpoint, bptt_steps, multi_gpu
        )

    def _apply_seed(self) -> None:
        """Seed torch and, if requested, enable deterministic mode."""
        if self._seed is not None:
            set_seed(self._seed)
        if self._deterministic:
            report = enable_deterministic(self._seed)
            self._determinism_report = report.to_dict()

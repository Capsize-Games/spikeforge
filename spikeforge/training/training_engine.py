"""Train the spiking network, yielding live metrics for streaming."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from spikeforge.data.data_loader import build_loader
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.network import inference
from spikeforge.runtime import device as device_mod
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology import registry
from spikeforge.topology.stage_module import StageModule
from spikeforge.training.checkpoint_mixin import CheckpointMixin
from spikeforge.training.encoding_mixin import EncodingMixin
from spikeforge.training.eval_mixin import EvalMixin
from spikeforge.training.scaleup_mixin import ScaleUpMixin
from spikeforge.training.session_mixin import SessionMixin
from spikeforge.training.topology_mixin import TopologyMixin

_Batch = Tuple[torch.Tensor, torch.Tensor]


class TrainingEngine(
    device_mod.DeviceMixin,
    CheckpointMixin,
    TopologyMixin,
    EncodingMixin,
    ScaleUpMixin,
    EvalMixin,
    SessionMixin,
):
    """Run a cancellable training loop that emits metric dicts."""

    _encoder: Optional[SpikeEncoder]
    _encode_spec: EncodeSpec
    _explicit_mode: Optional[str]
    _input_mode: str
    _mode: ExecutionMode
    _net: StageModule
    _optimizer: torch.optim.Adam
    _weight_decay: float
    _seed: Optional[int]
    _test_batches: Optional[List[_Batch]]
    _scaleups: Dict[str, Any]
    _tracking: Optional[str]
    _deterministic: bool
    _determinism_report: Optional[Dict[str, Any]]
    _download: bool

    def __init__(self, dataset: str = "mnist", hidden: int = 128,
        beta: float = 0.5, lr: float = 1e-2, weight_decay: float = 0.0,
        epochs: int = 1, num_steps: int = 10, subset: int = 10,
        batch_size: int = 64, checkpoint: Optional[str] = None,
        encode: Any = None, input_mode: Optional[str] = None,
        device: Optional[str] = None, topology: str = "fc_legacy",
        topology_params: Optional[Dict[str, Any]] = None,
        mode: str = "production", seed: Optional[int] = None,
        amp: bool = False, grad_checkpoint: bool = False,
        bptt_steps: Optional[int] = None, multi_gpu: bool = False,
        tracking: Optional[str] = None, deterministic: bool = False,
        download: bool = True,
    ) -> None:
        """Resolve inputs, build the topology, and optionally restore it."""
        self._store_settings(dataset, hidden, beta, lr, weight_decay,
                             epochs, num_steps, subset, batch_size,
                             download)
        self._configure_session(
            mode, seed, tracking, deterministic, encode, topology,
            topology_params, amp, grad_checkpoint, bptt_steps, multi_gpu)
        self._build(encode, input_mode, device, lr, checkpoint)

    def _make_topology(self, params: Dict[str, Any]) -> None:
        """Build the spec/module from ``params`` and warm it up."""
        self._architecture = registry.resolved_params(self._topology, params)
        self._apply_effective()
        self._spec, self._net = registry.build_topology(
            self._topology, params
        )
        self._net.to(self._device)
        device_mod.warmup(self._net, self._dummy_spikes())

    def _build(
        self, encode: Any, input_mode: Optional[str], device: Optional[str],
        lr: float, checkpoint: Optional[str] = None,
    ) -> None:
        """Resolve the input, then create the topology and restore a ckpt."""
        self._setup_input(encode, input_mode, device)
        self._apply_seed()
        self._adopt_topology(checkpoint)
        self._make_topology(self._topology_arguments())
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=lr, weight_decay=self._weight_decay
        )
        self._test_batches = None
        self._configure_scaleups(**self._scaleups)
        if checkpoint:
            self._restore(checkpoint)

    # --- forward ---------------------------------------------------------

    def _run_trajectory(self, spikes: torch.Tensor) -> Trajectory:
        """Run the shared simulator with the engine's gradient policy."""
        return run(
            self._net,
            spikes,
            mode=self._mode,
            grad_checkpoint=self._grad_checkpoint,
            bptt_steps=self._bptt_steps,
        )

    # --- training --------------------------------------------------------

    def _epoch_batches(self, train: bool = True) -> Any:
        """Return the (inputs, targets) batches for one epoch.

        Image modality reads through the shared loader; the event engine
        overrides this to bridge an event stream into the same contract.
        """
        return build_loader(
            self._dataset, self._subset, self._batch_size, train=train,
            download=self._download,
        )

    def train(
        self, should_stop: Optional[Callable[[], bool]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield a metrics dict after each batch until done or stopped."""
        loader = self._epoch_batches(train=True)
        total = len(loader) * self._epochs
        step = 0
        for epoch in range(self._epochs):
            for inputs, targets in loader:
                if should_stop and should_stop():
                    return
                metrics = self._train_batch(inputs, targets)
                step += 1
                metrics.update({"epoch": epoch, "step": step,
                                "total": total})
                metrics["test_accuracy"] = self._maybe_evaluate(step, total)
                yield metrics

    def predict_sample(self) -> Dict[str, List[int]]:
        """Return digits/labels for one held-out batch."""
        loader = build_loader(
            self._dataset, 1, 32, train=False, download=self._download
        )
        inputs, targets = next(iter(loader))
        preds = self.predict(inputs)
        return {
            "digits": [int(p) for p in preds[:10]],
            "labels": [int(t) for t in targets[:10]],
        }

    def infer(
        self, spikes: torch.Tensor, true_label: Optional[int] = None
    ) -> Dict[str, Any]:
        """Score the displayed sample's spikes for the active model."""
        shaped = self._to_input_shape(spikes.to(self._device))
        return inference.infer_spikes(
            self._net,
            shaped,
            self._num_classes,
            true_label,
            self.input_mode,
            mode=self._mode,
        )

    # --- accessors -------------------------------------------------------

    @property
    def net(self) -> StageModule:
        """Return the underlying topology module."""
        return self._net

    @property
    def num_steps(self) -> int:
        """Return effective time steps (encoder wins in spike mode)."""
        if self._encoder is not None and self._input_mode != "raw":
            return self._encoder.num_steps
        return self._num_steps

    @property
    def num_classes(self) -> int:
        """Return the number of output classes."""
        return self._num_classes

    @property
    def seed(self) -> Optional[int]:
        """Return the seed used to initialise this run, if any."""
        return self._seed

    @property
    def input_mode(self) -> str:
        """Return the effective input mode."""
        return self._input_mode

    @property
    def mode(self) -> str:
        """Return the active execution mode name."""
        return self._mode.value

    @property
    def execution_mode(self) -> ExecutionMode:
        """Return the active :class:`ExecutionMode` member."""
        return self._mode

    @property
    def dataset(self) -> str:
        """Return the dataset registry key."""
        return self._dataset

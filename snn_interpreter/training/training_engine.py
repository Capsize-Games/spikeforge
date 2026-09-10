"""Train the spiking network, yielding live metrics for streaming."""

from itertools import islice
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch.nn.functional import cross_entropy

from snn_interpreter.data.data_loader import build_loader
from snn_interpreter.data.datasets import dataset_info
from snn_interpreter.encoding.spike_encoder import SpikeEncoder
from snn_interpreter.network import inference
from snn_interpreter.runtime import device as device_mod
from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import registry
from snn_interpreter.topology.stage_module import StageModule
from snn_interpreter.training.checkpoint_mixin import CheckpointMixin
from snn_interpreter.training.encoding_mixin import EncodingMixin
from snn_interpreter.training.topology_mixin import TopologyMixin

EVAL_EVERY = 5  # evaluate the held-out set every N steps
EVAL_BATCHES = 4  # number of test batches to score

_Batch = Tuple[torch.Tensor, torch.Tensor]


class TrainingEngine(
    device_mod.DeviceMixin,
    CheckpointMixin,
    TopologyMixin,
    EncodingMixin,
):
    """Run a cancellable training loop that emits metric dicts."""

    _encoder: Optional[SpikeEncoder]
    _explicit_mode: Optional[str]
    _input_mode: str
    _mode: ExecutionMode
    _net: StageModule
    _optimizer: torch.optim.Adam
    _test_batches: Optional[List[_Batch]]

    def __init__(
        self, dataset: str = "mnist", hidden: int = 128, beta: float = 0.5,
        lr: float = 1e-2, epochs: int = 1, num_steps: int = 10,
        subset: int = 10, batch_size: int = 64,
        checkpoint: Optional[str] = None, encode: Any = None,
        input_mode: Optional[str] = None, device: Optional[str] = None,
        topology: str = "fc_legacy",
        topology_params: Optional[Dict[str, Any]] = None,
        mode: str = "production",
    ) -> None:
        """Resolve inputs, build the topology, and optionally restore it."""
        self._store_settings(
            dataset, hidden, beta, lr, epochs, num_steps, subset, batch_size
        )
        self._mode = ExecutionMode(mode)
        self._encode = encode
        self._topology = topology
        self._topology_params = dict(topology_params or {})
        self._setup_input(encode, input_mode, device)
        self._build(lr, checkpoint)

    def _store_settings(
        self, dataset: str, hidden: int, beta: float, lr: float,
        epochs: int, num_steps: int, subset: int, batch_size: int,
    ) -> None:
        """Store the plain training settings on the instance."""
        self._dataset = dataset
        self._num_classes, _ = dataset_info(dataset)
        self._hidden = hidden
        self._beta = beta
        self._lr = lr
        self._epochs = epochs
        self._num_steps = num_steps
        self._subset = subset
        self._batch_size = batch_size

    def _setup_input(
        self, encode: Any, input_mode: Optional[str], device: Optional[str]
    ) -> None:
        """Resolve the encoder, effective input mode, and compute device."""
        self._encoder = (
            SpikeEncoder.from_encode_config(encode)
            if encode is not None
            else None
        )
        self._explicit_mode = input_mode
        coding = encode.coding if encode is not None else "raw"
        self._input_mode = input_mode or coding
        self._set_device(device or "auto", encoder=self._encoder,
                         hidden=self._hidden, num_classes=self._num_classes,
                         num_steps=self._num_steps,
                         batch_size=self._batch_size)

    def _build(self, lr: float, checkpoint: Optional[str] = None) -> None:
        """Create the configured topology/optimiser and restore a ckpt."""
        self._adopt_topology(checkpoint)
        params = self._topology_arguments()
        self._architecture = registry.resolved_params(self._topology, params)
        self._apply_effective()
        self._spec, self._net = registry.build_topology(
            self._topology, params
        )
        self._net.to(self._device)
        device_mod.warmup(self._net, self._dummy_spikes())
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._test_batches = None
        if checkpoint:
            self._restore(checkpoint)

    # --- evaluation ------------------------------------------------------

    def _load_test_batches(self) -> List[_Batch]:
        """Cache a few held-out batches for scoring."""
        if self._test_batches is None:
            loader = build_loader(self._dataset, 1, 1000, train=False)
            self._test_batches = list(islice(loader, EVAL_BATCHES))
        return self._test_batches

    def evaluate(self) -> float:
        """Return held-out accuracy over the cached test batches."""
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self._load_test_batches():
                correct += int(
                    (self.predict(inputs) == targets.to(self._device)).sum()
                )
                total += len(targets)
        return 100.0 * correct / max(total, 1)

    def _maybe_evaluate(self, step: int, total: int) -> Optional[float]:
        """Evaluate periodically and on the final step."""
        due = step == 1 or step == total or step % EVAL_EVERY == 0
        return self.evaluate() if due else None

    # --- training --------------------------------------------------------

    def train(
        self, should_stop: Optional[Callable[[], bool]] = None
    ) -> Iterator[Dict[str, Any]]:
        """Yield a metrics dict after each batch until done or stopped."""
        loader = build_loader(
            self._dataset, self._subset, self._batch_size, train=True
        )
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

    def _train_batch(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Run one optimisation step and return loss/accuracy."""
        outputs = run(
            self._net, self._encode_batch(inputs), mode=self._mode
        ).logits
        loss = cross_entropy(outputs, targets.to(self._device))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets.to(self._device)).float().mean().item()
        return {"loss": float(loss.item()), "train_accuracy": float(accuracy)}

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predicted digits for a batch of images."""
        with torch.no_grad():
            trajectory = run(
                self._net, self._encode_batch(inputs), mode=self._mode
            )
        return trajectory.logits.argmax(dim=1)

    def predict_sample(self) -> Dict[str, List[int]]:
        """Return digits/labels for one held-out batch."""
        loader = build_loader(self._dataset, 1, 32, train=False)
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

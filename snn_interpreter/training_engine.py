"""Train the spiking network, yielding live metrics for streaming."""

import torch
import torch.nn.functional as F

from snn_interpreter import inference, model_store
from snn_interpreter.data_loader import build_loader
from snn_interpreter.datasets import dataset_info
from snn_interpreter.spike_encoder import SpikeEncoder
from snn_interpreter.spiking_net import SpikingNet

EVAL_EVERY = 5       # evaluate the held-out set every N steps
EVAL_BATCHES = 4     # number of test batches to score


class TrainingEngine:
    """Run a cancellable training loop that emits metric dicts."""

    def __init__(self, dataset="mnist", hidden=128, beta=0.5, lr=1e-2,
                 epochs=1, num_steps=10, subset=10, batch_size=64,
                 checkpoint=None, encode=None, input_mode=None):
        self._dataset = dataset
        self._num_classes, _ = dataset_info(dataset)
        self._hidden = hidden
        self._beta = beta
        self._lr = lr
        self._epochs = epochs
        self._num_steps = num_steps
        self._subset = subset
        self._batch_size = batch_size
        self._checkpoint = checkpoint
        self._encode = encode
        self._setup_input(encode, input_mode)
        self._build(lr)
        if checkpoint:
            self._restore(checkpoint)

    def _setup_input(self, encode, input_mode):
        """Resolve the encoder and the effective input mode."""
        self._encoder = (SpikeEncoder.from_encode_config(encode)
                         if encode is not None else None)
        self._explicit_mode = input_mode
        coding = encode.coding if encode is not None else "raw"
        self._input_mode = input_mode or coding

    def _build(self, lr):
        """Create the network, optimiser, and lazy test-batch cache."""
        self._net = SpikingNet(hidden=self._hidden, beta=self._beta,
                               num_classes=self._num_classes)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._test_batches = None

    # --- checkpointing ---------------------------------------------------

    def _restore(self, checkpoint):
        """Load weights and, unless overridden, the input mode."""
        ckpt = model_store.load(checkpoint)
        self._net.load_state_dict(ckpt["state_dict"])
        if self._explicit_mode is None:
            meta = ckpt.get("meta", {})
            self._input_mode = meta.get("input_mode", "raw")

    def save(self, name):
        """Persist the current model and its metadata."""
        meta = {
            "dataset": self._dataset,
            "hidden": self._hidden,
            "beta": self._beta,
            "lr": self._lr,
            "num_steps": self.num_steps,
            "num_classes": self._num_classes,
            "input_mode": self._input_mode,
            "coding": self._input_mode,
            "encode": self._encode.model_dump() if self._encode else None,
        }
        return model_store.save(name, self._net, meta)

    # --- encoding --------------------------------------------------------

    def _encode_batch(self, inputs):
        """Return [T,B,784] spikes for the configured input mode."""
        if self._encoder is None or self._input_mode == "raw":
            return self._repeat_pixels(inputs)
        if self._input_mode == "random":
            raise ValueError("random coding carries no label signal")
        return self._encoder.encode(inputs)

    def _repeat_pixels(self, inputs):
        """Legacy raw path: repeat normalised pixels across steps."""
        flat = inputs.view(inputs.size(0), -1)
        return flat.unsqueeze(0).repeat(self._num_steps, 1, 1)

    # --- evaluation ------------------------------------------------------

    def _load_test_batches(self):
        """Cache a few held-out batches for scoring."""
        if self._test_batches is None:
            loader = build_loader(self._dataset, 1, 1000, train=False)
            self._test_batches = [
                batch for _, batch in zip(range(EVAL_BATCHES), loader)
            ]
        return self._test_batches

    def evaluate(self):
        """Return held-out accuracy over the cached test batches."""
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self._load_test_batches():
                correct += int((self.predict(inputs) == targets).sum())
                total += len(targets)
        return 100.0 * correct / max(total, 1)

    def _maybe_evaluate(self, step, total):
        """Evaluate periodically and on the final step."""
        due = step == 1 or step == total or step % EVAL_EVERY == 0
        return self.evaluate() if due else None

    # --- training --------------------------------------------------------

    def train(self, should_stop=None):
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

    def _train_batch(self, inputs, targets):
        """Run one optimisation step and return loss/accuracy."""
        outputs = self._net.forward_spikes(self._encode_batch(inputs))
        loss = F.cross_entropy(outputs, targets)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
        return {"loss": float(loss.item()),
                "train_accuracy": float(accuracy)}

    def predict(self, inputs):
        """Return predicted digits for a batch of images."""
        with torch.no_grad():
            outputs = self._net.forward_spikes(self._encode_batch(inputs))
        return outputs.argmax(dim=1)

    def predict_sample(self):
        """Return digits/labels for one held-out batch."""
        inputs, targets = next(iter(build_loader(self._dataset, 1, 32,
                                                 train=False)))
        preds = self.predict(inputs)
        return {"digits": [int(p) for p in preds[:10]],
                "labels": [int(t) for t in targets[:10]]}

    def infer(self, spikes, true_label=None):
        """Score the displayed sample's spikes for the active model."""
        return inference.infer_spikes(self._net, spikes, self._num_classes,
                                      true_label, self.input_mode)

    # --- accessors -------------------------------------------------------

    @property
    def net(self):
        return self._net

    @property
    def num_steps(self):
        """Return effective time steps (encoder wins in spike mode)."""
        if self._encoder is not None and self._input_mode != "raw":
            return self._encoder.num_steps
        return self._num_steps

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def input_mode(self):
        return self._input_mode

    @property
    def coding(self):
        return self._input_mode

    @property
    def dataset(self):
        return self._dataset

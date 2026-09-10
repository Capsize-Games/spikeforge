"""Train the spiking network, yielding live metrics for streaming."""

import torch
import torch.nn.functional as F
from snntorch import utils
from torch.utils.data import DataLoader

from snn_interpreter import model_store
from snn_interpreter.datasets import build_dataset, dataset_info
from snn_interpreter.spiking_net import SpikingNet

EVAL_EVERY = 5       # evaluate the held-out set every N steps
EVAL_BATCHES = 4     # number of test batches to score


def build_loader(dataset="mnist", subset=10, batch_size=64, train=True):
    """Create a normalised loader reduced by the subset factor."""
    data = build_dataset(dataset, train=train)
    if subset > 1:
        data = utils.data_subset(data, subset)
    return DataLoader(data, batch_size=batch_size, shuffle=train)


class TrainingEngine:
    """Run a cancellable training loop that emits metric dicts."""

    def __init__(self, dataset="mnist", hidden=128, beta=0.5, lr=1e-2,
                 epochs=1, num_steps=10, subset=10, batch_size=64,
                 checkpoint=None):
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
        self._build(lr)
        if checkpoint:
            self._restore(checkpoint)

    def _build(self, lr):
        """Create the network, optimiser, and lazy test-batch cache."""
        self._net = SpikingNet(hidden=self._hidden, beta=self._beta,
                               num_classes=self._num_classes)
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=lr
        )
        self._test_batches = None

    # --- checkpointing ---------------------------------------------------

    def _restore(self, checkpoint):
        """Load weights from a saved checkpoint to continue training."""
        ckpt = model_store.load(checkpoint)
        self._net.load_state_dict(ckpt["state_dict"])

    def save(self, name):
        """Persist the current model and its metadata."""
        meta = {
            "dataset": self._dataset,
            "hidden": self._hidden,
            "beta": self._beta,
            "lr": self._lr,
            "num_steps": self._num_steps,
            "num_classes": self._num_classes,
        }
        return model_store.save(name, self._net, meta)

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
        """Evaluate on held-out data periodically and on the final step."""
        if step == 1 or step == total or step % EVAL_EVERY == 0:
            return self.evaluate()
        return None

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
                metrics["test_accuracy"] = self._maybe_evaluate(
                    step, total
                )
                yield metrics

    def _train_batch(self, inputs, targets):
        """Run one optimisation step and return loss/accuracy."""
        outputs = self._net(inputs, self._num_steps)
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
            outputs = self._net(inputs, self._num_steps)
        return outputs.argmax(dim=1)

    def predict_sample(self):
        """Return digits/labels for one held-out batch."""
        inputs, targets = next(iter(
            build_loader(self._dataset, 1, 32, train=False)
        ))
        preds = self.predict(inputs)
        return {
            "digits": [int(p) for p in preds[:10]],
            "labels": [int(t) for t in targets[:10]],
        }

    @property
    def net(self):
        return self._net

    @property
    def num_steps(self):
        return self._num_steps

    @property
    def dataset(self):
        return self._dataset

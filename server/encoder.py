"""Factory producing encoded spike payloads from a client config."""

import torch

from snn_interpreter.delta_trainer import DeltaTrainer
from snn_interpreter.latency_trainer import LatencyTrainer
from snn_interpreter.random_spikegen import RandomSpikeGenerator
from snn_interpreter.trainer import SSNTrainer

from server.schemas import EncodeConfig


def to_list(tensor):
    """Convert a tensor to a nested python list of floats."""
    return tensor.detach().cpu().float().tolist()


class EncoderEngine:
    """Build trainers from config and expose JSON-serialisable data."""

    def __init__(self, config: EncodeConfig):
        self._config = config
        self._trainer = _build_trainer(config)

    # --- payload builders -------------------------------------------------

    def sample_image(self):
        """Return the raw input image (single channel grid) as nested list."""
        if self._config.coding in ("delta", "random"):
            return None
        tensor = self._trainer.input_data[self._config.sample_index][0]
        return to_list(tensor)

    def reconstruction(self):
        """Return gain=1 and low-gain averaged reconstructions."""
        if self._config.coding != "rate":
            return None
        trainer = self._trainer
        sample = trainer.spike_data[:, 0, 0]
        low = trainer.spike_data_low_gain[:, 0, 0]
        return {
            "gain1": to_list(sample.mean(dim=0)),
            "low": to_list(low.mean(dim=0)),
            "size": list(trainer.data_size),
        }

    def spike_frame(self, step: int):
        """Return one 2-D frame of spike activity for sample 0."""
        if self._config.coding == "delta":
            return to_list(self._trainer.spike_data)
        if self._config.coding == "random":
            return to_list(self._trainer.spike_rand[step])
        return to_list(self._raw_spike_data()[step, 0, 0])

    def spike_tensor(self):
        """Return the full spike volume for a trainer."""
        if self._config.coding == "delta":
            return self._trainer.spike_data
        if self._config.coding == "random":
            return self._trainer.spike_rand
        return self._raw_spike_data()

    def raster(self, max_neurons=784):
        """Return (time, neuron) spike coordinate pairs for the raster."""
        spikes = self._spike_sample_matrix()
        time_idx, neuron_idx = torch.where(spikes > 0)
        return {
            "time": to_list(time_idx),
            "neurons": to_list(neuron_idx),
            "num_steps": int(spikes.size(0)),
            "num_neurons": int(spikes.size(1)),
        }

    def num_steps(self):
        """Return the number of time steps in the encoded data."""
        if self._config.coding == "delta":
            return int(self._trainer.data.numel())
        if self._config.coding == "random":
            return int(self._trainer.num_steps)
        return int(self._trainer.num_steps)

    def target_label(self):
        """Return the first target label when available."""
        trainer = getattr(self._trainer, "latency_targets", None)
        if trainer is not None and self._config.coding == "latency":
            return int(self._trainer.latency_targets[0].item())
        targets = getattr(self._trainer, "_spike_targets", None)
        if targets is not None:
            return int(targets[0].item())
        return None

    def _raw_spike_data(self):
        """Select which latency/rate volume to expose based on flags."""
        coding = self._config.coding
        trainer = self._trainer
        if coding == "latency":
            data = trainer.latency_data
            key = "base"
            if self._config.linear:
                key = "linear" if not self._config.normalize else "normalized"
            if self._config.clip:
                key = "clip"
            return data[key]
        if coding == "random":
            return trainer.spike_rand
        return trainer.spike_data

    def _spike_sample_matrix(self):
        """Reduce the sample-0 spikes to a (time x neurons) 0/1 matrix."""
        spikes = self._raw_spike_data()
        if self._config.coding == "delta":
            return spikes.unsqueeze(1)  # 1-D -> (time, 1 neuron)
        if spikes.dim() == 5:  # [T, B, C, H, W] -> sample 0
            spikes = spikes[:, 0, 0]
        elif spikes.dim() == 4:  # [T, C, H, W] -> drop channel
            spikes = spikes[:, 0]
        return spikes.reshape(spikes.size(0), -1)


def _build_trainer(cfg: EncodeConfig):
    """Instantiate the trainer matching a coding type."""
    return {
        "rate": lambda: _rate_trainer(cfg),
        "latency": lambda: _latency_trainer(cfg),
        "delta": lambda: _delta_trainer(cfg),
        "random": lambda: _random_trainer(cfg),
    }[cfg.coding]()


def _rate_trainer(cfg: EncodeConfig):
    return SSNTrainer(
        batch_size=cfg.batch_size,
        subset=cfg.subset,
        vectorization_num_steps=cfg.num_steps,
        vector_value=cfg.vector_value,
        reconstruction_gain=cfg.gain,
        animation_interval=cfg.interval_ms,
    )


def _latency_trainer(cfg: EncodeConfig):
    return LatencyTrainer(
        batch_size=cfg.batch_size,
        subset=cfg.subset,
        vectorization_num_steps=cfg.num_steps,
        animation_interval=cfg.interval_ms,
        tau=cfg.tau,
        threshold=cfg.threshold,
    )


def _delta_trainer(cfg: EncodeConfig):
    return DeltaTrainer(
        threshold=cfg.delta_threshold, off_spike=cfg.off_spike
    )


def _random_trainer(cfg: EncodeConfig):
    return RandomSpikeGenerator(
        num_steps=cfg.num_steps, scale=cfg.random_scale
    )

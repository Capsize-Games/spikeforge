"""CPU/GPU device selection with a safe fallback to CPU."""

import torch


def cuda_available():
    """Return True when a usable CUDA device is present."""
    return torch.cuda.is_available()


def gpu_name():
    """Return the current CUDA device name, or None without a GPU."""
    return torch.cuda.get_device_name(0) if cuda_available() else None


def device_options():
    """Return selectable device names plus GPU metadata for the UI."""
    available = ["gpu", "cpu"] if cuda_available() else ["cpu"]
    return {"available": available, "gpu_name": gpu_name()}


def resolve(requested="gpu"):
    """Map 'gpu'/'cpu' to a torch.device, falling back to CPU if needed."""
    want_gpu = str(requested).lower() in ("gpu", "cuda")
    return torch.device("cuda" if want_gpu and cuda_available() else "cpu")


def warmup(net, steps, features=28 * 28):
    """Trigger CUDA context init so the first training step isn't slow."""
    if not cuda_available():
        return
    dummy = torch.zeros(1, steps, features, device="cuda")
    with torch.no_grad():
        net.forward_spikes(dummy)
    torch.cuda.synchronize()


class DeviceMixin:
    """Give an object a resolved torch device and a label property."""

    def _set_device(self, requested="gpu"):
        self._device = resolve(requested)

    @property
    def device(self):
        """Return the resolved device label ('cuda' or 'cpu')."""
        return self._device.type

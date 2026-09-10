"""CPU/GPU device selection with a safe fallback to CPU."""

import time

import torch

# Bound the Auto benchmark so engine construction stays fast.
_BENCH_STEPS = 10
_BENCH_BATCH = 1024
_BENCH_REPS = 2

_AUTO_CACHE = {}


def prime():
    """Initialise the CUDA context early to avoid first-use stalls."""
    if not cuda_available():
        return
    a = torch.rand(8, 8, device="cuda")
    _ = a @ a
    torch.cuda.synchronize()


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


def _clock(device):
    """Return a timer reading, synchronising CUDA first when needed."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def _spikes_for(encoder, images, steps):
    """Build the [T,B,F] input for one device-path benchmark."""
    if encoder is None:
        flat = images.view(images.size(0), -1)
        return flat.unsqueeze(0).repeat(steps, 1, 1)
    return encoder.encode(images)


def _time_path(encoder, hidden, num_classes, steps, batch, device):
    """Median seconds for one encode+transfer+forward+backward on a device."""
    from snn_interpreter.spiking_net import SpikingNet
    net = SpikingNet(hidden=hidden, num_classes=num_classes).to(device)
    images = torch.rand(batch, 1, 28, 28)
    targets = torch.randint(0, num_classes, (batch,), device=device)
    sample = _spikes_for(encoder, images, steps)[:steps].to(device)
    net.forward_spikes(sample)  # untimed warmup (also primes CUDA)
    _clock(device)
    times = []
    for _ in range(_BENCH_REPS):
        net.zero_grad(set_to_none=True)
        start = _clock(device)
        out = net.forward_spikes(sample)
        torch.nn.functional.cross_entropy(out, targets).backward()
        times.append(_clock(device) - start)
    times.sort()
    return times[len(times) // 2]


def pick_auto(encoder, hidden, num_classes, num_steps, batch_size):
    """Benchmark both devices on this shape and return the faster one."""
    if not cuda_available():
        return "cpu"
    prime()
    mode = getattr(encoder, "coding", "raw")
    key = (mode, int(hidden), int(num_classes), int(num_steps), int(batch_size))
    if key not in _AUTO_CACHE:
        steps = min(int(num_steps), _BENCH_STEPS)
        batch = min(int(batch_size), _BENCH_BATCH)
        cpu = _time_path(encoder, hidden, num_classes, steps, batch, "cpu")
        gpu = _time_path(encoder, hidden, num_classes, steps, batch, "cuda")
        _AUTO_CACHE[key] = "cuda" if gpu < cpu else "cpu"
    return _AUTO_CACHE[key]


def _choose(requested, bench):
    """Resolve a device request to 'cuda' or 'cpu'."""
    choice = str(requested).lower()
    if choice in ("gpu", "cuda"):
        return "cuda"
    if choice == "cpu":
        return "cpu"
    if not cuda_available():
        return "cpu"
    return pick_auto(**bench) if bench else "cuda"


def resolve(requested="auto", **bench):
    """Map 'auto'/'gpu'/'cpu' to a torch.device, falling back to CPU."""
    chosen = _choose(requested, bench)
    return torch.device("cuda" if chosen == "cuda" else "cpu")


def warmup(net, steps, features=28 * 28):
    """Prime a CUDA net so the first training step isn't slow."""
    param = next(net.parameters(), None)
    if param is None or param.device.type != "cuda":
        return
    dummy = torch.zeros(1, steps, features, device=param.device)
    with torch.no_grad():
        net.forward_spikes(dummy)
    torch.cuda.synchronize()


class DeviceMixin:
    """Give an object a resolved torch device and a label property."""

    def _set_device(self, requested="auto", **bench):
        self._device = resolve(requested, **bench)

    @property
    def device(self):
        """Return the resolved device label ('cuda' or 'cpu')."""
        return self._device.type

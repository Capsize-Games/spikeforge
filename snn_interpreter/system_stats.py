"""CPU RAM and GPU VRAM snapshots for the dashboard monitor."""

import torch

from snn_interpreter import device as device_mod

try:  # psutil is optional; /proc/meminfo is the fallback
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

_MEMINFO = "/proc/meminfo"


def _block(total, available, name=None):
    """Build a memory summary dict from total/available bytes."""
    used = max(total - available, 0)
    percent = round(100.0 * used / total, 1) if total else 0.0
    info = {"total": int(total), "used": int(used),
            "available": int(available), "percent": percent}
    if name is not None:
        info["name"] = name
    return info


def _from_psutil():
    """Read CPU RAM through psutil."""
    mem = psutil.virtual_memory()
    return _block(mem.total, mem.available)


def _from_proc():
    """Read CPU RAM from /proc/meminfo when psutil is absent."""
    values = {}
    with open(_MEMINFO, "r", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) * 1024
    return _block(values["MemTotal"], values["MemAvailable"])


def cpu_memory():
    """Return CPU RAM total/used/available in bytes, or None."""
    if psutil is not None:
        return _from_psutil()
    try:
        return _from_proc()
    except (OSError, KeyError, ValueError):
        return None


def gpu_memory():
    """Return GPU VRAM total/used/available in bytes, or None."""
    if not device_mod.cuda_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return _block(total, free, name=torch.cuda.get_device_name(0))


def snapshot(engine=None):
    """Return the full resource snapshot sent to the client."""
    active = getattr(engine, "device", None) if engine is not None else None
    return {
        "cpu": cpu_memory(),
        "gpu": gpu_memory(),
        "device": {**device_mod.device_options(), "active": active},
    }

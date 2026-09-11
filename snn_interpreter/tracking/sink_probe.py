"""Isolated probes for the optional tracking backends.

Importing ``tensorboard`` or ``wandb`` is confined to this module so a missing
extra is a recorded reason rather than an import-time crash anywhere else in
the tracking package.
"""

from typing import Any, Optional


def tensorboard_writer() -> Optional[Any]:
    """Return TensorBoard's ``SummaryWriter`` class, or None if absent."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return None
    return SummaryWriter


def wandb_module() -> Optional[Any]:
    """Return the ``wandb`` module, or None when the extra is absent."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb


def tensorboard_available() -> bool:
    """Return True when the TensorBoard backend can be imported."""
    return tensorboard_writer() is not None


def wandb_available() -> bool:
    """Return True when the Weights & Biases backend can be imported."""
    return wandb_module() is not None

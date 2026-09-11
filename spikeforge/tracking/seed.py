"""Deterministic seeding for reproducible initialisation.

:func:`set_seed` fixes Python's and PyTorch's random state, which makes the
*initial parameters* of a freshly built topology and the data-shuffling order
follow from the seed alone. What is guaranteed to reproduce:

- the topology structure and its resolved ``TopologySpec``;
- the dataset, encode config, and training hyperparameters;
- the initial parameter values for the same library build on CPU;
- the reported library versions and the run seed.

What is **not** guaranteed bit-for-bit:

- CUDA kernels (cuDNN convolutions and parallel reductions) whose summation
  order varies per run unless the deterministic flags are enabled;
- hardware thread scheduling and any floating-point summation order that
  follows from it;
- dataset contents if the source files change between runs.

Enabling ``torch.use_deterministic_algorithms`` and the cuDNN deterministic
flags narrows the first gap at a throughput cost; this helper deliberately
leaves the global backend flags untouched so seeding never slows the default
training path.
"""

import random

import torch


def set_seed(seed: int) -> int:
    """Seed Python, PyTorch, and every CUDA device, returning the seed.

    Call this immediately before building a topology or starting a run so the
    initial weights and data shuffling follow from ``seed`` alone. It is the
    helper the manifest's ``seed`` field refers to.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

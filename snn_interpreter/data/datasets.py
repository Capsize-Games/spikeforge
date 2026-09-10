"""Registry of torchvision datasets usable for training.

Every dataset is normalised to 1x28x28 grayscale so the same network
architecture works across all of them; only the class count varies.
"""

from typing import Dict, List, Tuple

import torchvision.datasets as tv_datasets
from torchvision import transforms

from snn_interpreter.config import DATA_DIR

# name -> (dataset class, constructor kwargs, num_classes, description)
_REGISTRY: Dict[str, Tuple[type, Dict[str, object], int, str]] = {
    "mnist": (tv_datasets.MNIST, {}, 10, "Handwritten digits (0-9)"),
    "fashion": (
        tv_datasets.FashionMNIST,
        {},
        10,
        "Clothing/accessory categories",
    ),
    "kmnist": (tv_datasets.KMNIST, {}, 10, "Japanese Kuzushiji characters"),
    "qmnist": (tv_datasets.QMNIST, {}, 10, "Extended MNIST (NIST digits)"),
    "usps": (tv_datasets.USPS, {}, 10, "USPS handwritten digits"),
    "emnist_digits": (
        tv_datasets.EMNIST,
        {"split": "digits"},
        10,
        "EMNIST balanced digits",
    ),
    "emnist_letters": (
        tv_datasets.EMNIST,
        {"split": "letters"},
        26,
        "EMNIST handwritten letters",
    ),
    "cifar10": (
        tv_datasets.CIFAR10,
        {},
        10,
        "10-class colour objects (grayscaled)",
    ),
}

DEFAULT_DATASET = "mnist"


def transform() -> transforms.Compose:
    """Grayscale, resize to 28x28, and normalise to [0,1]."""
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )


def dataset_names() -> List[str]:
    """Return the list of selectable dataset keys."""
    return list(_REGISTRY.keys())


def dataset_info(name: str) -> Tuple[int, str]:
    """Return (num_classes, description) for a dataset key."""
    _, _, num_classes, desc = _REGISTRY[_resolve(name)]
    return num_classes, desc


def build_dataset(name: str, train: bool = True) -> tv_datasets.VisionDataset:
    """Instantiate a dataset, downloading it into the data dir."""
    cls, kwargs, _, _ = _REGISTRY[_resolve(name)]
    return cls(
        DATA_DIR,
        train=train,
        download=True,
        transform=transform(),
        **kwargs,
    )


def _resolve(name: str) -> str:
    """Map an unknown dataset name back to the default."""
    return name if name in _REGISTRY else DEFAULT_DATASET


def catalog() -> List[Dict[str, object]]:
    """Return dataset metadata for the client dropdown."""
    return [
        {"name": key, "classes": val[2], "description": val[3]}
        for key, val in _REGISTRY.items()
    ]

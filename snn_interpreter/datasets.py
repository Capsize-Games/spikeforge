"""Registry of torchvision datasets usable for training.

Every dataset is normalised to 1x28x28 grayscale so the same network
architecture works across all of them; only the class count varies.
"""

import torchvision.datasets as D
from torchvision import transforms

from snn_interpreter.config import DATA_DIR

# name -> (dataset class, constructor kwargs, num_classes, description)
_REGISTRY = {
    "mnist": (D.MNIST, {}, 10, "Handwritten digits (0-9)"),
    "fashion": (D.FashionMNIST, {}, 10, "Clothing/accessory categories"),
    "kmnist": (D.KMNIST, {}, 10, "Japanese Kuzushiji characters"),
    "qmnist": (D.QMNIST, {}, 10, "Extended MNIST (NIST digits)"),
    "usps": (D.USPS, {}, 10, "USPS handwritten digits"),
    "emnist_digits": (
        D.EMNIST, {"split": "digits"}, 10, "EMNIST balanced digits"
    ),
    "emnist_letters": (
        D.EMNIST, {"split": "letters"}, 26, "EMNIST handwritten letters"
    ),
    "cifar10": (D.CIFAR10, {}, 10, "10-class colour objects (grayscaled)"),
}

DEFAULT_DATASET = "mnist"


def transform():
    """Grayscale, resize to 28x28, and normalise to [0,1]."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ])


def dataset_names():
    """Return the list of selectable dataset keys."""
    return list(_REGISTRY.keys())


def dataset_info(name):
    """Return (num_classes, description) for a dataset key."""
    cls, kwargs, num_classes, desc = _REGISTRY[_resolve(name)]
    return num_classes, desc


def build_dataset(name, train=True):
    """Instantiate a dataset, downloading it into the data dir."""
    cls, kwargs, _, _ = _REGISTRY[_resolve(name)]
    return cls(
        DATA_DIR, train=train, download=True,
        transform=transform(), **kwargs,
    )


def _resolve(name):
    """Map an unknown dataset name back to the default."""
    return name if name in _REGISTRY else DEFAULT_DATASET


def catalog():
    """Return dataset metadata for the client dropdown."""
    return [
        {"name": key, "classes": val[2], "description": val[3]}
        for key, val in _REGISTRY.items()
    ]

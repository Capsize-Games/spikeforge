"""Registry of torchvision datasets usable for training.

Every dataset is normalised to 1x28x28 grayscale so the same network
architecture works across all of them; only the class count varies. Each
entry is a :class:`DatasetSpec` carrying its metadata, including the
``modality`` that tells the UI which encodings to offer.
"""

from typing import Dict, List, Tuple

import torchvision.datasets as tv_datasets
from torchvision import transforms

from snn_interpreter.config import DATA_DIR
from snn_interpreter.data.dataset_spec import DatasetSpec
from snn_interpreter.events import tonic_api

_REGISTRY: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        "mnist", 10, "Handwritten digits (0-9)", "image",
        tv_datasets.MNIST, {},
    ),
    "fashion": DatasetSpec(
        "fashion", 10, "Clothing/accessory categories", "image",
        tv_datasets.FashionMNIST, {},
    ),
    "kmnist": DatasetSpec(
        "kmnist", 10, "Japanese Kuzushiji characters", "image",
        tv_datasets.KMNIST, {},
    ),
    "qmnist": DatasetSpec(
        "qmnist", 10, "Extended MNIST (NIST digits)", "image",
        tv_datasets.QMNIST, {},
    ),
    "usps": DatasetSpec(
        "usps", 10, "USPS handwritten digits", "image",
        tv_datasets.USPS, {},
    ),
    "emnist_digits": DatasetSpec(
        "emnist_digits", 10, "EMNIST balanced digits", "image",
        tv_datasets.EMNIST, {"split": "digits"},
    ),
    "emnist_letters": DatasetSpec(
        "emnist_letters", 26, "EMNIST handwritten letters", "image",
        tv_datasets.EMNIST, {"split": "letters"},
    ),
    "cifar10": DatasetSpec(
        "cifar10", 10, "10-class colour objects (grayscaled)", "image",
        tv_datasets.CIFAR10, {},
    ),
    # Event modality: loaded through tonic, gated by the `events` extra.
    "n_mnist": DatasetSpec(
        "n_mnist", 10, "N-MNIST neuromorphic digits (DVS events)", "event",
        tonic_class="NMNIST", kwargs={"train": True},
    ),
    "dvs128_gesture": DatasetSpec(
        "dvs128_gesture", 11, "DVS128 hand-gesture events (11 classes)",
        "event", tonic_class="DVSGesture", kwargs={"train": True},
    ),
    "cifar10_dvs": DatasetSpec(
        "cifar10_dvs", 10, "CIFAR10-DVS converted object events", "event",
        tonic_class="CIFAR10DVS",
    ),
    "ssc": DatasetSpec(
        "ssc", 35, "Spiking Speech Commands (35 classes)", "event",
        tonic_class="SSC", kwargs={"split": "train"},
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


def dataset_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for a dataset key."""
    return _REGISTRY[_resolve(name)]


def dataset_modality(name: str) -> str:
    """Return a dataset's modality (``image`` or ``event``)."""
    return dataset_spec(name).modality


def dataset_info(name: str) -> Tuple[int, str]:
    """Return (num_classes, description) for a dataset key."""
    spec = dataset_spec(name)
    return spec.num_classes, spec.description


def build_dataset(
    name: str, train: bool = True, download: bool = True
) -> tv_datasets.VisionDataset:
    """Instantiate a dataset, optionally downloading it into the data dir."""
    spec = dataset_spec(name)
    if spec.cls is None:
        raise ValueError(f"dataset {spec.name!r} has no image loader")
    return spec.cls(
        DATA_DIR,
        train=train,
        download=download,
        transform=transform(),
        **spec.kwargs,
    )


def _resolve(name: str) -> str:
    """Map an unknown dataset name back to the default."""
    return name if name in _REGISTRY else DEFAULT_DATASET


def _available(spec: DatasetSpec) -> bool:
    """Return True when ``spec``'s loader can run in this environment."""
    if spec.modality == "event":
        return tonic_api.available()
    return spec.cls is not None


def dataset_available(name: str) -> bool:
    """Return True when a dataset's loader is usable right now.

    Image datasets depend on torchvision (always present); event datasets
    depend on the optional ``tonic`` package behind the ``events`` extra.
    """
    return _available(dataset_spec(name))


def catalog() -> List[Dict[str, object]]:
    """Return dataset metadata for the client dropdown.

    The ``modality`` and ``available`` keys are additive: existing clients
    keep reading ``name``/``classes``/``description`` unchanged, and event
    datasets can be gated on ``available`` until ``tonic`` is installed.
    """
    return [
        {
            "name": spec.name,
            "classes": spec.num_classes,
            "description": spec.description,
            "modality": spec.modality,
            "available": _available(spec),
        }
        for spec in _REGISTRY.values()
    ]

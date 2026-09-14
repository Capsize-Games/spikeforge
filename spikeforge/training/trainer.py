"""Core data loading and rate coding for the SNN trainer."""

from typing import Optional, Tuple

import torch
from snntorch import spikegen, utils
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from spikeforge.config import MNIST_PATH


class SNNTrainer:
    """Load an MNIST subset and encode it as rate-coded spike trains."""

    _data_path: str = MNIST_PATH
    _batch_size: int = 128
    _subset: int = 10
    _vectorization_num_steps: int = 10
    _vector_value: float = 0.5
    _data_size: Tuple[int, int] = (28, 28)
    _do_shuffle: bool = True
    _train: bool = True
    _download: bool = True
    _reconstruction_gain: float = 0.25
    _animation_interval: int = 200

    _transform: Optional[transforms.Compose] = None
    _mnist_train: Optional[datasets.MNIST] = None
    _mnist_train_subset: Optional[Dataset] = None
    _train_loader: Optional[DataLoader] = None
    _rate_coded_vector: Optional[torch.Tensor] = None
    _spike_data: Optional[torch.Tensor] = None
    _spike_data_low_gain: Optional[torch.Tensor] = None
    _spike_targets: Optional[torch.Tensor] = None
    _input_data: Optional[torch.Tensor] = None
    _subset_size: Optional[int] = None

    def __init__(
        self, data_path: str = "/tmp/data/mnist", batch_size: int = 128,
        subset: int = 10, vectorization_num_steps: int = 10,
        vector_value: float = 0.5, data_size: Tuple[int, int] = (28, 28),
        do_shuffle: bool = True, train: bool = True, download: bool = True,
        reconstruction_gain: float = 0.25, animation_interval: int = 200,
    ) -> None:
        """Store the dataset options and eagerly build the coded batch."""
        self._data_path = data_path
        self._batch_size = batch_size
        self._subset = subset
        self._vectorization_num_steps = vectorization_num_steps
        self._vector_value = vector_value
        self._data_size = data_size
        self._do_shuffle = do_shuffle
        self._train = train
        self._download = download
        self._reconstruction_gain = reconstruction_gain
        self._animation_interval = animation_interval
        self._initialise()

    def _initialise(self) -> None:
        """Run the data, vectorisation, and rate-coding setup steps."""
        self._prepare_data()
        self._load_training_data()
        self._prepare_vectorization()
        self._apply_rate_coding()

    def _prepare_data(self) -> None:
        """Build the transform and load the (subset) MNIST training set."""
        self._transform = transforms.Compose(
            [
                transforms.Resize(self._data_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,)),
            ]
        )
        self._mnist_train = datasets.MNIST(
            self._data_path,
            train=self._train,
            download=self._download,
            transform=self._transform,
        )
        self._mnist_train_subset = utils.data_subset(
            self._mnist_train, self._subset
        )
        self._subset_size = len(self._mnist_train_subset)

    def _load_training_data(self) -> None:
        """Wrap the reduced dataset in a shuffled data loader."""
        self._train_loader = DataLoader(
            self._mnist_train_subset,
            batch_size=self._batch_size,
            shuffle=self._do_shuffle,
        )

    def _prepare_vectorization(self) -> None:
        """Bernoulli-sample the legacy demo vector at the fixed value."""
        raw_vector = (
            torch.ones(self._vectorization_num_steps) * self._vector_value
        )
        self._rate_coded_vector = torch.bernoulli(raw_vector)

    def _apply_rate_coding(self) -> None:
        """Rate-code the first training batch at unit and low gain."""
        if self._train_loader is None:
            return
        data_it, targets_it = next(iter(self._train_loader))
        steps = self._vectorization_num_steps
        self._spike_data = spikegen.rate(data_it, num_steps=steps)
        self._spike_data_low_gain = spikegen.rate(
            data_it, num_steps=steps, gain=self._reconstruction_gain
        )
        self._spike_targets = targets_it
        self._input_data = data_it

    # --- read-only accessors for downstream exporters ---

    @property
    def spike_data(self) -> Optional[torch.Tensor]:
        """Return the gain=1 rate-coded spikes."""
        return self._spike_data

    @property
    def spike_data_low_gain(self) -> Optional[torch.Tensor]:
        """Return the low-gain rate-coded spikes."""
        return self._spike_data_low_gain

    @property
    def spike_targets(self) -> Optional[torch.Tensor]:
        """Return the labels for the coded batch."""
        return self._spike_targets

    @property
    def input_data(self) -> Optional[torch.Tensor]:
        """Return the raw (normalised) input batch."""
        return self._input_data

    @property
    def rate_coded_vector(self) -> Optional[torch.Tensor]:
        """Return the legacy Bernoulli demo vector."""
        return self._rate_coded_vector

    @property
    def subset_size(self) -> Optional[int]:
        """Return the number of samples after subsetting."""
        return self._subset_size

    @property
    def data_size(self) -> Tuple[int, int]:
        """Return the (H, W) size of each sample."""
        return self._data_size

    @property
    def num_steps(self) -> int:
        """Return the number of rate-coding time steps."""
        return self._vectorization_num_steps

    @property
    def interval(self) -> int:
        """Return the animation interval in milliseconds."""
        return self._animation_interval

    @property
    def gain(self) -> float:
        """Return the reconstruction gain used for the low-gain path."""
        return self._reconstruction_gain

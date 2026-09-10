"""Core data loading and rate coding for the SNN trainer."""

import torch
from snntorch import spikegen, utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from snn_interpreter.config import MNIST_PATH


class SSNTrainer:
    """Load an MNIST subset and encode it as rate-coded spike trains."""

    _data_path = MNIST_PATH
    _batch_size = 128
    _subset = 10
    _vectorization_num_steps = 10
    _vector_value = 0.5
    _data_size = (28, 28)
    _do_shuffle = True
    _train = True
    _download = True
    _reconstruction_gain = 0.25
    _animation_interval = 200

    _transform = None
    _mnist_train = None
    _mnist_train_subset = None
    _train_loader = None
    _rate_coded_vector = None
    _spike_data = None
    _spike_data_low_gain = None
    _spike_targets = None
    _input_data = None
    _subset_size = None

    def __init__(
        self, data_path="/tmp/data/mnist", batch_size=128, subset=10,
        vectorization_num_steps=10, vector_value=0.5, data_size=(28, 28),
        do_shuffle=True, train=True, download=True,
        reconstruction_gain=0.25, animation_interval=200,
    ):
        (self._data_path, self._batch_size, self._subset,
         self._vectorization_num_steps, self._vector_value,
         self._data_size, self._do_shuffle, self._train,
         self._download, self._reconstruction_gain,
         self._animation_interval) = (data_path, batch_size, subset,
                                      vectorization_num_steps,
                                      vector_value, data_size,
                                      do_shuffle, train, download,
                                      reconstruction_gain, animation_interval)
        self._prepare_data()
        self._load_training_data()
        self._prepare_vectorization()
        self._apply_rate_coding()

    def _prepare_data(self):
        self._transform = transforms.Compose([
            transforms.Resize(self._data_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ])
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

    def _load_training_data(self):
        self._train_loader = DataLoader(
            self._mnist_train_subset,
            batch_size=self._batch_size,
            shuffle=self._do_shuffle,
        )

    def _prepare_vectorization(self):
        raw_vector = torch.ones(
            self._vectorization_num_steps
        ) * self._vector_value
        self._rate_coded_vector = torch.bernoulli(raw_vector)

    def _apply_rate_coding(self):
        if self._train_loader is None:
            return
        data = iter(self._train_loader)
        data_it, targets_it = next(data)
        self._spike_data = spikegen.rate(
            data_it, num_steps=self._vectorization_num_steps
        )
        self._spike_data_low_gain = spikegen.rate(
            data_it,
            num_steps=self._vectorization_num_steps,
            gain=self._reconstruction_gain,
        )
        self._spike_targets = targets_it
        self._input_data = data_it

    # --- read-only accessors for downstream exporters ---

    @property
    def spike_data(self):
        return self._spike_data

    @property
    def spike_data_low_gain(self):
        return self._spike_data_low_gain

    @property
    def spike_targets(self):
        return self._spike_targets

    @property
    def input_data(self):
        return self._input_data

    @property
    def rate_coded_vector(self):
        return self._rate_coded_vector

    @property
    def subset_size(self):
        return self._subset_size

    @property
    def data_size(self):
        return self._data_size

    @property
    def num_steps(self):
        return self._vectorization_num_steps

    @property
    def interval(self):
        return self._animation_interval

    @property
    def gain(self):
        return self._reconstruction_gain

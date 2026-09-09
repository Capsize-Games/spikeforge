from typing import Optional
import snntorch as snn
from snntorch import utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SSNTrainer:
    _do_shuffle: bool = True
    _transform: Optional[transforms.Compose] = None
    _mnist_train: Optional[datasets.MNIST] = None
    _mnist_train_subset: Optional[datasets.MNIST] = None
    _train_loader: Optional[DataLoader] = None
    _batch_size: Optional[int] = None
    _data_path: Optional[str] = None
    _num_classes: Optional[int] = None
    _rate_coded_vector: Optional[torch.Tensor] = None
    _subset: Optional[int] = None
    _data_size: Optional[tuple] = None
    _dtype: torch.dtype = torch.float
    _vectorization_num_steps: Optional[int] = None
    _vector_value: Optional[float] = None

    def __init__(
        self,
        data_path: str = "/tmp/data/mnist",
        batch_size: int = 128,
        subset: int = 10,
        vectorization_num_steps: int = 10,
        vector_value: float = 0.5,
        data_size: tuple = (28, 28),
        do_shuffle: bool = True,
        train: bool = True,
        download: bool = True
    ):
        self._data_path = data_path
        self._batch_size = batch_size
        self._subset = subset
        self._vectorization_num_steps = vectorization_num_steps
        self._vector_value = vector_value
        self._data_size = data_size
        self._do_shuffle = do_shuffle
        self._train = train
        self._download = download

        self._prepare_data()
        self._load_training_data()
        self._prepare_vectorization()

    def _log(self, message: str):
        print(message)

    def _prepare_data(self):
        self._transform = transforms.Compose([
            transforms.Resize(self._data_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])
        self._mnist_train = datasets.MNIST(
            self._data_path,
            train=self._train,
            download=self._download,
            transform=self._transform
        )
        self._mnist_train_subset = utils.data_subset(
            self._mnist_train,
            self._subset
        )
        if self._mnist_train_subset is not None:
            self._log(
                f"The size of mnist_train_subset is {len(self._mnist_train_subset)}"
            )

    def _load_training_data(self):
        self._train_loader = DataLoader(
            self._mnist_train_subset,
            batch_size=self._batch_size,
            shuffle=self._do_shuffle
        )

    def _prepare_vectorization(self):
        raw_vector = torch.ones(
            self._vectorization_num_steps
        ) * self._vector_value
        self._rate_coded_vector = torch.bernoulli(raw_vector)
        if (
            self._mnist_train_subset is not None and 
            self._rate_coded_vector is not None
        ):
            self._log(f"Converted vector: {self._rate_coded_vector}")
            self._log(f"The output is spiking {self._rate_coded_vector.sum() * 100 / len(self._rate_coded_vector):.2f}% of the time")



def main():
    snn_trainer = SSNTrainer()


if __name__ == "__main__":
    main()

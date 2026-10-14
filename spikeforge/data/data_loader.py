"""Dataset loader construction shared by training and evaluation."""

from snntorch import utils
from torch.utils.data import DataLoader, Dataset

from spikeforge.data.datasets import build_dataset


def build_loader(
    dataset: str = "mnist",
    subset: int = 10,
    batch_size: int = 64,
    train: bool = True,
    download: bool = True,
) -> DataLoader:
    """Create a normalised loader reduced by the subset factor.

    ``download=False`` fails on a cold cache instead of fetching the
    dataset, so offline callers can request no-download behavior.
    """
    data: Dataset = build_dataset(dataset, train=train, download=download)
    if subset > 1:
        data = utils.data_subset(data, subset)
    return DataLoader(data, batch_size=batch_size, shuffle=train)

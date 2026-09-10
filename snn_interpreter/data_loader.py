"""Dataset loader construction shared by training and evaluation."""

from snntorch import utils
from torch.utils.data import DataLoader

from snn_interpreter.datasets import build_dataset


def build_loader(dataset="mnist", subset=10, batch_size=64, train=True):
    """Create a normalised loader reduced by the subset factor."""
    data = build_dataset(dataset, train=train)
    if subset > 1:
        data = utils.data_subset(data, subset)
    return DataLoader(data, batch_size=batch_size, shuffle=train)

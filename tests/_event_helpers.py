"""The stubbed event backend shared by the reference-path tests."""

from typing import Any, Dict, List, Tuple

import scripts.train_reference_models as trm

#: Split lengths of the stub. Unequal, and neither a multiple of the eval
#: batch, so an off-by-one or a truncation shows up as a wrong count.
_TRAIN_SAMPLES = 23
_TEST_SAMPLES = 7

class _FakeTonic:
    """A tonic-shaped 28x28 dataset with distinct train and test splits."""

    sensor_size = (28, 28, 2)

    def __init__(self, root: str, **kwargs: Any) -> None:
        """Record which split the registry's kwargs selected."""
        self._train = bool(kwargs.get("train"))

    def __len__(self) -> int:
        """Return this split's own length."""
        return _TRAIN_SAMPLES if self._train else _TEST_SAMPLES

    def __getitem__(self, index: int) -> Tuple[Dict[str, List[int]], int]:
        """Return a two-event stream whose column encodes the split."""
        column = index % 27
        offset = 0 if self._train else 1
        stream = {
            "x": [column, column + offset],
            "y": [index % 28, (index + 1) % 28],
            "t": [0, 1000],
            "p": [1, 0],
        }
        return stream, index % 4


def _reference(**overrides: Any) -> trm.Reference:
    """Return a tiny event reference configuration."""
    fields: Dict[str, Any] = {
        "name": "probe-event",
        "dataset": "n_mnist",
        "topology": "fc_legacy",
        "epochs": 1,
        "num_steps": 4,
        "hidden": 8,
        "batch_size": 4,
        "topology_params": {"input_size": 784},
    }
    fields.update(overrides)
    return trm.Reference(**fields)

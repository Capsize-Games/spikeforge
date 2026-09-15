"""Typed metadata for a dataset registry entry.

The registry used to store an unlabelled tuple per dataset;
:class:`DatasetSpec` replaces it with named fields so callers (and the
WebSocket payload) can see a dataset's modality without unpacking positional
data. It is purely additive: the tuple's ``num_classes``/``description`` live
on as attributes and the image constructors keep their ``cls``/``kwargs``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

#: The data modality a registry dataset delivers.
Modality = Literal["image", "event", "sequence"]

#: The dataset splits a registry entry can declare.
Split = Literal["train", "test"]


@dataclass(frozen=True)
class DatasetSpec:
    """A registry entry: how to build a dataset plus its metadata.

    Image datasets set ``cls`` to their torchvision class and ``kwargs`` to
    its constructor arguments. Event datasets leave ``cls`` as ``None``,
    point ``tonic_class`` at the matching ``tonic.datasets`` class, and
    declare their splits in ``splits``. Sequence datasets are fully
    synthetic and need no loader. ``modality`` drives which encodings the UI
    offers.

    ``kwargs`` holds the constructor arguments that are the same for every
    split (EMNIST's character ``split``, for instance, selects a character
    set rather than a train/test partition). ``splits`` maps a split name to
    the arguments that select it, because tonic's classes disagree on how:
    ``NMNIST`` and ``DVSGesture`` take ``train=True/False`` while ``SSC``
    takes ``split="train"/"test"``. A dataset with no ``"test"`` entry
    declares that it ships no held-out partition, which
    :func:`~spikeforge.data.datasets.dataset_split_kwargs` turns into a
    named error rather than a silent fallback to the training data.
    """

    name: str
    num_classes: int
    description: str
    modality: Modality = "image"
    cls: Optional[type] = None
    kwargs: Dict[str, object] = field(default_factory=dict)
    tonic_class: Optional[str] = None
    splits: Dict[str, Dict[str, Any]] = field(default_factory=dict)

"""Typed metadata for a dataset registry entry.

The registry used to store an unlabelled tuple per dataset;
:class:`DatasetSpec` replaces it with named fields so callers (and the
WebSocket payload) can see a dataset's modality without unpacking positional
data. It is purely additive: the tuple's ``num_classes``/``description`` live
on as attributes and the image constructors keep their ``cls``/``kwargs``.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

#: The data modality a registry dataset delivers.
Modality = Literal["image", "event"]


@dataclass(frozen=True)
class DatasetSpec:
    """A registry entry: how to build a dataset plus its metadata.

    Image datasets set ``cls`` to their torchvision class and ``kwargs`` to
    its constructor arguments. Event datasets leave ``cls`` as ``None``,
    point ``tonic_class`` at the matching ``tonic.datasets`` class, and put
    that class's split arguments in ``kwargs``. ``modality`` drives which
    encodings the UI offers.
    """

    name: str
    num_classes: int
    description: str
    modality: Modality = "image"
    cls: Optional[type] = None
    kwargs: Dict[str, object] = field(default_factory=dict)
    tonic_class: Optional[str] = None

"""The frozen core-boundary vocabulary: roots, shims, and the ship set.

The forbidden import roots, the design document's frozen boundary, the
sanctioned lazy-shim allow-list, and the module set the core
distribution ships -- shared by the visitor and the scan driver.
"""

from pathlib import Path
from typing import NamedTuple, Sequence, Set

#: Repository root (this file lives in ``scripts/``).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Forbidden import roots, frozen for ``protocol_version`` 1.0 (13 tokens).
#:
#: The set mixes distribution names and import roots deliberately. Where they
#: differ: the ``lava-nc`` *distribution* is imported as ``lava``, so only
#: ``lava`` can ever appear in an ``import`` statement; ``huggingface_hub``,
#: ``pydantic``, ``fastapi``, ``uvicorn``, ``nir``, ``nirtorch``, ``onnx``,
#: ``onnxruntime``, ``tonic``, ``tensorboard``, and ``wandb`` use the same
#: token for both. ``FROZEN_BOUNDARY`` below records the full design list for
#: the ``--list`` output.
FORBIDDEN_ROOTS: Set[str] = frozenset(
    {
        "fastapi",
        "pydantic",
        "uvicorn",
        "huggingface_hub",
        "nir",
        "nirtorch",
        "onnx",
        "onnxruntime",
        "norse",
        "lava",
        "tonic",
        "tensorboard",
        "wandb",
    }
)

#: The design document's frozen boundary, including the ``lava-nc``
#: distribution token that never appears as an import root.
FROZEN_BOUNDARY: Sequence[str] = (
    "fastapi",
    "pydantic",
    "uvicorn",
    "huggingface_hub",
    "nir",
    "nirtorch",
    "onnx",
    "onnxruntime",
    "norse",
    "lava",
    "lava-nc",
    "tonic",
    "tensorboard",
    "wandb",
)

#: Sanctioned lazy shims (tier 3) that may import a forbidden SDK *inside a
#: function* and report availability instead of raising. Module-level imports
#: are never exempt, even in these modules.
#:
#: The design document names ``tracking.tensorboard_sink`` and
#: ``tracking.wandb_sink`` as the tracking exception; the concrete ``import
#: wandb`` lives in ``tracking.sink_probe``, whose own docstring calls it "the
#: only module ... that imports tensorboard or wandb". It is the same
#: sanctioned site, so it is listed here too.
ALLOWED_LOCAL_IMPORTERS: Set[str] = frozenset(
    {
        "spikeforge.nir_bridge.api",
        "spikeforge.onnx_bridge.api",
        "spikeforge.events.tonic_api",
        "spikeforge.tracking.tensorboard_sink",
        "spikeforge.tracking.wandb_sink",
        "spikeforge.tracking.sink_probe",
    }
)

#: Directories the core distribution never ships, so they are never scanned.
#: Mirrors the ``exclude`` list in ``packages/spikeforge/pyproject.toml``
#: (``server*``, ``tests*``, ``spikeforge_targets*``, ``spikeforge_hub*``).
EXCLUDED_PREFIXES = ("server", "tests", "spikeforge_targets", "spikeforge_hub")

#: Package directories the core distribution ships, relative to the repo root.
SCAN_DIRS = ("spikeforge",)

#: Top-level single-file modules the core distribution ships.
SCAN_FILES = ("main.py", "main_encodings.py")


class Violation(NamedTuple):
    """A single forbidden import found in a scanned module."""

    path: Path
    line: int
    kind: str
    root: str
    statement: str

"""Build, serialize, and read the reproducibility record for one run.

A manifest captures everything needed to recreate and compare a training run:
the dataset, topology and params, encode config, training hyperparameters, the
resolved :class:`~spikeforge.topology.spec.TopologySpec`, the library
versions, the random seed, and the metric history. A schema version and a
stable hash of the configuration let two runs be compared at a glance.
"""

import time
from typing import Any, Dict, List, Mapping, Optional

from spikeforge.tracking.config_hash import config_hash as _hash_config
from spikeforge.tracking.versions import library_versions

#: Manifest schema revision; bump when the payload shape changes.
SCHEMA_VERSION = 1

#: Properties that reproduce for a fixed seed and library build.
_GUARANTEED = (
    "topology structure and resolved TopologySpec",
    "dataset, encode config, and training hyperparameters",
    "library versions and the run seed",
    "initial parameter values for the same library build on CPU",
)

#: Properties that can still vary despite an identical manifest and seed.
_NOT_GUARANTEED = (
    "bit-exact CUDA kernels (cuDNN, parallel reductions)",
    "hardware thread scheduling and float summation order",
    "dataset contents if the source files change between runs",
)


class ReproducibilityManifest:
    """The record that lets two runs be compared and one run be recreated."""

    def __init__(
        self,
        config: Mapping[str, Any],
        seed: Optional[int] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[float] = None,
        versions: Optional[Mapping[str, Any]] = None,
        tracking: Optional[Mapping[str, Any]] = None,
        determinism: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store the configuration, seed, history, and library versions.

        ``tracking`` and ``determinism`` are additive, opt-in blocks: they are
        omitted entirely from :meth:`to_dict` unless the caller supplies them,
        so a default run's manifest is byte-for-byte unchanged.
        """
        self._config = dict(config)
        self._seed = None if seed is None else int(seed)
        self._history = [dict(point) for point in (history or [])]
        self._versions = dict(versions or library_versions())
        self._created_at = (
            time.time() if created_at is None else float(created_at)
        )
        self._tracking = dict(tracking) if tracking else None
        self._determinism = dict(determinism) if determinism else None

    @property
    def config_hash(self) -> str:
        """Return the stable hash of the configuration alone."""
        return _hash_config(self._config)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able manifest persisted with a checkpoint."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": self._created_at,
            "config_hash": self.config_hash,
            "seed": self._seed,
            "versions": self._versions,
            "config": self._config,
            "history": self._history,
            "reproducible": {
                "bit_exact": False,
                "guaranteed": list(_GUARANTEED),
                "not_guaranteed": list(_NOT_GUARANTEED),
            },
        }
        if self._tracking is not None:
            payload["tracking"] = self._tracking
        if self._determinism is not None:
            payload["determinism"] = self._determinism
        return payload

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "ReproducibilityManifest":
        """Rebuild a manifest from :meth:`to_dict` output."""
        return cls(
            config=data.get("config", {}),
            seed=data.get("seed"),
            history=data.get("history", []),
            created_at=data.get("created_at"),
            versions=data.get("versions"),
            tracking=data.get("tracking"),
            determinism=data.get("determinism"),
        )

"""Checkpoint save/restore behaviour shared with the training engine."""

from typing import Any, Dict, List, Optional

from spikeforge.network import model_store
from spikeforge.neurons.registry import NEURONS
from spikeforge.tracking import sinks
from spikeforge.tracking.manifest import ReproducibilityManifest


class CheckpointMixin:
    """Persist and restore a model, its metadata, and its manifest."""

    _seed: Optional[int]
    _epochs: int
    _subset: int
    _batch_size: int
    _tracking: Optional[str]
    _deterministic: bool
    _determinism_report: Optional[Dict[str, Any]]

    def _stored_meta(self, checkpoint: str) -> Dict[str, Any]:
        """Return a checkpoint's stored meta mapping, or an empty dict."""
        return model_store.load(checkpoint).get("meta", {})

    def _restore(self, checkpoint: str) -> None:
        """Load weights and, unless overridden, the input mode."""
        ckpt = model_store.load(checkpoint)
        self._net.load_state_dict(ckpt["state_dict"])
        if self._explicit_mode is None:
            meta = ckpt.get("meta", {})
            self._input_mode = meta.get("input_mode", "raw")

    def save(
        self, name: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Persist the local manifest first, then forward it to any sink.

        Writing the local checkpoint first means a tracker outage can never
        lose a run; the sink is a strictly additional destination.
        """
        manifest = self._manifest(history)
        path = model_store.save(
            name, self._net, self._meta(), history, manifest
        )
        sinks.emit(manifest)
        return path

    def _stage_neurons(self) -> Dict[str, str]:
        """Return the resolved kind of every neuron stage, keyed by name.

        This is an additive, human-readable summary of the spec's per-stage
        heterogeneity, so a checkpoint's neuron layout is visible without
        decoding the full spec.
        """
        return {
            stage.name: stage.kind
            for stage in self._spec.stages
            if stage.kind in NEURONS
        }

    def _meta(self) -> Dict[str, Any]:
        """Return the checkpoint metadata card for the current model.

        ``encode_spec`` is the normalised, versioned encode contract a bundle
        can freeze verbatim; the legacy ``encode`` key is kept unchanged so
        existing readers keep working.
        """
        geometry = self._input_geometry()
        return {
            "dataset": self._dataset,
            "hidden": self._hidden,
            "beta": self._beta,
            "lr": self._lr,
            "num_steps": self.num_steps,
            "num_classes": self._num_classes,
            "input_mode": self._input_mode,
            "coding": self._input_mode,
            "device": self._device.type,
            "encode": self._encode.model_dump() if self._encode else None,
            "encode_spec": self._encode_spec_meta(),
            "input_size": (
                None
                if geometry is None
                else [int(geometry[0]), int(geometry[1])]
            ),
            "topology": self._topology,
            "topology_params": dict(self._architecture),
            "stage_neurons": self._stage_neurons(),
            "spec": self._spec.to_dict(),
        }

    def _manifest(
        self, history: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Build the reproducibility manifest persisted with the checkpoint."""
        return ReproducibilityManifest(
            self._manifest_config(),
            seed=self._seed,
            history=history,
            tracking=self._tracking_block(),
            determinism=self._determinism_block(),
        ).to_dict()

    def _tracking_block(self) -> Optional[Dict[str, Any]]:
        """Return the tracking block, or None when no sink was requested."""
        if not self._tracking:
            return None
        return sinks.describe(self._tracking)

    def _determinism_block(self) -> Optional[Dict[str, Any]]:
        """Return the determinism block, or None in the default fast mode."""
        if not self._deterministic:
            return None
        return dict(self._determinism_report or {})

    def _manifest_config(self) -> Dict[str, Any]:
        """Return the reproducibility-relevant configuration of the run."""
        config = {
            "dataset": self._dataset,
            "topology": self._topology,
            "topology_params": dict(self._architecture),
            "encode": self._encode.model_dump() if self._encode else None,
            "input_mode": self._input_mode,
            "coding": self._input_mode,
            "mode": self._mode.value,
            "device": self._device.type,
            "spec": self._spec.to_dict(),
        }
        config.update(self._hyperparameters())
        return config

    def _hyperparameters(self) -> Dict[str, Any]:
        """Return the manifest's training hyperparameter block."""
        return {
            "hidden": self._hidden,
            "beta": self._beta,
            "lr": self._lr,
            "epochs": self._epochs,
            "subset": self._subset,
            "batch_size": self._batch_size,
            "num_steps": self.num_steps,
            "num_classes": self._num_classes,
        }

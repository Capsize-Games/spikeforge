"""Checkpoint save/restore behaviour shared with the training engine."""

from typing import Any, Dict, List, Optional

from snn_interpreter.network import model_store


class CheckpointMixin:
    """Persist and restore a model plus its training metadata."""

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
        """Persist the current model, its metadata, and metric history."""
        meta = {
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
            "topology": self._topology,
            "topology_params": dict(self._architecture),
            "spec": self._spec.to_dict(),
        }
        return model_store.save(name, self._net, meta, history)

"""Bundle loading, session storage, and inference for ``spikeforge-serve``.

The service is a thin core over
:class:`~spikeforge.serving.session.InferenceSession`: it loads one
:class:`~spikeforge.serving.bundle.DeploymentBundle`, keeps a stateful session
per client id, and turns requests into steps. It imports no web framework, so
the ASGI layer and this core can be tested apart.
"""

import threading
from typing import Any, Dict, Iterator, List, Optional, Union

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.prediction import Prediction
from spikeforge.serving.session import InferenceSession
from spikeforge_serve import metrics as serve_metrics

#: Session id used when a request does not name one.
DEFAULT_SESSION = "default"

#: A bundle may be passed as a path or already loaded.
BundleSource = Union[str, DeploymentBundle]


def _as_frame(frame: Any) -> torch.Tensor:
    """Return ``frame`` as a float tensor with a leading batch dimension."""
    tensor = torch.as_tensor(frame, dtype=torch.float32)
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor


class ServingService:
    """Own one bundle and the stateful sessions that stream through it."""

    def __init__(
        self,
        bundle: BundleSource,
        device: Union[str, torch.device] = "cpu",
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
    ) -> None:
        """Remember the ``bundle`` source and the execution settings."""
        self._device = device
        self._mode = mode
        self._bundle: Optional[DeploymentBundle] = (
            bundle if isinstance(bundle, DeploymentBundle) else None
        )
        self._path: Optional[str] = (
            bundle if isinstance(bundle, str) else None
        )
        self._sessions: Dict[str, InferenceSession] = {}
        self._lock = threading.RLock()

    @property
    def bundle(self) -> DeploymentBundle:
        """Return the loaded bundle, reading it from disk on first use."""
        with self._lock:
            return self._ensure_bundle()

    def session(self, session_id: str = DEFAULT_SESSION) -> InferenceSession:
        """Return the session for ``session_id``, creating it on first use."""
        with self._lock:
            sid = session_id or DEFAULT_SESSION
            existing = self._sessions.get(sid)
            if existing is None:
                existing = InferenceSession.load(
                    self._ensure_bundle(),
                    device=self._device,
                    mode=self._mode,
                )
                self._sessions[sid] = existing
            return existing

    def reset(self, session_id: str = DEFAULT_SESSION) -> int:
        """Clear ``session_id``'s temporal state and return its step count."""
        with self._lock:
            session = self.session(session_id)
            session.reset()
            return session.steps

    def predict(
        self,
        frames: List[Any],
        session_id: str = DEFAULT_SESSION,
        encoded: bool = False,
    ) -> List[Prediction]:
        """Advance the session over ``frames``; one prediction per frame."""
        with self._lock:
            session = self.session(session_id)
            before = session.steps
            predictions = [
                self._step(session, frame, encoded) for frame in frames
            ]
            serve_metrics.count_steps(session.steps - before)
            return predictions

    def stream(
        self,
        frames: Any,
        session_id: str = DEFAULT_SESSION,
        encoded: bool = False,
    ) -> Iterator[Prediction]:
        """Yield one prediction per frame, carrying the session's state."""
        with self._lock:
            session = self.session(session_id)
            for frame in frames:
                before = session.steps
                prediction = self._step(session, frame, encoded)
                serve_metrics.count_steps(session.steps - before)
                yield prediction

    def _ensure_bundle(self) -> DeploymentBundle:
        """Return the cached bundle, loading it once from the path."""
        if self._bundle is None and self._path is not None:
            self._bundle = DeploymentBundle.load(self._path, strict=True)
        if self._bundle is None:  # pragma: no cover - defensive
            raise RuntimeError("no bundle configured")
        return self._bundle

    def _step(
        self, session: InferenceSession, frame: Any, encoded: bool
    ) -> Prediction:
        """Advance ``session`` one frame, encoding a raw sample when asked."""
        if encoded:
            return session.step(_as_frame(frame))
        return self._run_sample(session, frame)

    def _run_sample(
        self, session: InferenceSession, sample: Any
    ) -> Prediction:
        """Encode one raw sample and stream its whole spike train."""
        spikes = session.encode(sample)
        prediction: Optional[Prediction] = None
        for index in range(int(spikes.size(0))):
            prediction = session.step(spikes[index])
        if prediction is None:  # pragma: no cover - a train is never empty
            raise ValueError("encoded sample produced no timesteps")
        return prediction

"""A stateful, stream-oriented inference session over a deployment bundle."""

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Union,
)

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving import preprocess, tensor_codec
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import BackendError, StateError
from spikeforge.serving.inference_backend import (
    NIR_C_BACKEND,
    TORCH_BACKEND,
    InferenceBackend,
    resolve_backend,
)
from spikeforge.serving.nir_c_backend import NirCBackend
from spikeforge.serving.prediction import Prediction
from spikeforge.serving.torch_backend import TorchBackend
from spikeforge.simulator.step_stages import PostStep
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule


class InferenceSession:
    """Drive one topology timestep at a time, carrying state.

    A session steps a :class:`DeploymentBundle` through one
    :class:`InferenceBackend` — ``torch`` (the default) or ``nir-c``.
    Every :meth:`step` advances exactly one timestep and the carried
    state persists until :meth:`reset`.
    """

    def __init__(
        self, bundle: DeploymentBundle,
        module: Optional[StageModule] = None,
        device: Union[str, torch.device] = "cpu",
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        post_step: Optional[PostStep] = None,
        backend: str = TORCH_BACKEND,
        c_lib: Optional[str] = None,
    ) -> None:
        """Open ``backend`` over ``bundle``; see :meth:`load`.

        The torch backend needs the bundle's ``module``; the ``nir-c``
        backend builds its own graph from the bundle, takes no module
        and no ``post_step`` hook, and reads its C library from
        ``c_lib`` or the ``NIR_C_RUNTIME_LIB`` environment variable.
        """
        self._bundle = bundle
        self._backend_name = resolve_backend(backend)
        self._backend = _open_backend(
            self._backend_name, bundle, module, device, post_step, c_lib
        )
        self._device = torch.device(device)
        self._mode = mode
        self._post_step = post_step
        self._totals: Optional[torch.Tensor] = None
        self._steps = 0

    @classmethod
    def load(
        cls, bundle: Union[DeploymentBundle, str],
        device: Union[str, torch.device] = "cpu",
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        strict: bool = True, post_step: Optional[PostStep] = None,
        backend: str = TORCH_BACKEND,
        c_lib: Optional[str] = None,
    ) -> "InferenceSession":
        """Load a session from a bundle object or a ``.spkf`` path."""
        name = resolve_backend(backend)
        resolved = _resolve_bundle(bundle, strict)
        if name == NIR_C_BACKEND:
            return cls(
                resolved, device=device, mode=mode, post_step=post_step,
                backend=name, c_lib=c_lib,
            )
        module = resolved.build_module(device=device)
        return cls(
            resolved, module, device=device, mode=mode,
            post_step=post_step, backend=name,
        )

    def __enter__(self) -> "InferenceSession":
        """Return this session for a ``with`` block."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Close the backend on block exit."""
        self.close()

    @property
    def bundle(self) -> DeploymentBundle:
        """Return the bundle this session was loaded from."""
        return self._bundle

    @property
    def backend_name(self) -> str:
        """Return the backend this session steps through."""
        return self._backend_name

    @property
    def spec(self) -> TopologySpec:
        """Return the topology the session executes."""
        return self._backend.spec

    @property
    def mode(self) -> ExecutionMode:
        """Return the execution mode the session runs in."""
        return self._mode

    @property
    def post_step(self) -> Optional[PostStep]:
        """Return the per-step transform, or ``None`` for a plain session."""
        return self._post_step

    @property
    def steps(self) -> int:
        """Return how many steps have been taken since the last reset."""
        return self._steps

    def reset(self) -> None:
        """Clear the carried state and the cumulative readout."""
        self._backend.reset()
        self._totals = None
        self._steps = 0

    def step(self, frame: Any) -> Prediction:
        """Advance one timestep and return the resulting prediction."""
        return self._advance(self._as_input(frame))

    def run_stream(self, frames: Iterable[Any]) -> Iterator[Prediction]:
        """Yield one :class:`Prediction` per frame, carrying state."""
        for frame in frames:
            yield self.step(frame)

    def encode(self, sample: Any) -> torch.Tensor:
        """Encode a raw sample with the bundle's frozen encode spec.

        The shared :func:`~spikeforge.serving.preprocess.encode` is used, and
        ``topology_spec`` is supplied so the served spikes land in the same
        ``[T, B, ...]`` shape training produced. The spec is the one frozen in
        the bundle, so a stream cannot re-encode with a different contract.
        """
        spec = self._bundle.encode_spec()
        return preprocess.encode(
            sample, spec, self._backend.spec, geometry=spec.input_size
        )

    def step_sample(self, sample: Any) -> Prediction:
        """Encode a raw sample and return its final timestep prediction.

        Every encoded frame advances the carried state, so ``steps`` counts
        timesteps and ``mean_logits`` agrees with whole-tensor simulation.
        """
        predictions = self.run_stream(self.encode(sample))
        final = next(predictions)
        for prediction in predictions:
            final = prediction
        return final

    def run_samples(self, samples: Iterable[Any]) -> Iterator[Prediction]:
        """Yield each raw sample's final prediction, carrying neuron state."""
        for sample in samples:
            yield self.step_sample(sample)

    def state(self) -> Dict[str, Any]:
        """Return the session's serializable carried state."""
        return {
            "steps": self._steps,
            "totals": tensor_codec.encode(self._totals),
            "state": self._backend.capture_state(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state`.

        A state whose stage names do not match this session's spec is refused
        with :class:`StateError`.
        """
        if not isinstance(state, Mapping):
            raise StateError("state payload must be an object")
        self._backend.restore_state(state.get("state") or {})
        self._totals = tensor_codec.decode(state.get("totals"), self._device)
        self._steps = int(state.get("steps", 0) or 0)

    def close(self) -> None:
        """Close the backend; stepping or resetting after this refuses."""
        self._backend.close()

    def _advance(self, inputs: torch.Tensor) -> Prediction:
        """Run one backend step and fold its readout into the totals."""
        readout, spikes = self._backend.step(inputs)
        total = readout if self._totals is None else self._totals + readout
        self._totals = total.detach()
        self._steps += 1
        return Prediction(
            logits=readout,
            spikes=spikes,
            class_totals=self._totals,
            steps=self._steps,
            label=int(self._totals.argmax(dim=-1).reshape(-1)[0]),
        )

    def _as_input(self, frame: Any) -> torch.Tensor:
        """Return ``frame`` as a tensor on the session's device."""
        if isinstance(frame, torch.Tensor):
            return frame.to(self._device)
        return torch.as_tensor(frame, dtype=torch.float32, device=self._device)


def _open_backend(
    name: str, bundle: DeploymentBundle,
    module: Optional[StageModule],
    device: Union[str, torch.device],
    post_step: Optional[PostStep], c_lib: Optional[str],
) -> InferenceBackend:
    """Return the ``name`` backend, refusing contradictory options."""
    if name == NIR_C_BACKEND:
        if module is not None:
            raise BackendError(
                "the nir-c backend builds its own graph from "
                "the bundle; pass no module"
            )
        if post_step is not None:
            raise BackendError(
                "the nir-c backend cannot apply a torch post_step hook"
            )
        return NirCBackend.from_bundle(bundle, device=device, lib=c_lib)
    if module is None:
        raise BackendError("the torch backend needs the bundle's module")
    return TorchBackend(module, device=device, post_step=post_step)


def _resolve_bundle(
    bundle: Union[DeploymentBundle, str], strict: bool
) -> DeploymentBundle:
    """Return ``bundle`` itself, or the bundle loaded from its path."""
    if isinstance(bundle, str):
        return DeploymentBundle.load(bundle, strict=strict)
    return bundle

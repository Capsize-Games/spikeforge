"""A stateful, stream-oriented inference session over a deployment bundle."""

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.serving import preprocess, tensor_codec
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.errors import StateError
from spikeforge.serving.prediction import Prediction
from spikeforge.serving.state_tree import StateTree
from spikeforge.simulator.grad_policy import detach_state
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.state import initial_state, neuron_kinds
from spikeforge.simulator.step_stages import PostStep, step_stages
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import (
    CURRENT_KEY,
    PREV_KEY,
    StageModule,
)

#: Carried-state entries that are not neuron stages.
_RESERVED: Tuple[str, ...] = (PREV_KEY, CURRENT_KEY)


class InferenceSession:
    """Drive one topology timestep at a time, carrying neuron state.

    A session is built from a :class:`DeploymentBundle`. Every :meth:`step`
    advances exactly one timestep and the carried state persists until
    :meth:`reset`. The per-step body is the same shared
    :func:`~spikeforge.simulator.step_stages.step_stages` the closed-loop
    simulator uses, so a stream and a whole-tensor ``run`` agree by
    construction rather than by convention.
    """

    def __init__(
        self,
        bundle: DeploymentBundle,
        module: StageModule,
        device: Union[str, torch.device] = "cpu",
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        post_step: Optional[PostStep] = None,
    ) -> None:
        """Wrap an already-built ``module`` and its ``bundle``.

        ``post_step`` is an optional per-step transform of the stage outputs
        and carried state, applied by the shared step body. Leaving it unset
        keeps every step byte-identical; a quantization hook such as
        ``spikeforge_targets.activation_quant.ActivationQuantizer`` is the
        intended user, and it records its own error report.
        """
        self._bundle = bundle
        self._module = module
        self._device = torch.device(device)
        self._mode = mode
        self._post_step = post_step
        self._spec: TopologySpec = spec_of(module)
        self._neurons: Tuple[str, ...] = tuple(neuron_kinds(self._spec))
        self._state: Optional[Dict[str, Any]] = None
        self._totals: Optional[torch.Tensor] = None
        self._steps = 0

    @classmethod
    def load(
        cls,
        bundle: Union[DeploymentBundle, str],
        device: Union[str, torch.device] = "cpu",
        mode: ExecutionMode = ExecutionMode.PRODUCTION,
        strict: bool = True,
        post_step: Optional[PostStep] = None,
    ) -> "InferenceSession":
        """Load a session from a bundle object or a ``.spkf`` path.

        ``post_step`` opts into a per-step transform (activation/membrane
        quantization); it defaults to ``None`` so existing behaviour is
        unchanged.
        """
        resolved = _resolve_bundle(bundle, strict)
        module = resolved.build_module(device=device)
        return cls(
            resolved,
            module,
            device=device,
            mode=mode,
            post_step=post_step,
        )

    @property
    def bundle(self) -> DeploymentBundle:
        """Return the bundle this session was loaded from."""
        return self._bundle

    @property
    def spec(self) -> TopologySpec:
        """Return the topology the session executes."""
        return self._spec

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
        self._state = None
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
            sample, spec, self._spec, geometry=spec.input_size
        )

    def step_sample(self, sample: Any) -> Prediction:
        """Encode one raw sample and advance one step over its spike train."""
        return self.step(self.encode(sample))

    def run_samples(self, samples: Iterable[Any]) -> Iterator[Prediction]:
        """Yield one :class:`Prediction` per raw sample, carrying state."""
        for sample in samples:
            yield self.step_sample(sample)

    def state(self) -> Dict[str, Any]:
        """Return the session's serializable carried state."""
        return {
            "steps": self._steps,
            "totals": _encode_optional(self._totals),
            "state": StateTree(self._state).to_dict(),
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`state`.

        A state whose stage names do not match this session's spec is refused
        with :class:`StateError`.
        """
        if not isinstance(state, Mapping):
            raise StateError("state payload must be an object")
        tree = StateTree.from_dict(state.get("state") or {}).to(self._device)
        self._validate(tree.raw())
        self._state = dict(tree.raw()) or None
        self._totals = _decode_optional(state.get("totals"), self._device)
        self._steps = int(state.get("steps", 0) or 0)

    def _advance(self, inputs: torch.Tensor) -> Prediction:
        """Run one shared step and fold its readout into the totals."""
        with torch.no_grad():
            if self._state is None:
                self._state = dict(initial_state(self._spec, inputs))
            outputs, state = step_stages(
                self._module,
                inputs,
                self._state,
                post_step=self._post_step,
            )
            readout = outputs[self._spec.output].detach()
            total = readout if self._totals is None else self._totals + readout
            self._state = detach_state(state)
            spikes = self._spikes(outputs)
            totals = total.detach()
        self._totals = totals
        self._steps += 1
        return Prediction(
            logits=readout,
            spikes=spikes,
            class_totals=totals,
            steps=self._steps,
            label=int(totals.argmax(dim=-1).reshape(-1)[0]),
        )

    def _spikes(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return this step's neuron-stage spikes, bounded to the neurons."""
        return {
            name: outputs[name].detach()
            for name in self._neurons
            if name in outputs
        }

    def _validate(self, raw: Mapping[str, Any]) -> None:
        """Raise when ``raw`` names stages this session's spec lacks."""
        if not raw:
            return
        present = set(raw) - set(_RESERVED)
        unknown = present - set(self._neurons)
        missing = set(self._neurons) - present
        if unknown or missing:
            raise StateError(
                "stage mismatch: unknown="
                + repr(sorted(unknown))
                + " missing="
                + repr(sorted(missing))
            )

    def _as_input(self, frame: Any) -> torch.Tensor:
        """Return ``frame`` as a tensor on the session's device."""
        if isinstance(frame, torch.Tensor):
            return frame.to(self._device)
        return torch.as_tensor(frame, dtype=torch.float32, device=self._device)


def _resolve_bundle(
    bundle: Union[DeploymentBundle, str], strict: bool
) -> DeploymentBundle:
    """Return ``bundle`` itself, or the bundle loaded from its path."""
    if isinstance(bundle, str):
        return DeploymentBundle.load(bundle, strict=strict)
    return bundle


def _encode_optional(value: Optional[torch.Tensor]) -> Any:
    """Return ``value`` encoded for JSON, or None when unset."""
    return None if value is None else tensor_codec.encode(value)


def _decode_optional(
    value: Any, device: torch.device
) -> Optional[torch.Tensor]:
    """Return the tensor encoded by :func:`_encode_optional`, or None."""
    if value is None:
        return None
    return tensor_codec.decode(value, device)

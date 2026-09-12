"""The shared per-step stage sweep used by the loop and by serving.

The temporal loop in :mod:`spikeforge.simulator.execution` and the stateful
:class:`~spikeforge.serving.session.InferenceSession` both advance a topology
one timestep at a time. They must not diverge, so the single body that
normalises a raw frame and calls the injected step function lives here and
nowhere else.

``step_fn`` is injectable so the compiled step path is preserved: the closed
loop passes its eager or
:class:`~spikeforge.simulator.compiled_step.CompiledStep` callable, while
serving leaves it unset and gets ``module.step``. ``kind`` is injectable so
the loop does not pay for a repeated spec lookup each step. ``post_step`` is
an opt-in transform of the returned ``(outputs, state)`` pair; leaving it
``None`` keeps the step byte-identical, and it is how fixed-point activation
and membrane quantization are simulated without teaching the core about any
scheme.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from spikeforge.simulator.frames import normalise_frame
from spikeforge.simulator.module_spec import spec_of
from spikeforge.topology.stage_module import StageModule

StageOutputs = Dict[str, torch.Tensor]
StateMap = Dict[str, Any]
StepResult = Tuple[StageOutputs, StateMap]
_StepFn = Callable[[torch.Tensor, Optional[StateMap]], StepResult]
#: A transform of one step's ``(outputs, state)`` pair, applied after the step.
PostStep = Callable[[StageOutputs, StateMap], StepResult]


def step_stages(
    module: StageModule,
    frame: torch.Tensor,
    state: Optional[StateMap] = None,
    *,
    step_fn: Optional[_StepFn] = None,
    kind: Optional[str] = None,
    post_step: Optional[PostStep] = None,
) -> StepResult:
    """Advance ``module`` by one timestep and return ``(outputs, state)``.

    ``frame`` is the raw ``[B, ...]`` input for this step; it is normalised to
    the input stage's layout exactly as the closed loop normalises it. Passing
    ``step_fn`` keeps the compiled path (the caller's callable is used instead
    of ``module.step``), and passing ``kind`` skips a repeated spec lookup.
    ``post_step``, when given, transforms the result (activation/membrane
    quantization); ``None`` returns the step untouched.
    """
    if kind is None:
        spec = spec_of(module)
        kind = spec.stage(spec.input).kind
    fn = module.step if step_fn is None else step_fn
    outputs, next_state = fn(normalise_frame(frame, kind), state)
    if post_step is None:
        return outputs, next_state
    return post_step(outputs, next_state)

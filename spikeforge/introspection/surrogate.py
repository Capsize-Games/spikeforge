"""Selectable surrogate gradients discovered from ``snntorch.surrogate``.

The installed module exposes surrogate gradients two ways: as
``torch.autograd.Function`` subclasses, and as factory functions that close
over their hyperparameters and return the ``spike_grad`` callable a snnTorch
neuron expects. Only the factories are selectable, and they are discovered by
inspecting the installed module, so this registry always matches what the
running snnTorch actually provides.

``custom_surrogate`` is excluded because it requires the caller to supply a
custom function. ``LSO`` is listed because upstream exposes it, but upstream's
wrapper calls ``StochasticSpikeOperator`` with the slope in the mean position,
so applying it raises ``TypeError``; that is reported by the curve resolver
rather than hidden here.
"""

import inspect
import types
from typing import Any, Callable, Dict, List, Tuple

import snntorch.surrogate as _surrogate
import torch

#: A callable in the ``spike_grad`` position: a tensor in, spikes out.
Surrogate = Callable[[torch.Tensor], torch.Tensor]

#: A zero-argument factory returning a :data:`Surrogate`.
Factory = Callable[[], Surrogate]

_MIN_POINTS = 2


def _is_selectable_factory(obj: Any) -> bool:
    """Return True when ``obj`` is a zero-argument snntorch factory."""
    if not isinstance(obj, types.FunctionType):
        return False
    if obj.__module__ != _surrogate.__name__:
        return False
    try:
        parameters = inspect.signature(obj).parameters.values()
    except (TypeError, ValueError):
        return False
    kinds = (
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    )
    return all(
        parameter.default is not inspect.Parameter.empty
        or parameter.kind in kinds
        for parameter in parameters
    )


def _discover_factories() -> Dict[str, Factory]:
    """Return the selectable factories keyed by their public name."""
    return {
        name: obj
        for name, obj in vars(_surrogate).items()
        if _is_selectable_factory(obj)
    }


_FACTORIES: Dict[str, Factory] = _discover_factories()


def list_surrogates() -> List[str]:
    """Return the selectable surrogate names in sorted order."""
    return sorted(_FACTORIES)


def resolve_surrogate(name: str) -> Surrogate:
    """Return the ``spike_grad`` callable registered under ``name``.

    Raises ``KeyError`` when ``name`` is not a selectable surrogate, so a
    caller (and the future UI) never silently falls back to a default.
    """
    if name not in _FACTORIES:
        raise KeyError(f"unknown surrogate: {name!r}")
    return _FACTORIES[name]()


def _sample_gradient(
    spike_grad: Surrogate, x_min: float, x_max: float, count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the sampled coordinates and derivative of ``spike_grad``."""
    x = torch.linspace(float(x_min), float(x_max), count)
    x.requires_grad_(True)
    spike_grad(x).sum().backward()
    coordinates = x.detach().reshape(-1)
    derivative = x.grad.detach().reshape(-1)
    return coordinates, derivative


def surrogate_curve(
    name: str, x_min: float, x_max: float, points: int = 101
) -> Dict[str, Any]:
    """Return the surrogate derivative sampled over ``[x_min, x_max]``.

    ``y`` is the backward-pass derivative ``dS/dU`` at each ``x``; the
    result holds parallel ``x``/``y`` lists plus ``name``, JSON-serialisable
    for a later chart.
    """
    count = int(points)
    if count < _MIN_POINTS:
        raise ValueError("points must be at least 2")
    coordinates, derivative = _sample_gradient(
        resolve_surrogate(name), x_min, x_max, count
    )
    return {
        "name": name,
        "x": coordinates.tolist(),
        "y": derivative.tolist(),
    }

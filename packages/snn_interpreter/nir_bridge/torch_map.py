"""Map common ``torch.nn`` modules to NIR nodes for ``nirtorch`` extraction.

Only modules with a faithful NIR primitive are listed: a ``nn.Linear`` becomes
``nir.Affine`` when it carries a bias and ``nir.Linear`` when it does not, and
``nn.Flatten`` becomes ``nir.Flatten``. Every other module type is deliberately
absent, so ``nirtorch`` reports it rather than an approximation being guessed.
NIR primitives are resolved through :mod:`snn_interpreter.nir_bridge.require`,
keeping the direct ``nir`` import inside :mod:`snn_interpreter.nir_bridge.api`.
"""

from typing import Any, Callable, Dict

import torch

from snn_interpreter.nir_bridge.require import require_node


def _numpy(tensor: Any) -> Any:
    """Detach a torch tensor into a numpy array."""
    return tensor.detach().cpu().numpy()


def linear_node(module: torch.nn.Linear) -> Any:
    """Return the ``Affine``/``Linear`` node rendering a ``nn.Linear``."""
    weight = _numpy(module.weight)
    bias = module.bias
    if bias is None:
        return require_node("Linear", "linear")(weight)
    return require_node("Affine", "linear")(weight, _numpy(bias))


def flatten_node(module: torch.nn.Flatten) -> Any:
    """Return the ``Flatten`` node rendering a ``nn.Flatten``."""
    cls = require_node("Flatten", "flatten")
    return cls({"input": None}, int(module.start_dim), int(module.end_dim))


#: Torch module type -> the NIR node builder that faithfully renders it.
NODE_MAP: Dict[type, Callable[[Any], Any]] = {
    torch.nn.Linear: linear_node,
    torch.nn.Flatten: flatten_node,
}

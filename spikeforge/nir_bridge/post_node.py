"""The per-node transform :class:`NirInterpreter` optionally applies.

A ``PostNode`` is handed every evaluated node's name, kind, output, carried
state, and recorded membrane, and returns the output, state and membrane the
interpreter then stores. It is the single seam through which a caller can
model something the graph itself cannot express -- fixed-point activation and
membrane quantization is the shipped user -- without teaching the interpreter
about any scheme. Returning the arguments unchanged is exactly the default
behaviour, so a hook is never required to understand every node kind.

The membrane is ``None`` on a node that exposes none, and a hook must pass
that ``None`` through rather than substituting a tensor: the interpreter uses
it to decide whether the node has a membrane trace at all.
"""

from typing import Any, Callable, Optional, Tuple

import torch

#: ``(name, kind, output, state, membrane)`` in, ``(output, state, membrane)``
#: out, applied before any of the three is stored.
PostNode = Callable[
    [str, str, torch.Tensor, Any, Optional[torch.Tensor]],
    Tuple[torch.Tensor, Any, Optional[torch.Tensor]],
]

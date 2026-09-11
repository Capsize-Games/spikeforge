"""Optional ``torch.compile`` wrapping of a module's per-step function.

Compilation is strictly opt-in. :class:`CompiledStep` wraps a built topology
module's ``step`` and, when compilation is requested, hands it to
``torch.compile``. If no compiler is available, the compile call raises, or
the compiled graph raises on its first invocation, the wrapper degrades to
the eager ``step`` and records that decision in :attr:`CompiledStep.status`.
The caller can therefore always observe whether a run was actually compiled.

Nothing here mutates global state or monkeypatches the module: when a compiled
path is unusable the wrapper simply calls the original bound method.
"""

from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from spikeforge.simulator.module_spec import spec_of
from spikeforge.topology.spec import TopologySpec

#: Sentinel meaning "resolve ``torch.compile`` from the installed torch".
AUTO_COMPILER: Any = object()

#: A compiler maps a callable to a (potentially) compiled callable.
CompileFn = Callable[[Callable[..., Any]], Callable[..., Any]]

#: Documented :attr:`CompiledStep.status` values.
STATUS_EAGER = "eager"
STATUS_UNAVAILABLE = "unavailable"
STATUS_COMPILED = "compiled"
STATUS_FALLBACK = "fallback"


def compiled_step_available() -> bool:
    """Return True when this torch build exposes ``torch.compile``."""
    return callable(getattr(torch, "compile", None))


def _select_compiler(compiler: Any) -> Optional[CompileFn]:
    """Resolve ``compiler`` to a compile callable, or None when unavailable."""
    if compiler is AUTO_COMPILER:
        candidate = getattr(torch, "compile", None)
        return candidate if callable(candidate) else None
    return compiler


class CompiledStep:
    """A module's ``step``, optionally compiled, with an eager fallback.

    ``status`` is one of ``"eager"`` (compilation was not requested),
    ``"unavailable"`` (no compiler could be resolved), ``"compiled"`` (a
    compiled callable is in use), or ``"fallback"`` (compilation, or its first
    invocation, failed so the eager path is used). ``compiled`` is True only
    while ``status`` is ``"compiled"``.

    ``compiler`` is resolved dynamically by default (``AUTO_COMPILER``); pass
    an explicit callable to choose a backend or to inject a double, or pass
    ``None`` to force the unavailable/eager path.
    """

    def __init__(
        self,
        module: Any,
        enabled: bool = True,
        compiler: Any = AUTO_COMPILER,
    ) -> None:
        """Wrap ``module.step``, compiling it when ``enabled`` and possible."""
        self._spec: TopologySpec = spec_of(module)
        self._eager: Callable[..., Any] = module.step
        self._compiled: Optional[Callable[..., Any]] = None
        self._verified = False
        self._status = STATUS_EAGER
        if enabled:
            self._attempt(module, compiler)

    def _attempt(self, module: Any, compiler: Any) -> None:
        """Compile ``module.step``, recording the resulting status."""
        compile_fn = _select_compiler(compiler)
        if compile_fn is None:
            self._status = STATUS_UNAVAILABLE
            return
        try:
            self._compiled = compile_fn(module.step)
        except Exception:
            self._compiled = None
            self._status = STATUS_FALLBACK
            return
        self._status = STATUS_COMPILED

    @property
    def spec(self) -> TopologySpec:
        """Return the topology spec of the wrapped module."""
        return self._spec

    @property
    def status(self) -> str:
        """Return the documented compilation status string."""
        return self._status

    @property
    def compiled(self) -> bool:
        """Return True while a compiled callable is actively in use."""
        return self._status == STATUS_COMPILED

    def step(
        self, x: torch.Tensor, state: Optional[Mapping[str, Any]] = None
    ) -> Tuple[Any, Any]:
        """Advance one step through the compiled path, else the eager path."""
        if not self.compiled:
            return self._eager(x, state)
        if self._verified:
            return self._compiled(x, state)
        return self._first(x, state)

    def _first(
        self, x: torch.Tensor, state: Optional[Mapping[str, Any]]
    ) -> Tuple[Any, Any]:
        """Run the first compiled step, falling back to eager if it raises."""
        try:
            result = self._compiled(x, state)
        except Exception:
            self._compiled = None
            self._status = STATUS_FALLBACK
            return self._eager(x, state)
        self._verified = True
        return result

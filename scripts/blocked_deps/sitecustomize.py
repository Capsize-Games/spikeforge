"""CI-only import blocker for the optional extras.

Loaded automatically by the ``site`` module when this directory is first on
``PYTHONPATH`` (see the ``blocked-deps`` job in ``.github/workflows/ci.yml``).
It makes the listed optional packages unimportable so the test suite can prove
graceful degradation *even when the extras are installed in the environment*,
without uninstalling anything.

This file is CI scaffolding, not product code: nothing in ``spikeforge``
or ``server`` imports it.

``find_spec`` deliberately returns a spec (rather than raising) so that
existence probes such as ``torch._dynamo.trace_rules`` behave as they would
when the package is genuinely absent; only an actual import fails.
"""

import importlib.machinery
import sys

#: Optional dependencies whose absence must be handled honestly.
#:
#: The ``fastapi``/``pydantic``/``uvicorn`` entries are the ARCH-0001 Phase 1c
#: addition: they are the base dependencies of the ``spikeforge-server``
#: distribution ([`plans/arch-0001-core-boundary.md`]), so blocking them proves
#: that a headless core install does not reach for the server stack. The
#: remaining entries are the pre-existing data/event, tracking, hub, and
#: deploy-backend SDKs. ``lava-nc`` is the distribution name; the import
#: root it shadows here is ``lava``, both of which are listed.
BLOCKED = frozenset(
    {
        # server stack — must be absent from a headless core install
        "fastapi",
        "pydantic",
        "uvicorn",
        # data/event SDKs
        "onnx",
        "onnxruntime",
        "onnxscript",
        "tonic",
        # logging/tracking SDKs
        "tensorboard",
        "wandb",
        # model hub
        "huggingface_hub",
        # deploy backends
        "norse",
        "lava",
        "lava-nc",
        "spinnaker2",
        "sinabs",
        "rockpool",
    }
)


class _MissingLoader:
    """A loader that reports the blocked module as genuinely missing."""

    def create_module(self, spec: object) -> object:
        """Refuse to create the blocked module."""
        raise ModuleNotFoundError(
            "blocked optional dependency for verification: "
            f"{getattr(spec, 'name', spec)}"
        )

    def exec_module(self, module: object) -> None:
        """Refuse to execute the blocked module."""
        raise ModuleNotFoundError(
            "blocked optional dependency for verification: "
            f"{getattr(module, '__name__', module)}"
        )


class _Blocker:
    """A meta-path finder that reports blocked packages as missing."""

    def find_spec(
        self, name: str, path: object = None, target: object = None
    ) -> object:
        """Return a missing-module spec for a blocked top-level package."""
        if name.split(".")[0] in BLOCKED:
            return importlib.machinery.ModuleSpec(name, _MissingLoader())
        return None


sys.meta_path.insert(0, _Blocker())

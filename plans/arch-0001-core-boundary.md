# ARCH-0001 core boundary

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** w4ffl35 (maintainer) · **Depends on:** [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)

## Decision

The `snn-interpreter` distribution MUST install and import **without any of the
following distributions present**. This list is the enforced core boundary and is
frozen for `protocol_version` 1.0.

```text
fastapi
pydantic
uvicorn
huggingface_hub
nir
nirtorch
onnx
onnxruntime
norse
lava
lava-nc
tonic
tensorboard
wandb
```

The list mixes distribution names and import roots deliberately. Where they
differ: distribution `lava-nc` is imported as `lava`; `huggingface_hub`,
`pydantic`, `fastapi`, `uvicorn`, `nir`, `nirtorch`, `onnx`, `onnxruntime`,
`tonic`, `tensorboard`, and `wandb` use the same token for both.

## The boundary rule

Three tiers define what "forbidden" means. A core module violates the boundary
if it breaks tier 1 or tier 2; tier 3 is explicitly permitted.

1. **Tier 1 — never a required dependency.** None of the forbidden distributions
   may appear in the core distribution's `[project].dependencies`. They may
   appear only in the core's optional extras
   ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)).
2. **Tier 2 — never imported at import time.** No core module may perform a
   top-level `import fastapi`, `from pydantic import ...`, and so on. A core
   module that needs an optional SDK must import it *inside a function*, through
   one of the existing lazy shims, and degrade honestly when it is absent.
3. **Tier 3 — lazy access is allowed.** The established shims
   (`nir_bridge/api.py`, `onnx_bridge/api.py`, `targets/backends/api.py`,
   `events/tonic_api.py`, `hub/probe.py`, `targets/probe.py`,
   `tracking/tensorboard_sink.py`, `tracking/wandb_sink.py`) may import a
   forbidden SDK inside a function and report availability rather than raise.
   These modules are the only sanctioned exception sites and are enumerated in
   the static scan's allow-list.

The server distribution is exempt by construction: `fastapi`, `pydantic`, and
`uvicorn` are its *base* dependencies
([`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)).
That is why Phase 1 removes `server/` from the core distribution — today it is
packaged into core by [`setup.py`](setup.py:37), which is the single largest
boundary violation in the tree.

## CI enforcement

Enforcement has two independent mechanisms; both must pass.

### 1. Extend the blocked-deps runtime gate

The existing blocker at
[`scripts/blocked_deps/sitecustomize.py`](scripts/blocked_deps/sitecustomize.py:21)
makes a listed package unimportable at runtime so graceful-degradation paths are
exercised even when the optional extras are installed. Its `BLOCKED` set today
covers the SDKs but **not** `fastapi`, `pydantic`, or `uvicorn`, which were only
ever exercised because `server/` shipped in the same distribution. Extend it:

```python
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
```

The `blocked-deps` CI job then proves every `available: false` and typed
unavailable path — including "the server stack is not installed" — in one run.

### 2. Static import scan

Add `scripts/check_core_boundary.py`, run in a new `core-boundary` job. It walks
the core distribution's packages (everything in `include = ["snn_interpreter*"]`,
`exclude = ["server*", "snn_targets*", "snn_hub*"]`), parses each module's AST,
and fails if any *module-level* import names a forbidden root. Function-local
imports are permitted only inside the enumerated shim allow-list; a
function-local import of a forbidden SDK anywhere else is a failure too, so the
boundary cannot be widened silently.

```text
check_core_boundary.py
  forbidden = {fastapi, pydantic, uvicorn, huggingface_hub, nir, nirtorch,
               onnx, onnxruntime, norse, lava, tonic, tensorboard, wandb}
  scan roots = snn_interpreter/**  and  main.py, main_encodings.py
  rule       = no module-level import of a forbidden root, anywhere
  rule       = no function-level import of a forbidden root outside
               {"snn_interpreter.nir_bridge.api",
                "snn_interpreter.onnx_bridge.api",
                "snn_interpreter.targets.backends.api",
                "snn_interpreter.events.tonic_api",
                "snn_interpreter.hub.probe",
                "snn_interpreter.targets.probe",
                "snn_interpreter.tracking.tensorboard_sink",
                "snn_interpreter.tracking.wandb_sink"}
```

### 3. Headless install proof

Add a `headless` verification step that builds the distribution **with no
extras** and asserts the boundary from the outside:

```bash
python -m build packages/snn-interpreter            # no extras requested
python -m venv /tmp/headless && . /tmp/headless/bin/activate
pip install dist/snn_interpreter-*.whl
python - <<'PY'
import importlib.util as u
for mod in ("fastapi", "pydantic", "uvicorn", "huggingface_hub",
            "nir", "nirtorch", "onnx", "onnxruntime",
            "norse", "lava", "tonic", "tensorboard", "wandb"):
    assert u.find_spec(mod) is None, f"core install leaked forbidden dep: {mod}"
import snn_interpreter  # must import cleanly with the forbidden set absent
PY
```

The wheel must not contain `server/` either, verified by listing the archive and
asserting no `server/` entry is present.

## Corrections and findings

- **The current `blocked-deps` gate has a blind spot.** `sitecustomize.py`
  blocks the SDKs but not `fastapi`/`pydantic`/`uvicorn`, so today the gate would
  not catch a core-code import of the server stack even if `server/` were
  removed. Extending `BLOCKED` as above is part of this decision.
- **The `web` extra is the wrong home for the server stack.** `setup.py` §extras
  declares `web = [fastapi, uvicorn, websockets, pydantic]`. Phase 1 promotes
  those to the base dependencies of the `snn-interpreter-server` distribution
  and removes `web` from core
  ([`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md)).
- **`nir` and `huggingface_hub` remain core-optional, not core-required.** The
  `nir_bridge`, `hub`, and `targets` subpackages stay in the core distribution
  until Phases 3–4; they satisfy the boundary because their SDK imports are
  lazy. They are forbidden only as *required* deps.

## Related decisions

- [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) — which distribution owns which root.
- [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md) — extras placement that keeps core headless.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — when the server leaves core.
- [`plans/arch-0001-risk-register.md`](plans/arch-0001-risk-register.md) — the boundary-drift risk and its owner.

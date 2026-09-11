# ARCH-0001 packaging and versioning

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** w4ffl35 (maintainer) · **Depends on:** [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)

## Decision

Adopt PEP 621 [`pyproject.toml`](packages/snn-interpreter/pyproject.toml) as the
single packaging authority for **each** distribution, one file per distribution
under the `packages/` workspace. The monolithic
[`setup.py`](setup.py:22) stops being the authority; it is retired after the
`tests/test_packaging_profiles.py` assertions are ported to read the new files.

The distribution names and import roots are fixed by
[`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md):
`snn-interpreter` / `snn_interpreter`, `snn-interpreter-server` / `server`,
`snn-targets` / `snn_targets`, `snn-hub` / `snn_hub`, and the private npm
package `snn-interpreter-client` (product/repo `snn-dashboard`).

## Extras-to-package mapping

`web` is **removed from core** and becomes the base dependencies of the server
distribution. The `hub`, `norse`, and `lava` extras remain on core through
Phases 1–2 and migrate to the distributions that own their code.

| Extra (today) | Phase 1 owner | After extraction | Notes |
|---|---|---|---|
| `dev` | `snn-interpreter` | `snn-interpreter` | pytest, pytest-cov, ruff |
| `nir` | `snn-interpreter` | `snn-interpreter` | `nir_bridge` stays in core |
| `events` | `snn-interpreter` | `snn-interpreter` | `events/` stays in core; `event_runtime/` moves Phase 3 |
| `onnx` | `snn-interpreter` | `snn-interpreter` | `onnx_bridge` stays in core, lazy |
| `tracking` | `snn-interpreter` | `snn-interpreter` | `tracking/` sinks stay in core, lazy |
| `tracking-wandb` | `snn-interpreter` | `snn-interpreter` | lazy sink |
| `docs` | `snn-interpreter` | `snn-interpreter` | mkdocs-material |
| `web` | **removed** | — | becomes `snn-interpreter-server` base deps |
| `hub` | `snn-interpreter` | `snn-hub` base dep | extra removed once `hub/` leaves |
| `norse` | `snn-interpreter` | `snn-targets` extra | extra removed once `targets/backends/` leaves |
| `lava` | `snn-interpreter` | `snn-targets` extra | extra removed once `targets/backends/` leaves |

### Core dependencies (unchanged requirement set)

`torch>=2.5`, `torchvision>=0.20`, `snntorch>=1.0`, `matplotlib>=3.8`,
`Pillow>=10.0`, `numpy>=1.26`, `psutil>=5.9` — exactly the current
`install_requires` in [`setup.py`](setup.py:40). None of the core-boundary
forbidden list ([`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md))
appears here, which is what makes a headless install provable.

### Server dependencies

`fastapi>=0.110`, `uvicorn[standard]>=0.27`, `websockets>=12.0`,
`pydantic>=2.5` (the current `web` extra, promoted to base), plus a pin on the
core distribution and, after Phases 3–4, pins on `snn-targets` and `snn-hub`
because 24 server files import those capability roots today.

## Console-script ownership

The table below assigns every script in
[`setup.py`](setup.py:91) to the distribution that must install it. "Today" is
the current entry point; "After extraction" is the intended steady state, where
the script survives under the same name but resolves inside the owning root.

| Console script | End-state owner | Entry point today | Entry point after extraction |
|---|---|---|---|
| `snn-interpreter` | `snn-interpreter` | `main:main` | unchanged |
| `snn-interpreter-encodings` | `snn-interpreter` | `main_encodings:main` | unchanged |
| `snn-verify` | `snn-interpreter` | `snn_interpreter.cli.verify:main` | unchanged |
| `snn-records` | `snn-interpreter` | `snn_interpreter.cli.records_cli:main` | unchanged |
| `snn-benchmark` | `snn-interpreter` | `snn_interpreter.benchmark.cli:main` | unchanged |
| `snn-energy` | `snn-targets` | `snn_interpreter.energy.cli:main` | `snn_targets.energy.cli:main` |
| `snn-targets` | `snn-targets` | `snn_interpreter.cli.target_cli:main` | `snn_targets.cli.target_cli:main` |
| `snn-hub` | `snn-hub` | `snn_interpreter.hub.cli:main` | `snn_hub.cli:main` |

**Transitional rule.** Until the phase that extracts a script's owner fires, the
script stays on `snn-interpreter`. `snn-energy`, `snn-targets`, and `snn-hub`
therefore remain core console scripts through Phase 1. Ownership transfers at
Phase 3 (`snn-energy`, `snn-targets`) and Phase 4 (`snn-hub`) respectively. No
script name is ever owned by two distributions at once.

## Versioning

- **Per-distribution versions.** Each `pyproject.toml` carries its own
  `version`. `snn-interpreter` continues its `0.2.0` line; the Phase 1 boundary
  and protocol work is a minor bump to `snn-interpreter 0.3.0`. New
  distributions start at `0.1.0` (`snn-interpreter-server 0.1.0`, then
  `snn-targets 0.1.0`, `snn-hub 0.1.0`).
- **Protocol version is independent.** `protocol_version` is `"1.0"`
  ([`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md))
  and does not track any package version. A package release may ship without a
  protocol change; a protocol change requires its own MINOR/MAJOR rule.
- **Cross-repo pinning uses compatible-release on a minor.** Satellites pin core
  with `snn-interpreter~=X.Y.0` (for `0.x`, `~=0.3.0` means `>=0.3.0,<0.4.0`)
  because a pre-1.0 minor may be breaking. Core never pins a satellite; the
  server and monorepo dev environment pin satellites.
- **Compatibility matrix.** A repo-root `compatibility.json` records, per
  release tag, the exact triples (satellite version, core version,
  `protocol_version`) and the dashboard bundle version. CI asserts every
  satellite's declared pin equals the matrix entry, and the release workflow
  refuses to publish a satellite whose core pin is not in the matrix.

```json
{
  "protocol_version": "1.0",
  "releases": [
    { "snn-interpreter": "0.3.0", "snn-interpreter-server": "0.1.0",
      "snn-targets": null, "snn-hub": null, "dashboard": "0.1.0" }
  ]
}
```

## Release automation

- **Tag scheme.** `<distribution>-v<version>`, for example
  `snn-interpreter-v0.3.0` or `snn-interpreter-server-v0.1.0`. Tags are the only
  release trigger.
- **Workflow.** A single `.github/workflows/release.yml` parses the tag prefix,
  selects `packages/<distribution>`, runs `python -m build`, and publishes to
  PyPI with OIDC trusted publishing (no long-lived tokens). One job definition,
  many distributions, so the per-repo CI overhead concern in
  [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) §5 stays bounded.
- **Dashboard.** The npm package is `private: true`, so there is no npm publish.
  A dashboard "release" builds the Vite bundle and publishes a versioned build
  artifact; `snn-interpreter-server` consumes a pinned bundle version from
  `compatibility.json`. After the Phase 2 extraction this artifact is built in
  `w4ffl35/snn-dashboard`, not here.
- **Ordering.** Core tags first; satellites pin the just-released core; the
  protocol authority (`protocol/`) is published only as part of a core release.
  A protocol `protocol_version` bump is always accompanied by a server release
  and a dashboard bundle release, in that order.

## How a headless install avoids the forbidden deps

The proof is external, not asserted: the `headless` CI step builds
`packages/snn-interpreter` with **no extras**, installs the wheel into a clean
venv, asserts `importlib.util.find_spec` is `None` for all thirteen forbidden
distributions, imports `snn_interpreter`, and asserts the wheel contains no
`server/` entries. The runtime equivalent is the extended `blocked-deps` gate.
Both are specified in
[`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md). The two
mechanisms are complementary: the wheel test proves the artifact is clean; the
blocker proves the code degrades honestly when the SDKs vanish.

## How the monorepo keeps working after each extraction

Extraction is always **additive first, subtractive second**: the new repository
is created and pushed from history before the code is deleted here. After the
subtraction the monorepo still works because of two mechanisms.

1. **Pinned dependency.** The core repo's development requirements add the
   satellite at the exact version recorded in `compatibility.json` (for example
   `snn-targets==0.1.0`), so the dev venv and CI have the moved code available.
   The published core distribution does **not** depend on the satellite.
2. **Thin lazy re-export shim.** Legacy import paths are kept working by a small
   forwarding module in core that imports the satellite lazily and degrades:

   ```python
   # snn_interpreter/targets/__init__.py  (thin shim, post-Phase-3)
   import warnings
   warnings.warn(
       "snn_interpreter.targets moved to snn_targets; install 'snn-targets'.",
       DeprecationWarning,
       stacklevel=2,
   )
   from snn_targets import *          # raises a clear ImportError if absent
   ```

   If `snn-targets` is not installed the shim raises `ImportError` naming the
   migration target rather than `ModuleNotFoundError` about an unrelated SDK.
   Because the satellite is not on the forbidden list, shipping the shim in the
   core wheel does not violate the boundary.

### Extraction procedure

Use `git subtree` for single-directory extractions and `git filter-repo` for the
multi-directory `snn-targets` extraction, so history is preserved.

```bash
# Phase 3 — snn-targets absorbs four prefixes (renamed to the new root).
git filter-repo \
  --path snn_interpreter/targets \
  --path snn_interpreter/energy \
  --path snn_interpreter/event_runtime \
  --path-rename snn_interpreter/:snn_targets/ \
  --force
# then: git remote add origin git@github.com:w4ffl35/snn-targets.git && git push -u origin main

# Phase 4 — snn-hub and (conditionally) snn-server are single-directory splits.
git subtree split --prefix=snn_interpreter/hub -b snn-hub-split
git subtree split --prefix=server -b snn-server-split

# Phase 2 — the dashboard is a single-directory split.
git subtree split --prefix=client -b snn-dashboard-split
```

Because `git filter-repo` rewrites history in the working clone, it is run on a
**throwaway clone** of the core repo; the extraction PR only removes the moved
paths and adds the pin + shim. The pre-extraction commit remains in core's
history as the rollback point
([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)).

## Corrections and findings

- **There is no `pyproject.toml` and no `setup.cfg` today.** Packaging is
  defined only in [`setup.py`](setup.py:22). Introducing PEP 621 per
  distribution is net-new, and `tests/test_packaging_profiles.py` must be
  ported from reading `setup.py` text to reading the new files.
- **`find_packages` leaks `server/` into core today** — see
  [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) for
  the exclusion that fixes it.

## Related decisions

- [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) — distributions and import roots.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — the forbidden list the headless proof checks.
- [`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md) — the independent `protocol_version`.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — the ordered extraction steps.

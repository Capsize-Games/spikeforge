# ARCH-0001 packaging and versioning

**Status: accepted (proposed for maintainer sign-off).**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)

## Decision

Adopt PEP 621 [`pyproject.toml`](packages/spikeforge/pyproject.toml) as the
single packaging authority for **each** distribution, one file per distribution
under the `packages/` workspace. The monolithic
[`setup.py`](setup.py:22) stops being the authority; it is retired after the
`tests/test_packaging_profiles.py` assertions are ported to read the new files.

The distribution names and import roots are fixed by
[`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md):
`spikeforge` / `spikeforge`, `spikeforge-server` / `server`,
`spikeforge-targets` / `spikeforge_targets`, `spikeforge-hub` / `spikeforge_hub`, and the private npm
package `spikeforge-dashboard` (product/repo `spikeforge-dashboard`).

## Extras-to-package mapping

`web` is **removed from core** and becomes the base dependencies of the server
distribution. The `hub`, `norse`, and `lava` extras remain on core through
Phases 1–2 and migrate to the distributions that own their code.

| Extra (today) | Phase 1 owner | After extraction | Notes |
|---|---|---|---|
| `dev` | `spikeforge` | `spikeforge` | pytest, pytest-cov, ruff |
| `nir` | `spikeforge` | `spikeforge` | `nir_bridge` stays in core |
| `events` | `spikeforge` | `spikeforge` | `events/` stays in core; `event_runtime/` moves Phase 3 |
| `onnx` | `spikeforge` | `spikeforge` | `onnx_bridge` stays in core, lazy |
| `tracking` | `spikeforge` | `spikeforge` | `tracking/` sinks stay in core, lazy |
| `tracking-wandb` | `spikeforge` | `spikeforge` | lazy sink |
| `docs` | `spikeforge` | `spikeforge` | mkdocs-material |
| `web` | **removed** | — | becomes `spikeforge-server` base deps |
| `hub` | `spikeforge` | `spikeforge-hub` base dep | extra removed once `hub/` leaves |
| `norse` | `spikeforge` | `spikeforge-targets` extra | extra removed once `targets/backends/` leaves |
| `lava` | `spikeforge` | `spikeforge-targets` extra | extra removed once `targets/backends/` leaves |

### Core dependencies (unchanged requirement set)

`torch>=2.5`, `torchvision>=0.20`, `snntorch>=1.0`, `matplotlib>=3.8`,
`Pillow>=10.0`, `numpy>=1.26`, `psutil>=5.9` — exactly the current
`install_requires` in [`setup.py`](setup.py:40). None of the core-boundary
forbidden list ([`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md))
appears here, which is what makes a headless install provable.

### Server dependencies

`fastapi>=0.110`, `uvicorn[standard]>=0.27`, `websockets>=12.0`,
`pydantic>=2.5` (the current `web` extra, promoted to base), plus a pin on the
core distribution and, after Phases 3–4, pins on `spikeforge-targets` and `spikeforge-hub`
because 24 server files import those capability roots today.

## Console-script ownership

The table below assigns every script in
[`setup.py`](setup.py:91) to the distribution that must install it. "Today" is
the current entry point; "After extraction" is the intended steady state, where
the script survives under the same name but resolves inside the owning root.

| Console script | End-state owner | Entry point today | Entry point after extraction |
|---|---|---|---|
| `spikeforge` | `spikeforge` | `main:main` | unchanged |
| `spikeforge-encodings` | `spikeforge` | `main_encodings:main` | unchanged |
| `spikeforge-verify` | `spikeforge` | `spikeforge.cli.verify:main` | unchanged |
| `spikeforge-records` | `spikeforge` | `spikeforge.cli.records_cli:main` | unchanged |
| `spikeforge-benchmark` | `spikeforge` | `spikeforge.benchmark.cli:main` | unchanged |
| `spikeforge-energy` | `spikeforge-targets` | `spikeforge.energy.cli:main` | `spikeforge_targets.energy.cli:main` |
| `spikeforge-targets` | `spikeforge-targets` | `spikeforge.cli.target_cli:main` | `spikeforge_targets.cli.target_cli:main` |
| `spikeforge-hub` | `spikeforge-hub` | `spikeforge.hub.cli:main` | `spikeforge_hub.cli:main` |

**Transitional rule.** Until the phase that extracts a script's owner fires, the
script stays on `spikeforge`. `spikeforge-energy`, `spikeforge-targets`, and `spikeforge-hub`
therefore remain core console scripts through Phase 1. Ownership transfers at
Phase 3 (`spikeforge-energy`, `spikeforge-targets`) and Phase 4 (`spikeforge-hub`) respectively. No
script name is ever owned by two distributions at once.

## Versioning

- **Per-distribution versions.** Each `pyproject.toml` carries its own
  `version`. `spikeforge` continues its `0.2.0` line; the Phase 1 boundary
  and protocol work is a minor bump to `spikeforge 0.3.0`. New
  distributions start at `0.1.0` (`spikeforge-server 0.1.0`, then
  `spikeforge-targets 0.1.0`, `spikeforge-hub 0.1.0`).
- **Protocol version is independent.** `protocol_version` is `"1.0"`
  ([`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md))
  and does not track any package version. A package release may ship without a
  protocol change; a protocol change requires its own MINOR/MAJOR rule.
- **Cross-repo pinning uses compatible-release on a minor.** Satellites pin core
  with `spikeforge~=X.Y.0` (for `0.x`, `~=0.3.0` means `>=0.3.0,<0.4.0`)
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
    { "spikeforge": "0.3.0", "spikeforge-server": "0.1.0",
      "spikeforge-targets": null, "spikeforge-hub": null, "dashboard": "0.1.0" }
  ]
}
```

## Release automation

- **Tag scheme.** `<distribution>-v<version>`, for example
  `spikeforge-v0.3.0` or `spikeforge-server-v0.1.0`. Tags are the only
  release trigger.
- **Workflow.** A single `.github/workflows/release.yml` parses the tag prefix,
  selects `packages/<distribution>`, runs `python -m build`, and publishes to
  PyPI with OIDC trusted publishing (no long-lived tokens). One job definition,
  many distributions, so the per-repo CI overhead concern in
  [`plans/repo_topology_plan.md`](plans/repo_topology_plan.md) §5 stays bounded.
- **Dashboard.** The npm package is `private: true`, so there is no npm publish.
  A dashboard "release" builds the Vite bundle and publishes a versioned build
  artifact; `spikeforge-server` consumes a pinned bundle version from
  `compatibility.json`. After the Phase 2 extraction this artifact is built in
  `capsize-games/spikeforge-dashboard`, not here.
- **Ordering.** Core tags first; satellites pin the just-released core; the
  protocol authority (`protocol/`) is published only as part of a core release.
  A protocol `protocol_version` bump is always accompanied by a server release
  and a dashboard bundle release, in that order.

## How a headless install avoids the forbidden deps

The proof is external, not asserted: the `headless` CI step builds
`packages/spikeforge` with **no extras**, installs the wheel into a clean
venv, asserts `importlib.util.find_spec` is `None` for all thirteen forbidden
distributions, imports `spikeforge`, and asserts the wheel contains no
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
   `spikeforge-targets==0.1.0`), so the dev venv and CI have the moved code available.
   The published core distribution does **not** depend on the satellite.
2. **Thin lazy re-export shim.** Legacy import paths are kept working by a small
   forwarding module in core that imports the satellite lazily and degrades:

   ```python
   # spikeforge/targets/__init__.py  (thin shim, post-Phase-3)
   import warnings
   warnings.warn(
       "spikeforge.targets moved to spikeforge_targets; install 'spikeforge-targets'.",
       DeprecationWarning,
       stacklevel=2,
   )
   from spikeforge_targets import *          # raises a clear ImportError if absent
   ```

   If `spikeforge-targets` is not installed the shim raises `ImportError` naming the
   migration target rather than `ModuleNotFoundError` about an unrelated SDK.
   Because the satellite is not on the forbidden list, shipping the shim in the
   core wheel does not violate the boundary.

### Extraction procedure

Use `git subtree` for single-directory extractions and `git filter-repo` for the
multi-directory `spikeforge-targets` extraction, so history is preserved.

```bash
# Phase 3 — spikeforge-targets absorbs four prefixes (renamed to the new root).
git filter-repo \
  --path spikeforge/targets \
  --path spikeforge/energy \
  --path spikeforge/event_runtime \
  --path-rename spikeforge/:spikeforge_targets/ \
  --force
# then: git remote add origin git@github.com:capsize-games/spikeforge-targets.git && git push -u origin main

# Phase 4 — spikeforge-hub and (conditionally) spikeforge-server are single-directory splits.
git subtree split --prefix=spikeforge/hub -b spikeforge-hub-split
git subtree split --prefix=server -b spikeforge-server-split

# Phase 2 — the dashboard is a single-directory split.
git subtree split --prefix=client -b spikeforge-dashboard-split
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

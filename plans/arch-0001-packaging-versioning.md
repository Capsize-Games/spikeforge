# ARCH-0001 packaging and versioning

**Status: accepted — implemented.**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)

## Decision

Adopt PEP 621 [`pyproject.toml`](packages/spikeforge/pyproject.toml) as the
single packaging authority for **each** distribution, one file per distribution
under the `packages/` workspace. The monolithic `setup.py` has been retired;
`tests/test_packaging_profiles.py` now reads the new files.

The **four** distributions and their import roots are fixed by
[`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md):
`spikeforge` / `spikeforge`, `spikeforge-server` / `server`,
`spikeforge-targets` / `spikeforge_targets`, `spikeforge-hub` / `spikeforge_hub`. The private npm
package `spikeforge-dashboard` lives in
[`capsize-games/spikeforge-dashboard`](https://github.com/capsize-games/spikeforge-dashboard).

Install is one command from a clone — `./install.sh` installs all four
distributions editable (core + targets + hub + server) — and, once the
distributions are published, `pip install "spikeforge[all]"` (library bundle:
core + targets + hub) or `pip install spikeforge-server` (server).

## Extras-to-package mapping

`web` is **removed from core** and became the base dependencies of the server
distribution. The `hub`, `norse`, and `lava` extras migrated to the
distributions that own their code: the `hub` extra folded into `spikeforge-hub`'s
base dependencies, and `norse`/`lava` became `spikeforge-targets` extras. The
table records the pre-split mapping and the delivered end state.

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

Each console script is owned by exactly one distribution. The table records the
delivered end state.

| Console script | Owner | Entry point |
|---|---|---|
| `spikeforge` | `spikeforge` | `main:main` |
| `spikeforge-encodings` | `spikeforge` | `main_encodings:main` |
| `spikeforge-verify` | `spikeforge` | `spikeforge.cli.verify:main` |
| `spikeforge-records` | `spikeforge` | `spikeforge.cli.records_cli:main` |
| `spikeforge-benchmark` | `spikeforge` | `spikeforge.benchmark.cli:main` |
| `spikeforge-energy` | `spikeforge-targets` | `spikeforge_targets.energy.cli:main` |
| `spikeforge-targets` | `spikeforge-targets` | `spikeforge_targets.cli.target_cli:main` |
| `spikeforge-hub` | `spikeforge-hub` | `spikeforge_hub.cli:main` |
| `spikeforge-server` | `spikeforge-server` | `server.__main__:main` |

No script name is owned by two distributions at once.

## Versioning

- **Per-distribution versions.** Each `pyproject.toml` carries its own
  `version`. The delivered versions are `spikeforge 0.3.0`,
  `spikeforge-targets 0.1.0`, `spikeforge-hub 0.1.0`, and
  `spikeforge-server 0.1.0`. Satellites pin core with the compatible-release
  form `spikeforge~=0.3.0`; `spikeforge-server` additionally pins
  `spikeforge-targets~=0.1.0` and `spikeforge-hub~=0.1.0`, and the core `all`
  extra bundles `spikeforge-targets~=0.1.0` plus `spikeforge-hub~=0.1.0`.
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
      "spikeforge-targets": "0.1.0", "spikeforge-hub": "0.1.0", "dashboard": "0.1.0" }
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

Extraction was always **additive first, subtractive second**: the new repository
is created and pushed from history before the code is deleted here. After the
subtraction the monorepo still works because the core repo's development
requirements pin the satellite at the exact version recorded in
`compatibility.json` (for example `spikeforge-targets==0.1.0`), so the dev venv
and CI have the moved code available. The published core distribution does
**not** depend on the satellite.

### No back-compat aliases

Because the project is pre-1.0 and has never been published to a package index,
the extraction shipped **without** legacy import-path aliases. The old
`spikeforge.{targets,energy,event_runtime,hub}` paths were deleted outright;
importers use the owning root directly (`spikeforge_targets`,
`spikeforge_targets.energy`, `spikeforge_targets.event_runtime`,
`spikeforge_hub`). There is no shim, no `DeprecationWarning` window, and no
dual-path lifetime to retire later.

### Extraction procedure

`git subtree` was used for single-directory extractions and `git filter-repo`
for the multi-directory `spikeforge-targets` extraction, so history is
preserved.

```bash
# Phase 3 — spikeforge-targets absorbs four prefixes (renamed to the new root).
git filter-repo \
  --path spikeforge_targets \
  --path spikeforge_targets/energy \
  --path spikeforge_targets/event_runtime \
  --path-rename spikeforge/:spikeforge_targets/ \
  --force
# then: git remote add origin git@github.com:capsize-games/spikeforge-targets.git && git push -u origin main

# Phase 4 — spikeforge-hub is a single-directory split (the spikeforge-server
# split stays on the shelf because trigger T4 did not fire).
git subtree split --prefix=spikeforge_hub -b spikeforge-hub-split

# Phase 2 — the dashboard is a single-directory split.
git subtree split --prefix=client -b spikeforge-dashboard-split
```

The extractions ran on **throwaway clones** of the core repo; the extraction PRs
only removed the moved paths and added the pins. The pre-extraction commit
remains in core's history as the rollback point
([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)).

## Corrections and findings

- **PEP 621 packaging is now the authority.** The pre-split tree had no
  `pyproject.toml` and no `setup.cfg`; packaging lived only in `setup.py`.
  `setup.py` is now retired and `tests/test_packaging_profiles.py` reads the
  per-distribution `packages/*/pyproject.toml` files.
- **`server/` no longer leaks into core.** The core distribution's package
  discovery excludes `server/`, `tests/`, `spikeforge_targets/`, and
  `spikeforge_hub/` — see
  [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md).

## Related decisions

- [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) — distributions and import roots.
- [`plans/arch-0001-core-boundary.md`](plans/arch-0001-core-boundary.md) — the forbidden list the headless proof checks.
- [`plans/arch-0001-protocol-contract.md`](plans/arch-0001-protocol-contract.md) — the independent `protocol_version`.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — the ordered extraction steps.

# Contributing to spikeforge

Thanks for your interest in improving `spikeforge`. This document explains
how to set up the project, the conventions a change must follow, and the gates
it must pass before it can be merged.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Bug reports** — open an issue with the bug template and include the exact
  command, the observed output, and the expected output.
- **Feature requests** — open an issue with the feature template; describe the
  user goal, not just the implementation.
- **Pull requests** — small, focused changes with tests and a clear rationale
  are the easiest to review.

## Getting set up

The project targets **Python 3.10–3.13** and Node.js 18+ (CI uses Node 22) for
the TypeScript client.

```bash
# From the repository root. scripts/dev.sh prefers ./venv when present.
scripts/dev.sh setup          # core dev extra + server distribution + npm install
```

Or, by hand:

```bash
python -m venv venv
venv/bin/pip install -e "./packages/spikeforge[dev]"
venv/bin/pip install -e ./packages/spikeforge-server
(cd client && npm install)
```

Optional capabilities are extras (`nir`, `events`, `onnx`, `hub`, `norse`,
`lava`, `tracking`, `tracking-wandb`, `docs`). Each has an isolated probe, so a
missing package is *reported*, never raised at import. See the
[requirements reference](documentation/requirements.md#optional-extras) for the full matrix.

## The style contract

[`rules.md`](rules.md) is binding, not advisory. In brief:

- **Python**: files ≤ 250 lines, classes ≤ 200, functions ≤ 20; one class per
  file; 79-column lines; full type hints on every signature; docstrings on
  public modules, classes, and functions. `ruff` enforces the mechanical parts.
- **TypeScript**: files ≤ 250 lines, components ≤ 200, hooks/helpers ≤ 120;
  80-column lines; `strict` on, **no `any`**, no `@ts-ignore`; one exported
  component per file.
- **No shortcuts**: never `# noqa`, `type: ignore`, `eslint-disable`, shims,
  monkeypatches, or `!important`. Fix the underlying issue.
- **Honesty rule**: unsupported, unavailable, or estimated things are reported
  explicitly with a named reason — never faked and never silently degraded.

## Gates a change must pass

Run the same checks CI runs, locally, before opening a pull request:

```bash
ruff check .                                  # lint
pytest                                        # full suite
bash scripts/build_docs.sh --check            # documentation links
(cd client && npm ci && npm run build)        # TypeScript build
```

`scripts/dev.sh check` runs the lint, test, and client steps together.

Baseline when this file was written: **ruff clean**, **739 passed, 1 skipped**
with the optional extras installed. If you add a capability, add the test that
proves it and keep the baseline honest.

## Pull requests

- Keep changes focused; one concern per pull request.
- Add or update tests for any behavior change.
- Update the [CHANGELOG](CHANGELOG.md) under `## [Unreleased]`.
- Keep the [README](README.md) and [Open-source checklist](OPEN_SOURCE_CHECKLIST.md)
  accurate if your change alters what they claim.
- Fill in the pull-request template; it mirrors the gates above.

## Reporting security issues

Do **not** open a public issue for a vulnerability. Follow
[SECURITY.md](SECURITY.md).

## License

Contributions are accepted under the project's [BSD 3-Clause license](LICENSE).
By submitting a pull request you agree that your contribution is licensed under
those terms. Add yourself to the [AUTHORS](AUTHORS) file with a significant
contribution.

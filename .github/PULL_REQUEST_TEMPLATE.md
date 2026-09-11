<!--
Thanks for the pull request. Keep it focused on one concern, and fill in the
sections below. Delete this comment before submitting.
-->

## Summary

<!-- What does this change do, and why? Link the issue it closes with "Closes #N". -->

## Related issue

<!-- e.g. Closes #123. Write "N/A" if there is none. -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (explain the migration in the summary)
- [ ] Documentation only
- [ ] Packaging / CI / tooling only

## How it was verified

<!-- Paste the exact commands and their results. -->

## Checklist

- [ ] `ruff check .` is clean.
- [ ] `pytest` passes (baseline: 739 passed, 1 skipped with the optional extras
      installed). New behavior has a test that proves it.
- [ ] `bash scripts/build_docs.sh --check` passes if documentation changed.
- [ ] `cd client && npm ci && npm run build` passes if the client changed.
- [ ] The style contract in [`rules.md`](../rules.md) is respected: no file over
      the line limit, no `# noqa` / `type: ignore` / `eslint-disable` / `any`,
      no shims or monkeypatches.
- [ ] The **honesty rule** holds: unsupported, unavailable, or estimated results
      are reported with a named reason — nothing is faked or silently degraded.
- [ ] Product behavior is unchanged unless this pull request is explicitly a
      behavior change.
- [ ] [`CHANGELOG.md`](../CHANGELOG.md) is updated under `## [Unreleased]`.
- [ ] The [`README.md`](../README.md) and
      [`OPEN_SOURCE_CHECKLIST.md`](../OPEN_SOURCE_CHECKLIST.md) claims are still
      accurate.

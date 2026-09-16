# Agent guide — spikeforge

Instructions for any coding agent working in this repository. Not specific to
one vendor; `CLAUDE.md` defers to this file.

## Read these first

- **[`rules.md`](rules.md)** — the project's architecture invariants and code
  style, including the hard file, class, and function length limits and the
  prohibited shortcuts (`# noqa`, `type: ignore`, `any`, `@ts-ignore`).
- **[`COPY_POLICY.md`](COPY_POLICY.md)** — required before drafting or editing
  any public-facing prose: the landing page under `landing/`, the model hub
  page rendered by `scripts/build_hub_page.py`, `README.md`, and the
  documentation under `docs/`.

## Public-facing copy, in short

The full policy is `COPY_POLICY.md`; this is a reminder of what comes up most
in this repository, not a replacement for reading it.

- Establish the facts first. Do not invent features, quantities, benchmarks,
  audiences, or licensing terms. Existing AI-generated wording is not a source
  of verified facts — check the implemented behavior.
- Interface labels and headings name the content or action. A section that
  needs a recognizable name does not get a slogan.
- Do not write text to fill a component. Deletion is a valid edit.
- Keep required disclosures: the pre-1.0 warning, the "Implications and
  boundaries" link, the hub's curation and licensing notices, and the
  "not tuned attempts at state of the art" caveat. Shortening copy is not a
  reason to drop any of them.

The landing page is translated into **seventeen** locales across
`landing/locales.js` and `landing/cta-locales.js`. An English change that is
not carried into the other sixteen leaves them stating the old copy, so treat
a copy change as a change to every locale. `node landing/i18n_test.js` checks
the catalogue loads and the switcher still lists all seventeen.

## Verification

```bash
pytest                        # the suite; pytest.ini adds coverage flags
ruff check .                  # lint
node landing/i18n_test.js     # landing i18n
python scripts/build_hub_page.py --out /tmp/hub.html   # hub page renders
```

## Commits and pull requests

Do not add AI vendor branding, "generated with" footers, session links, or AI
co-author trailers to commits, pull requests, issues, or release notes. Do not
claim human review or human-executed testing that did not happen. Describe what
was actually run, including anything skipped or excluded.

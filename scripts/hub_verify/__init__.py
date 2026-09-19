"""The community-upload verification sandbox runner (issue #48, P4).

Invoked by ``.github/workflows/hub-verify.yml`` on a ``repository_dispatch``
from ``spikeforge-hub-api``, on a GitHub-hosted (never self-hosted) runner:
the self-hosted ``spikeforge-ci`` runner holds the Hetzner deploy key, and
running ``torch.load`` on a stranger's bytes there is the single most
predictable way this design could be compromised
(``plans/hub_accounts_plan.md`` §7.2).

This package is CI scaffolding, not a distributed package: nothing under
``spikeforge*`` imports it, and it ships in no wheel. It reuses the existing,
already-shipped pipeline rather than reimplementing any of it:

* :mod:`spikeforge.serving.bundle` -- checksum/manifest verification and the
  ``weights_only=True`` load already used for every bundle.
* :mod:`spikeforge_hub.inspect` / :mod:`spikeforge_hub.compat` -- the same
  structural inspection and preset classification the curated catalog uses.
* :mod:`spikeforge.nir_bridge` -- the same NIR export and independent
  reference-interpreter drift check invariant 4 of ``rules.md`` requires.
* :mod:`spikeforge_targets.energy` -- the same SOP/MAC/AC accounting used
  elsewhere in the project.

See :mod:`hub_verify.cli` for the orchestration entry point and
:mod:`hub_verify.report` for the report/callback contract this runner
implements against ``spikeforge-hub-api``'s (not yet built, as of this
writing) ``POST /internal/v1/verifications/{version_id}``.
"""

# Local `snn_interpreter` tree: provenance and migration note

Records the standing of the sibling `snn_interpreter/` directory for
Capsize-Games/spikeforge#57. This note moves, deletes, and retires
nothing; retirement is explicitly unauthorized in the adopting pull
request.

## Provenance

- `spikeforge` (this repository) is the canonical source for the client
  and the engine. All client work lands here.
- The sibling `snn_interpreter/` directory (same parent directory as this
  checkout) is retained as an **unversioned mirror**: it carries no `.git`
  directory and is not a checkout of any branch. It is a recoverable
  copy of an earlier tree, not a second maintained tree.
- At the time of writing the two manifests are byte-identical
  (`client/package.json` on `origin/main` versus the mirror's copy), and
  `client/src` shares 68 identical files. The adopting pull request adds
  one line to the canonical manifest (the `@capsizellc/commons`
  dependency); that is the only post-note divergence. The remaining
  differences are classified per file group in
  `plans/client_diff_classification.md`; nothing in the mirror is
  silently dropped by that classification.

## Migration and rollback

- Migration in the adopting pull request: none. No path moves, no file is
  deleted, no consumer is repointed.
- Repointing the `999` consumers named in Capsize-Games/spikeforge#57
  rides its own branch and is out of scope here. This note records the
  requirement; it does not perform the repoint.
- Rollback of the adopting pull request is a plain revert: the mirror is
  untouched, so there is nothing to restore.
- The mirror remains recoverable until the provenance gate is explicitly
  closed (the acceptance criterion in Capsize-Games/spikeforge#57).
  Closing that gate is a separate decision and is not taken here.

## Coordination

Capsize-Games/spikeforge#57 coordinates with Capsize-Games/hq#5, #24,
#30, #34, and #35. Follow-ups (further shared-package adoptions, the
`999` repoint, any retirement proposal) reference this note rather than
re-deriving provenance.

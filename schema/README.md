# Manifest schema

`capsize-1.json` is copied from the previously public MIT-licensed
`Capsize-Games/capsize-ci` revision
`1a50d06c20022f9bbc44bad002add302e1dae646`. The license is retained in
[`../licenses/capsize-ci-MIT.txt`](../licenses/capsize-ci-MIT.txt).

The repository keeps this snapshot locally so manifest validation does not
require access to the now-private CI repository.

This local copy also accepts `fastapi-react` for this repository's existing
combined backend/frontend manifest. The upstream schema's original enum
rejected that value before the privacy migration. No other schema constraint
is changed.

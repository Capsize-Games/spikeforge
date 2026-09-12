# Backend execution (WS-B)

Capability *declaration* becomes executable *deployment*.

### Substitution executor

[`targets/rewrite.py`](../spikeforge_targets/rewrite.py:1) applies a target's
**declared** substitutions to produce a target-ready graph and reports what
changed ([`rewrite_report.py`](../spikeforge_targets/rewrite_report.py:1)):
`applied`, `skipped`, `unfixable`. Two rules ship — `IF`→`beta=0` `LIF` for
`norse` and `AvgPool2d`→`SumPool2d`+`Scale` for `lava_loihi2`. An unfixable
primitive is named, never dropped, and a post-rewrite **drift check** quantifies
any residual.

### Real backends: reference, norse, lava_loihi2

[`targets/backends.compile_run()`](../spikeforge_targets/backends/__init__.py:137)
is the single entry point. It rewrites, optionally quantizes, gates on the
backend's availability, then compiles, runs, and compares the result to the
reference interpreter — returning a `BackendResult` whose `status` is `ok`,
`unavailable`, or `error`. Nothing raises and nothing is faked.

| Backend | Extra | Behavior |
|---|---|---|
| `reference` | — | In-process NIR interpreter; always available |
| `norse` | `norse` | Pure-PyTorch simulator; runs when the extra is installed |
| `lava_loihi2` | `lava` | Lava/Loihi 2 path; runs when the SDK is installed |

An absent SDK yields `status: "unavailable"` with a note naming the extra. SDK
imports are confined to
[`backends/api.py`](../spikeforge_targets/backends/api.py:1).

### `deploy` / `rewrite` / `run`

```bash
spikeforge-verify deploy  --topology conv_net --target reference   # capability view
spikeforge-verify rewrite --topology conv_net --target norse       # substitutions + drift
spikeforge-verify run     --topology conv_net --target reference   # compile + run + compare
```

The same commands are on `spikeforge-targets`. `deploy` exits `0` only when
`deployable`; `run` exits non-zero unless `status == "ok"` and the comparison
to the reference is within tolerance.

### WebSocket + client

A `deploy_run` action (`backend_run` reply) adds the *executed* view beside the
existing `deployment_report`. The
[`BackendRunPanel`](../client/src/components/BackendRunPanel.tsx:1) renders the
status, the rewrite report, the drift, and the reference comparison.

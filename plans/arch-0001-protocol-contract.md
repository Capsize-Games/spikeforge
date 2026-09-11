# ARCH-0001 protocol contract

**Status: accepted — implemented.**
**Date:** 2026-09-11 · **Issue:** ARCH-0001 *Phased repo split: core library, deploy targets, dashboard*
**Owner:** Capsize Games (maintainer) · **Depends on:** [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md)

## Decision

Adopt a **JSON Schema (draft 2020-12) source of truth** for the WebSocket
protocol, checked into a new top-level `protocol/` directory, and add a
**`protocol_version` field to every client and server envelope**.

**The initial `protocol_version` value is the string `"1.0"`.**

The hand-written TypeScript protocol types under `client/src/` are generated from
— or validated against — these schemas. The Python pydantic models under
`server/schemas/` become a *derived* consumer of the same schemas, guarded by a
parity test. The JSON Schema is the only artifact that may be edited as an
authority; everything else is generated or checked.

## Why this must precede any split

`plans/repo_topology_plan.md` §2 identifies the client/server protocol as the
one real coupling: `server/schemas/client_message.py` and
`server/schemas/server_message.py` are pydantic `BaseModel`s with a **closed**
`Literal` `type` discriminator and **no version field**, `server/messages.py`
emits bare dicts through `ws.send_json` via `send_locked`, and the TS types are
hand-mirrored. A single Python test
([`tests/test_client_animation_payload.py`](tests/test_client_animation_payload.py))
is the only guard. That is tolerable while client and server share a repo; it is
not tolerable once Phase 2 extracts the dashboard. The contract must become
explicit first, which is exactly why
[`plans/arch-0001-adr-repo-topology.md`](plans/arch-0001-adr-repo-topology.md)
makes it a Phase 1 deliverable rather than a consequence of the split.

## `protocol/` directory layout

```text
protocol/
  README.md                          # how to change the contract
  protocol_version.txt               # "1.0" — single source of the version string
  envelope.schema.json               # $defs shared by both directions
  client_message.schema.json         # inbound envelope
  server_message.schema.json         # outbound envelope
  payloads/
    encode_config.schema.json        # mirrors server/schemas/encode_config.py
    train_config.schema.json         # mirrors server/schemas/train_config.py
    model_query.schema.json          # mirrors server/schemas/model_query.py
    hub_query.schema.json            # mirrors server/schemas/hub_query.py
    server_payloads.schema.json      # the free-form payload discriminated by type
    animation_payload.schema.json    # the payload guarded by the parity test
  codegen/
    generate_ts.mjs                  # json-schema-to-typescript driver
    tsconfig.json
```

## The envelopes

Every message in either direction carries `protocol_version` and `type`, and
both envelopes set `additionalProperties: true` so unknown fields are ignored by
default — this is what makes the additive-by-default policy below enforceable.

### Client to server — `protocol/client_message.schema.json`

The `type` enum captures all inbound message kinds. The source literal in
[`server/schemas/client_message.py`](server/schemas/client_message.py:16) lists
**35** tokens today (see Corrections); the schema is generated to match the
source and must stay in sync via the parity test.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://spikeforge.dev/protocol/1.0/client_message.schema.json",
  "title": "ClientMessage",
  "type": "object",
  "required": ["protocol_version", "type"],
  "properties": {
    "protocol_version": { "type": "string", "const": "1.0" },
    "type": {
      "type": "string",
      "enum": [
        "configure", "run", "stop", "train", "stop_train", "predict",
        "select_sample", "infer", "save_model", "list_models",
        "load_model", "delete_model", "new_model", "stats",
        "cancel_download", "nir_export", "nir_validate",
        "trajectory", "metrics", "encoding_report",
        "surrogates", "surrogate_curve", "benchmark",
        "targets", "deployment_report", "deploy_run", "energy_report",
        "model_search", "model_diff",
        "hub_list", "hub_search", "hub_download", "hub_cancel",
        "hub_inspect", "hub_import"
      ]
    },
    "config": { "$ref": "payloads/encode_config.schema.json" },
    "train": { "$ref": "payloads/train_config.schema.json" },
    "query": { "$ref": "payloads/model_query.schema.json" },
    "hub": { "$ref": "payloads/hub_query.schema.json" },
    "name": { "type": ["string", "null"] },
    "sparse": { "type": "boolean", "default": true }
  },
  "additionalProperties": true
}
```

### Server to client — `protocol/server_message.schema.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://spikeforge.dev/protocol/1.0/server_message.schema.json",
  "title": "ServerMessage",
  "type": "object",
  "required": ["protocol_version", "type"],
  "properties": {
    "protocol_version": { "type": "string", "const": "1.0" },
    "type": {
      "type": "string",
      "enum": [
        "status", "image", "raster", "spike_frame",
        "curve", "error", "config_ack", "run_state",
        "train_metrics", "train_state", "prediction",
        "model_saved", "model_list", "model_loaded", "model_cleared",
        "inference", "system_stats", "download_state",
        "nir_graph", "nir_validation",
        "trajectory", "metrics", "encoding_report",
        "surrogate_list", "surrogate_curve", "benchmark",
        "target_list", "deployment_report", "backend_run",
        "energy_report", "animation_state",
        "hub_list", "hub_search", "hub_download_state",
        "hub_inspect", "hub_import"
      ]
    },
    "payload": true,
    "source": { "type": ["string", "null"] },
    "kind": { "type": ["string", "null"] }
  },
  "additionalProperties": true
}
```

The outbound envelope MUST declare `kind`. Today `server/messages.py` emits a
`kind` field (for example `server/messages.py:57` and `server/messages.py:67`)
that the pydantic `ServerMessage` in
[`server/schemas/server_message.py`](server/schemas/server_message.py:8) does not
declare and does not strip, so it already travels on the wire as an undeclared
property. Making it explicit in the schema removes the shadow contract (see
Corrections).

## Compatibility policy

**Additive is the default; breaking changes are a MAJOR bump.** Concretely:

- **MINOR-incompatible additive change (no version bump).** Adding an optional
  field to a `payload`, adding a new payload key, or adding a descriptive field
  to an envelope is allowed **without** a `protocol_version` change. Consumers
  MUST ignore unknown fields (`additionalProperties: true`) and MUST NOT rely on
  field ordering.
- **MINOR bump (additive, breaking for closed consumers).** Adding a new value
  to a `type` enum requires a MINOR bump (`1.0` → `1.1`). The enum is closed on
  purpose; a consumer that does not know a new `type` must ignore the message
  and continue, never crash.
- **MAJOR bump (breaking).** Removing or renaming a `type`, removing or
  retyping an existing field, or changing required-ness is a MAJOR bump
  (`1.x` → `2.0`). MAJOR bumps require a coordinated release of the server and
  the dashboard and the negotiated-version window (below).

### Negotiation and the missing-field transition

- Every message carries `protocol_version`. The server compares the **MAJOR**
  component of an inbound `protocol_version` with its own. Matching MAJOR is
  accepted; a mismatching MAJOR is rejected with a `type: "error"` message
  carrying `payload.code = "protocol_version_mismatch"`.
- **Transition window (Phase 1 only; now closed):** during Phase 1 an inbound
  message *lacking* `protocol_version` was treated as legacy `"0.x"`, accepted
  with a one-line deprecation log, and answered with a `config_ack` stating
  `protocol_version: "1.0"`. That window is closed: a missing
  `protocol_version` is now a `protocol_version_mismatch` error.
- The server always stamps its own `protocol_version` on every outbound
  message, so an old client can detect a newer server.

## TypeScript generation and validation

The dashboard stops hand-mirroring. Two generated/validated artifacts replace
the handwritten types:

1. **Codegen.** `protocol/codegen/generate_ts.mjs` reads every
   `protocol/**/*.schema.json` with `json-schema-to-typescript` and writes
   `client/src/protocol/generated.ts`. The client adds a dev dependency on
   `json-schema-to-typescript` and a script `npm run gen:protocol`.
2. **Runtime validation.** The dashboard compiles the same schemas with `ajv`
   into a small validator and assert-validates inbound frames in development
   builds. Production builds keep the type imports but may compile the validator
   out.

CI enforcement (added to the existing `client` job, which today runs
`npm ci && npm run build`):

```text
1. npm run gen:protocol
2. git diff --exit-code client/src/protocol/generated.ts   # generated is current
3. npm run build                                            # tsc + vite
```

Step 2 makes a schema change without regenerated types a CI failure, and a
generated-type change without a schema change a failure too.

## How the existing parity test evolves

[`tests/test_client_animation_payload.py`](tests/test_client_animation_payload.py)
stays and keeps its exact role: it is the cross-package integration test that
proves a real emitted animation payload matches the client's expected shape. It
is joined by a new `tests/test_protocol_schema_parity.py` that asserts:

1. the pydantic `ClientMessage` `type` literal set equals the `client_message`
   schema's `type` enum;
2. the pydantic `ServerMessage` `type` literal set equals the `server_message`
   schema's `type` enum;
3. a captured sample of real `server/messages.py` emissions validates against
   `server_message.schema.json` (via the `jsonschema` library), which is what
   catches the undeclared `kind` field;
4. `protocol/protocol_version.txt` equals the `const` in both envelopes.

Together these four assertions plus the animation payload test are the protocol
contract's regression net; they are the tests that MUST survive the Phase 2
dashboard extraction ([`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md)).

## Corrections and findings

- **Closed-enum counts.** The issue brief states "33 inbound / 36 outbound". The
  verified source literal in
  [`server/schemas/client_message.py`](server/schemas/client_message.py:16)
  lists **35** inbound tokens and
  [`server/schemas/server_message.py`](server/schemas/server_message.py:11)
  lists **36** outbound tokens. The schema enums above match the verified
  source; the parity test is the authority and will fail loudly if either drifts.
- **`kind` is an undeclared wire field.** See the outbound envelope note above;
  it is emitted by `server/messages.py` but absent from the pydantic model.
- **No `protocol_version` exists today**, and the discriminator is closed with
  no negotiation, so any add/remove is a silent break. This decision closes that
  gap before the dashboard leaves the repository.

## Related decisions

- [`plans/arch-0001-target-topology.md`](plans/arch-0001-target-topology.md) — why `protocol/` is a dependency leaf.
- [`plans/arch-0001-packaging-versioning.md`](plans/arch-0001-packaging-versioning.md) — how the protocol version relates to package versions.
- [`plans/arch-0001-migration-plan.md`](plans/arch-0001-migration-plan.md) — which parity tests survive extraction.
- [`plans/arch-0001-decision-metrics.md`](plans/arch-0001-decision-metrics.md) — the dashboard-extraction trigger.

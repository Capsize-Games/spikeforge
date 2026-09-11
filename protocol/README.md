# WebSocket protocol contract

This directory is the **single source of truth** for the browser/server
WebSocket protocol. The JSON Schema files here (draft 2020-12) are the only
editable authority: the Python pydantic models under `server/schemas/`, the
generated TypeScript under `client/src/protocol/generated.ts`, and the wire
itself are all *derived* from or *checked against* these schemas.

- `protocol_version.txt` — the current version string. It is the single
  source of `protocol_version` for the server, both envelopes, and codegen.
- `envelope.schema.json` — the shared `$defs` vocabulary.
- `client_message.schema.json` — the inbound (browser → server) envelope.
- `server_message.schema.json` — the outbound (server → browser) envelope.
- `payloads/*.schema.json` — the nested payload shapes.
- `codegen/generate_ts.mjs` — emits `client/src/protocol/generated.ts`.

## How to change the contract

**Additive is the default; breaking changes are a MAJOR bump.** Edit the
schema first, then regenerate and let the parity test
(`tests/test_protocol_schema_parity.py`) prove the Python models still agree.

1. **Additive, non-breaking (no version bump).** Adding an optional payload
   key, a new descriptive envelope field, or relaxing a constraint is allowed
   without changing `protocol_version`. Every envelope and payload sets
   `additionalProperties: true`, so consumers MUST ignore unknown fields and
   MUST NOT rely on field ordering.
2. **New `type` enum value (MINOR bump).** Adding a value to a `type` enum is
   additive but breaking for a *closed* consumer, so it requires a MINOR bump
   (`1.0` → `1.1`). A consumer that does not recognise a new `type` MUST
   ignore the message and continue, never crash.
3. **MAJOR bump (`1.x` → `2.0`).** Removing or renaming a `type`, removing or
   retyping an existing field, changing required-ness, or changing a `const`
   is a MAJOR bump. MAJOR bumps require a coordinated server + dashboard
   release and a shim window.

When the version changes, update `protocol_version.txt` and the `const` in
**both** envelopes together; the parity test asserts they stay equal.

## Regenerating the TypeScript

```bash
cd client
npm install            # once, to pick up json-schema-to-typescript
npm run gen:protocol   # writes client/src/protocol/generated.ts
```

CI enforces `git diff --exit-code client/src/protocol/generated.ts` after
`npm run gen:protocol`, so a schema change without regenerated types fails,
and a generated-type change without a schema change fails too.

## Compatibility policy (summary)

| Change | Version action | Consumer rule |
|---|---|---|
| New optional field / payload key | none | ignore unknown fields |
| New `type` enum value | MINOR | ignore unknown `type` |
| Remove / rename / retype / re-required | MAJOR | coordinate a release |

**Missing `protocol_version`.** An inbound message without
`protocol_version` is rejected with a `type: "error"` message carrying
`payload.code = "protocol_version_mismatch"`, exactly like a present value
whose MAJOR component differs from the server's. (Phase 1 accepted a missing
version as legacy `"0.x"` with a one-line deprecation log; that transition
window is closed as of Phase 2.)

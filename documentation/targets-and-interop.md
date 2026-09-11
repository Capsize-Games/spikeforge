# Targets and interoperability (Phase 5)

Phase 5 turns an exported NIR graph into a *deployment story* and consumes
graphs from other frameworks. It adds a deployment-target registry, a
capability matrix, a per-target deployment report, and an external NIR
import/export surface with a round-trip fidelity guarantee. Every surface
(CLI, WebSocket, dashboard) renders from the same payload shapes.

### Target registry and availability model

[`spikeforge_targets/`](../spikeforge_targets/__init__.py:1) declares
what each target *can* run as a [`TargetSpec`](../spikeforge_targets/target_spec.py:10):
its kind, the pip `extra` that would install its SDK, the primitives it
supports, its substitutions, and its constraints (dtype, timestep,
quantization). The registry ships:

| Target | Kind | Extra | Notes |
|---|---|---|---|
| `reference` | reference | — | In-process NIR interpreter; always available |
| `lava_loihi2` | hardware | `lava` | Lava SDK path to Intel Loihi 2 |
| `spinnaker2` | hardware | `spinnaker2` | SpiNNaker2 digital hardware |
| `speck` | hardware | `speck` (`sinabs`) | SynSense Speck edge chip |
| `xylo` | hardware | `xylo` (`rockpool`) | SynSense Xylo LIF fabric |
| `norse` | simulator | `norse` | Norse PyTorch simulator |

SDKs are optional and are reported **honestly**. Availability is resolved on
demand through isolated probes
([`targets/probe.py`](../spikeforge_targets/probe.py:1) and
[`targets/backends/api.py`](../spikeforge_targets/backends/api.py:1) — the
only modules that import a backend SDK; both import nothing at module load
time). A target whose SDK is absent is returned with `"available": false` and
named in the report notes; it is never hidden or silently treated as ready.
The `reference` target is always available, and `norse`/`lava_loihi2` gain
executable backends when their extras are installed (see WS-B above).

### Capability matrix

[`classify(graph_or_spec, target)`](../spikeforge_targets/capability_matrix.py:13)
places every node of a graph in exactly one
[`CapabilityMatrix`](../spikeforge_targets/matrix_result.py:10) bucket:

- **supported** — the target runs the node's primitive natively.
- **substituted** — the target lacks the primitive but declares a replacement
  (for example Loihi 2 maps `AvgPool2d` to `SumPool2d`; Norse maps `IF` to a
  `beta=0` `LIF`). The record names the node, the primitive, and its
  substitute.
- **unsupported** — no native support and no declared substitute; the node
  name is reported explicitly.

The three buckets partition the node set, so a node is **never silently
dropped**.

### Deployment report and `deployable`

[`deployment_report(spec_or_graph, target)`](../spikeforge_targets/report.py:47)
returns JSON carrying the classified `nodes` (with per-bucket counts), the
target's `constraints`, an optional `validation` drift section, and
human-readable `notes`. `deployable` is true **only** when the target is
`available` and has zero unsupported nodes; a target with a missing SDK or a
gap is reported `deployable: false` rather than raising. The `deploy` CLI
command and the WebSocket `deployment_report` action emit the same payload.

### External NIR import/export and round-trip fidelity

[`spikeforge/nir_bridge/`](../spikeforge/nir_bridge/__init__.py:1)
grows a cross-library surface:

- [`save_graph`](../spikeforge/nir_bridge/serialization.py:52) /
  [`load_graph`](../spikeforge/nir_bridge/serialization.py:125) persist a
  graph in a version-stamped JSON envelope. Node semantics stay owned by
  `nir`'s own `to_dict`/`dict2NIRNode`; numpy values are tagged with dtype and
  shape, so a reload reconstructs the exact array rather than a rounded list.
- [`load_external`](../spikeforge/nir_bridge/ingest.py:21) /
  [`interpret_graph`](../spikeforge/nir_bridge/ingest.py:30) /
  [`interpret_file`](../spikeforge/nir_bridge/ingest.py:35) ingest a graph
  produced elsewhere and run it on the independent interpreter, which never
  touches snnTorch.
- [`roundtrip`](../spikeforge/nir_bridge/roundtrip.py:84) persists, reloads,
  and compares the reloaded interpretation against the in-memory export. The
  report is `identical: true` only when every spike and membrane trace and the
  readout match with zero maximum absolute error.

Failures are typed and named, never silent:
`GraphNotFoundError`, `MalformedGraphError`, `UnknownNodeKindError`, and
`UnsupportedNodeError` ([`errors.py`](../spikeforge/nir_bridge/errors.py:1)).

### CLI subcommands

The `verify` CLI gains four subcommands (all print JSON; they exit non-zero
on a negative result so they double as CI gates):

```bash
python -m spikeforge.cli.verify targets
python -m spikeforge.cli.verify deploy --topology conv_net --target reference
python -m spikeforge.cli.verify deploy --topology conv_net --target xylo
python -m spikeforge.cli.verify roundtrip --topology conv_net --out build/graph.json
python -m spikeforge.cli.verify ingest --file build/graph.json
python -m spikeforge.cli.verify test-deploy --topology fc_legacy
```

`targets` lists the registry with live availability; `deploy` classifies a
topology against a target and exits `0` only when `deployable`; `roundtrip`
exits `0` only when the persisted graph is `identical`; `ingest` runs a saved
external graph and prints its traced nodes, or a typed error with a non-zero
exit. `ingest` reads the version-stamped JSON envelope written by
`roundtrip --out` (or `nir_bridge.save_graph`), **not** the node/edge summary
that `export --out` writes.

### Simulator-backed test-deploy matrix

[`test_deploy.run_matrix(graph, spikes)`](../spikeforge_targets/test_deploy.py:1)
runs one graph across **every** registered target and returns a
[`TestDeployMatrix`](../spikeforge_targets/test_deploy_result.py:60) with one
[`DeployCell`](../spikeforge_targets/test_deploy_result.py:25) per target. Each
cell carries the target's capability classification
([`classify`](../spikeforge_targets/capability_matrix.py:13)) plus its
deployment outcome: `status` (`ok`/`unavailable`/`error`), the execution
`path` (e.g. a Lava emulator), a named `reason` when it did not complete, and
the reference `parity` comparison.

The matrix is deliberately honest, matching the availability model above:

- the `reference` cell always runs in-process and compares to itself;
- an SDK-backed simulator whose SDK is absent reports `available: false`,
  `status: "unavailable"`, and the enabling extra in its `reason`;
- a declared-only simulator with no backend wired reports `unavailable` with a
  reason naming it, **never** a failure and never silently skipped;
- an available backend that refuses the graph reports `status: "error"` with
  the offending node and kind named.

Every catalog target now has exactly one executable backend. Besides
`reference`, `norse`, and `lava_loihi2`, the vendor simulators are wired
through isolated probes and backends: **Speck**
(`sinabs`), **Xylo** (`rockpool`), and **SpiNNaker2** (`spinnaker2`). Each
probes its SDK by a lazy import plus a minimal capability check, lowers a
linear chain with the shared lowering, and runs through an isolated SDK
touch-point that names its `path` (`speck_simulator`, `xylo_simulator`,
`spinnaker2_simulator`). A present SDK yields `available: true` and
`status: "ok"|"error"`; an absent or unrecognised SDK yields
`available: false` with a named reason.

`ok` is true when the reference ran and no *available* backend errored, so a
missing optional SDK never fails the matrix. `estimate` is always `true` — at
the matrix level and on every cell: a simulator or emulator run is never a
device measurement, and hardware energy and latency stay declared estimates.

```bash
python -m spikeforge.cli.verify test-deploy --topology fc_legacy
python -m spikeforge.cli.verify test-deploy --topology fc_legacy --targets reference,norse
```

### WebSocket actions

Two read-only actions were added, with client types in
[`client/src/targetTypes.ts`](../client/src/targetTypes.ts:1):

| Action | Reply | Payload |
|---|---|---|
| `targets` | `target_list` | Availability-annotated target registry |
| `deployment_report` | `deployment_report` | Capability matrix, constraints, optional drift |

An unknown target or topology emits the existing `error` message. A report is
still produced when no sample is loaded — the drift section is simply omitted,
never fabricated.

### TargetsPanel

The dashboard renders a
[`TargetsPanel`](../client/src/components/TargetsPanel.tsx:19): it lists the
registry (kind, extra, availability), lets you select a target, and shows its
deployment report as supported/substituted/unsupported buckets, a constraint
table, and the optional drift table. A report is only shown when its target
matches the current selection, so a stale reply can never imply support for a
different target.

### Limitations

> **See also** [Implications and boundaries](implications-and-boundaries.md) for
> why each boundary below exists and what it implies for a user.

- **Availability, not capability.** The in-process `reference` target is
  always available. `norse` and `lava_loihi2` now have executable backends
  (WS-B) that compile and run when their extras are installed; without the SDK
  they report `available: false` and `run` returns `status: "unavailable"`.
  `spinnaker2`, `speck`, and `xylo` remain declarative placeholders.
- **Substitutions are executed.** The declared mapping stays the source of
  truth; the rewrite executor (WS-B) applies it and reports a post-rewrite
  drift check.
- **No on-device measurement.** `reference`, `norse`, and `lava_loihi2` compile
  and run, but no physical device is attached, so hardware timing and energy
  are not measured.
- **Graph exchange uses NIR's node vocabulary.** Import/export round-trips a
  graph in this project's version-stamped JSON envelope, and `nirtorch`
  extraction of third-party PyTorch modules ships in WS-F.

# Spikeforge — WS-D: Event-Driven Sparse Runtime and Energy Accounting

> Focused design for workstream **D** of the
> [`professional_roadmap.md`](plans/professional_roadmap.md). Read the roadmap
> first for the vision, cross-workstream interfaces, and packaging rules.

Design/spec only. Every claim about the current code is grounded in a file/line
reference so a Code-mode agent can execute this file-by-file.

**Objective.** Add a **sparse/event-driven execution path** beside the existing
dense unroll — event queues and spike-propagated synaptic operations rather than
dense tensor MACs — and an **energy/latency accounting model** that counts
synaptic operations, MACs vs. ACs, and timesteps, then maps those counts to a
**declared per-target cost table**. Surface both in the benchmark harness, a new
report, the CLI, and the dashboard. The dense path stays the untouched default.

**Honesty headline.** Energy numbers are **estimates from declared cost tables**,
never presented as measured, unless a real device is present and reports its own
timing. Every `EnergyReport` carries an explicit `estimate: true` field and a
`basis` describing where the numbers come from.

---

## 1. Current state and gap analysis

| Capability | Current reality | Anchor |
|---|---|---|
| Execution loop | One dense temporal loop | [`execution.py`](spikeforge/simulator/execution.py:122) |
| Runner entry | `run(...)` with modes | [`runner.py`](spikeforge/simulator/runner.py:22) |
| Trajectory capture | `S[t]`/`U[t]`/`I[t]` | [`trajectory.py`](spikeforge/simulator/trajectory.py:9) |
| Stage step | Stateless per-step `(input, state)` | [`stage_module.py`](spikeforge/topology/stage_module.py) |
| Benchmark harness | Time/memory only | [`harness.py`](spikeforge/benchmark/harness.py:1) |
| Op counting | None | n/a |
| Energy model | None | n/a |
| Target cost data | Constraints only (dtype, quantization) | [`target_spec.py`](spikeforge_targets/target_spec.py:28) |
| Sparse input | Input is already binary spikes | [`input_shape.py`](spikeforge/simulator/input_shape.py) |

### 1.1 Invariants that must not break

- The dense path ([`execute`](spikeforge/simulator/execution.py:122),
  [`run`](spikeforge/simulator/runner.py:22)) is the default and is
  unchanged; existing trajectories and tests stay byte-identical.
- The sparse path returns a `Trajectory` with the same shape contract, so it is
  a drop-in for readout comparison.
- `ExecutionMode.PRODUCTION`/`EDUCATIONAL` semantics are unchanged
  ([`execution_mode.py`](spikeforge/runtime/execution_mode.py:12)).
- Existing benchmark payload keys stay additive
  ([`server_message.py`](server/schemas/server_message.py:11)).
- `reference` target remains available; energy cost tables are declared, so an
  absent target still yields an estimate with a note.

---

## 2. Sparse / event-driven runtime

### 2.1 Approach

The simulator already treats each stage as a pure `(input, state) -> (output,
state)` function driven by binary spike frames
([`execution.py`](spikeforge/simulator/execution.py:82)). Dense execution
multiplies full spike tensors by weight matrices, paying for zeros. The sparse
path instead propagates **events**: for each step, only indices where
`spike == 1` contribute, and synaptic operations are applied to those indices
only.

This is an **inference path** (and training-free): it has no surrogate
gradients, which is fine because its purpose is honest op-counting and fast
sparse inference. Training continues to use the dense path.

### 2.2 Module layout

```
spikeforge_targets/event_runtime/
  __init__.py
  spike_view.py      SparseSpikes: indices/values per frame; density helpers
  ops.py             event-driven linear/conv/pool applied to sparse spikes
  counters.py        SynapticCounter: SOP, MAC, AC, timestep tallies
  sparse_runner.py   sparse_run(module, spikes, counters=True) -> SparseResult
  dense_compare.py   compare a SparseResult to a dense Trajectory
  errors.py          typed sparse errors (unsupported stage kind)
```

### 2.3 Counting semantics

| Symbol | Meaning | Rule |
|---|---|---|
| **AC** | accumulate operation | 1 per output element per step |
| **SOP** | synaptic operation | 1 per active (spike `==1`) input-output pair |
| **MAC** | multiply-accumulate | dense baseline: 1 per input-output pair per step |
| **timesteps** | `T` | number of temporal steps |

For a dense layer, `SOP == MAC`. For a sparse spike input, `SOP` counts only the
active pairs, so `SOP/MAC` is the sparse efficiency ratio the acceptance test
measures.

### 2.4 Acceptation bar

- **Parity:** `dense_compare` asserts the sparse readout equals the dense readout
  within tolerance on a seeded fixture.
- **Reduction:** on a sparse fixture (low spike density), measured `SOP < MAC`
  by the expected ratio; the test asserts a strict reduction.

---

## 3. Energy and latency accounting

### 3.1 Cost tables (declared, not measured)

Each target gains a declared cost table: energy per SOP/MAC/AC and latency per
timestep. Stored as bundled JSON next to the catalog and surfaced through the
same `TargetSpec` shape, so a target's costs travel with its capability matrix.

```
spikeforge_targets/energy/costs/
  reference.json     in-process estimates
  norse.json         simulator estimates
  lava_loihi2.json   declared device estimates
  spinnaker2.json
  speck.json
  xylo.json
```

| Field | Meaning |
|---|---|
| `sop_pj` | energy per synaptic operation, picojoules |
| `mac_pj` | energy per dense MAC |
| `ac_pj` | energy per accumulate |
| `step_ns` | nominal latency per timestep |
| `source` | citation/notes for the numbers |
| `measured` | `false` for every bundled table |

A target with no bundled table still produces a report with `basis:
"unavailable"` and a note, never a fabricated number.

### 3.2 Module layout

```
spikeforge_targets/energy/
  __init__.py
  cost_table.py      load/validate a target's declared costs
  target_costs.py    bundled per-target table lookup
  accounting.py      account(run_or_spec, target) -> EnergyReport
  report.py          JSON-able report assembly
  probe.py           isolated device probe (measured energy when present)
  cli.py             spikeforge-energy entry point
  errors.py          typed energy errors
```

### 3.3 Report shape

```json
{
  "target": "reference",
  "estimate": true,
  "basis": "declared cost table",
  "timesteps": 25,
  "ops": {"sop": 120000, "mac": 400000, "ac": 80000},
  "efficiency": {"sop_over_mac": 0.30},
  "energy": {"sop_pj": 1200000.0, "mac_pj": 4000000.0, "total_pj": 1600000.0},
  "latency": {"step_ns": 1.0, "total_ns": 25.0},
  "notes": ["estimate only; no device measured"]
}
```

`estimate` is `true` unless `probe.py` finds a device that reports its own
timing, in which case `measured` fields are added and the note changes.

---

## 4. Surfaces

### 4.1 Benchmark harness

[`benchmark/harness.py`](spikeforge/benchmark/harness.py:1) gains an opt-in
`--energy` path that runs the sparse runner and attaches an `energy` block to
each topology/mode record. Because the harness already reports unavailable
metrics as `null`, an unavailable cost table is `null` with a note, consistent
with [`harness.py`](spikeforge/benchmark/harness.py:1).

### 4.2 WebSocket

| Action | Request | Reply | Payload |
|---|---|---|---|
| `energy_report` | `{train, name: target, sparse?}` | `energy_report` | `EnergyReport` + optional dense/sparse comparison |

Routed via [`protocol_handlers.py`](server/protocol_handlers.py:19) to a new
`server/energy_handlers.py` and `server/energy_payloads.py`.

### 4.3 CLI

New console script `spikeforge-energy` ([`energy/cli.py`](spikeforge_targets/energy/cli.py)):

```
spikeforge-energy account --topology conv_net --target reference [--sparse]
spikeforge-energy report  --topology conv_net --target lava_loihi2 --out energy.json
```

JSON output; `report` writes to `--out` when given, mirroring
[`_run_export`](spikeforge/cli/verify.py:72).

### 4.4 Client

[`EnergyPanel.tsx`](client/src/components/EnergyPanel.tsx) renders the report:
a dense-vs-sparse SOP/MAC bar, energy totals with a prominent "estimate" badge, a
per-stage breakdown, and the target cost source note. Types in
`client/src/energyTypes.ts`; hook `client/src/hooks/useEnergy.ts`.

---

## 5. Phases, deliverables, acceptance

### D1 — Sparse/event-driven runner

- **Deliverables:** `event_runtime/{__init__,spike_view,ops,counters,
  sparse_runner,dense_compare,errors}.py`.
- **Acceptance:** `sparse_run` on `conv_net` matches the dense readout within
  tolerance on a seeded fixture; on a low-density fixture `SOP < MAC` by the
  expected ratio; an unsupported stage kind raises a typed error, never a silent
  dense fallback.

### D2 — Energy/latency accounting

- **Deliverables:** `energy/{__init__,cost_table,target_costs,accounting,
  report,probe,errors}.py`, `energy/costs/*.json`.
- **Acceptance:** `account` maps op counts to energy/latency for a target with a
  table and reports `estimate: true`; a target without a table reports
  `basis: "unavailable"`, never a number; the report is JSON-able.

### D3 — Surfaces

- **Deliverables:** `energy/cli.py`, harness `--energy`, `server/energy_handlers.py`,
  `server/energy_payloads.py`, schema additions, `EnergyPanel.tsx`.
- **Acceptance:** `spikeforge-energy account` runs headless; `energy_report` round-trips
  over a live connection; the benchmark record carries the `energy` block; the
  client builds and renders; existing benchmark payload keys are unchanged.

---

## 6. Risks and deferred items

- **Fidelity of sparse ops:** conv/pool sparse implementations must reproduce the
  dense numerics exactly; the parity test is the safety net.
- **Energy credibility:** numbers are inherently estimates; the design makes
  that loud (`estimate`, `basis`, `source`) rather than hidden.
- **Training-free only:** the sparse path does not backpropagate; training stays
  dense. Stated explicitly so no one expects surrogate gradients from it.
- **Deferred:** measured on-device energy for Lava/Loihi (only when a device
  reports its own timing), event-driven *training*, and per-layer energy
  calibration against real silicon.

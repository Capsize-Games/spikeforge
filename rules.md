# Project Rules — spikeforge

## Project Overview

`spikeforge` is a spike-encoding playground for spiking neural networks
(SNNs), built on [snnTorch](https://snntorch.readthedocs.io/) and PyTorch. It
loads MNIST-style datasets, converts samples into **rate**, **latency**,
**delta**, and **random** spike codes, and drives them through a fully
connected LIF network for training, inference, and matplotlib/MP4 exports.

- `spikeforge/` — importable Python package: datasets, spike encoders,
  the FC LIF `SpikingNet`, the training loop, checkpointing, and exporters.
- `server/` — FastAPI backend that runs encoding/training and streams results
  to the client over a JSON WebSocket protocol.
- `main.py` / `main_encodings.py` — thin CLI entry points that run the encoder
  pipelines and write artifacts into `build/`.
- `client/` — the React + TypeScript dashboard (Vite + React 18).

### Design invariants

- **Single source of truth for encoding math.** `SpikeEncoder` owns
  rate/latency/delta/random conversion. The viewer and the training path both
  call it; nothing else re-implements coding logic. Datasets likewise flow
  through one shared `transform()` so normalization stays identical everywhere.
- **One class per file, grouped by subdirectory.** Small, focused modules are
  preferred over large multi-purpose ones.
- **Legacy compatibility is deliberate, not accidental.** The original trainer
  classes and the raw-pixel input path stay importable and behavior-compatible
  so old checkpoints and `main_encodings.py` keep working.

### Architecture invariants (Phase 1 spine)

1. **`TopologySpec` is the single source of truth.** Every model is declared
   as a spec and rendered twice — into the snnTorch module (`build_module`)
   and into the NIR graph (`to_nir`) — so the two cannot silently diverge.
2. **`nir`/`nirtorch` imports stay confined to `nir_bridge/api.py`.** Every
   other module reaches the packages through that isolated probe, which
   contains upstream API churn to one file.
3. **Exactly one temporal loop lives in `simulator/`.** Model `forward`s
   never contain a time loop; each stage is a pure `(input, state)` step and
   the single runner drives them over time.
4. **Validation stays independent.** The reference NIR interpreter executes
   the exported graph from its node parameters alone and never calls
   snnTorch, so a drift check is a genuine cross-check, not a tautology.
5. **Legacy compatibility is deliberate.** `fc_legacy` preserves the
   `SpikingNet` parameter names, method contracts, and state-dict keys, so
   old checkpoints keep loading and inferring unchanged.

---

## Public-Facing Copy

Website text, application text, documentation, and product descriptions follow
[`COPY_POLICY.md`](COPY_POLICY.md). Read it before drafting or editing any of
them. It governs the landing page under `landing/`, the model hub page rendered
by `scripts/build_hub_page.py`, `README.md`, and `docs/`.

Two points that bite most often here:

- **Do not invent quantities or capabilities.** Check the implemented behavior.
  A count on the landing page (walkthroughs, datasets, targets) is a claim, and
  claims get verified before they ship.
- **The landing page is translated into seventeen locales.** An English change
  that is not carried into `landing/locales.js` and `landing/cta-locales.js`
  leaves sixteen languages stating the old copy.

Required disclosures stay: the pre-1.0 warning, the "Implications and
boundaries" link, and the hub's curation, licensing, and
"not tuned attempts at state of the art" notices. Shortening copy is not a
reason to remove any of them.

---

## Python Code Style and Quality

### Hard limits

- No file over **250 lines**.
- No class over **200 lines**.
- No function or method over **20 lines**.
- **One class per file.** Group related classes into a subdirectory rather
  than stacking them in one module.
- Maximum line length: **79 characters** (PEP 8 / ruff `E501`).

When a limit is reached, split the logic — extract a helper function or move a
class/module into its own file. Do not grow a file just past the limit.

### Style

- Follow **PEP 8**. Lint and format with `ruff` (the declared dev dependency).
- **Use type hints** on every function and method signature, including return
  types. Annotate non-obvious attributes and properties too.
- Add **docstrings** to public modules, classes, and functions.
- Write **comments where they explain *why***, not what — skip comments that
  merely restate the code. Prefer clear names over narration.
- Write **sparse, DRY, easy-to-maintain code**. No dense, clever, or repeated
  logic: extract shared behavior into small helpers instead of copy-pasting it.
  Favor explicit, readable code over compact one-liners.

### Prohibited shortcuts

- **Never use `# noqa` / `# noqa: ...`** to silence linter warnings. Fix the
  underlying issue, and remove existing `noqa` comments while fixing what they
  masked.
- **Never use stopgaps, shims, hacks, or monkeypatches** to paper over a
  problem — this includes `type: ignore` comments, stub overrides, and runtime
  attribute injection. Implement the correct, first-class solution.

### Verification

- Run tests with `pytest` (with `pytest-cov` for coverage) from the `dev`
  extra before considering a change done.
- Keep static checks clean: `ruff` for lint/format and type annotations for
  the code you touch.

---

## TypeScript / TSX Code Style and Quality

### Hard limits

- No file over **250 lines**.
- No React component over **200 lines** (mirrors the class limit).
- No hook or helper function over **120 lines**.
- **One exported component per file.** Group related helpers into a
  subdirectory rather than stacking components in one module.
- Maximum line length: **80 characters**.

When a limit is reached, extract a hook, a helper, or a presentational
component into its own file. Do not grow a file just past the limit.

### Style

- Follow the existing **Prettier** conventions: double quotes, semicolons,
  trailing commas, and 2-space indentation.
- **Type everything.** `strict` is on; never fall back to `any`. Use a
  precise type, a discriminated union, or `unknown` plus a narrowing guard.
- Model protocol shapes as **discriminated unions** on a literal `type`
  field and narrow them with a `switch`.
- Keep stateful logic in **custom hooks** (`useSomething`) and keep
  components presentational.
- Prefer **derived values computed during render** over duplicated state, and
  prefer function components over class components.
- Declare props as a local `interface Props` and keep them read-only.
- Naming: `PascalCase` for components and types, `useSomething` for hooks,
  `camelCase` for values/functions, `UPPER_SNAKE_CASE` for module constants.
- Write **comments where they explain *why***, not what. Extract shared
  helpers instead of copy-pasting JSX or logic.

### Prohibited shortcuts

- **Never use `any` / `as any`, `@ts-ignore`, `@ts-expect-error`,
  `@ts-nocheck`, or `eslint-disable`.** Fix the underlying type instead.
- Do not paper over nullable values with non-null assertions; guard or narrow
  them. (The one allowed exception is the `#root` mount in `main.tsx`.)
- No stopgaps, shims, dead code, or unused exports.

### Verification

- `npm run build` (runs `tsc -b` then `vite build`) must pass with no type
  errors before a change is considered done.
- Keep every file within the limits above.

---

## CSS Style and Quality

### Hard limits

- No file over **300 lines**.
- Maximum line length: **80 characters**.
- Keep selectors flat and short — no descendant chains deeper than needed.

### Style

- **Define every colour as a custom property** in the theme blocks (`:root`
  and `[data-theme="light"]`); never hard-code a colour inside a rule.
- Group rules under a `/* --- section --- */` banner and keep related rules
  together, so the cascade order stays stable when a file is split.
- Class names are **kebab-case** and name the component or role, not the
  appearance.
- Prefer **class selectors**. Avoid IDs except the `#root` mount.
- Keep shared element defaults (e.g. `canvas`) in the base layer, and
  component-specific overrides after them.

### Prohibited shortcuts

- **Never use `!important`.** Fix the selector/specificity instead.
- No inline `style={{ ... }}` except values that genuinely cannot be
  expressed in CSS (e.g. JS-computed canvas sizes or positions).
- No unused rules.

### Verification

- The dashboard must still render correctly in both the dark and light
  themes; every colour change flows through a custom property.

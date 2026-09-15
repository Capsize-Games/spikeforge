# Ecosystem listings — getting spikeforge into the two directories this field searches

**Status:** both submitted on 2026-09-15, after the 0.3.5 release put the
corrected README and metadata on PyPI.

- NIR framework support table: [neuromorphs/NIR#196](https://github.com/neuromorphs/NIR/pull/196)
- Open Neuromorphic software guide: [open-neuromorphic.github.io#504](https://github.com/open-neuromorphic/open-neuromorphic.github.io/issues/504)

The NIR pull request deviates from the draft below in one place, deliberately.
The draft assumed a plain ✓ for *Read from NIR*. spikeforge loads a NIR graph
and **executes** it through the independent interpreter; it does not
reconstruct a native module the way the other listed frameworks do
(`nir_bridge/stage_builders.py` and `stage_mapping.py` are export-direction
only, and nothing builds a `TopologySpec` from a graph). The pull request
states that difference and invites the maintainers to mark the cell ⬚ if they
read the column as "convert to native" -- overclaiming a cell in the table
this project's positioning rests on would cost more than the row is worth.

## Why this is the promotion work worth doing

Searching for the *name* already works: "spikeforge spiking neural network"
returns the GitHub repository and all seven PyPI projects at the top. What
does not work is **categorical** discovery — someone browsing the NIR
ecosystem, or searching "SNN framework comparison", never encounters the
project. Two directories account for almost all of that traffic, both are
free, and both explicitly invite submissions.

This is deliberately sequenced **after** the PyPI and documentation fixes.
Reviewers of either submission will click straight through to
<https://pypi.org/project/spikeforge/> and <https://docs.spikeforge.net/>, so
those pages had to be right first.

---

## 1. NIR's framework support table

**Where:** <https://github.com/neuromorphs/NIR> — the README's framework
support table.

**Verified state (2026-09-14):** ten frameworks listed, with columns
*Framework*, *Write to NIR*, *Read from NIR*, *Examples*:

| Listed framework | Write | Read |
|---|---|---|
| hxtorch (BrainScaleS-2) | ✓ | ✓ |
| jaxsnn (BrainScaleS-2) | — | ✓ |
| Lava-DL | — | ✓ |
| Nengo | ✓ | ✓ |
| Norse | ✓ | ✓ |
| Rockpool (SynSense Xylo) | ✓ | ✓ |
| Sinabs (SynSense Speck) | ✓ | ✓ |
| snnTorch | ✓ | ✓ |
| SpiNNaker2 | — | ✓ |
| Spyx | ✓ | ✓ |

spikeforge is absent, despite supporting **both** directions. The README has
no documented contribution process for the table, so the route is a pull
request adding the row and a linked example.

**What makes the row worth accepting rather than just longer:** none of the
ten re-executes an exported graph through an *independent* interpreter and
reports numerical drift against the original model. That is a NIR-specific
capability, which is exactly the argument to make — not "please list us."

**Draft row:**

```markdown
| [spikeforge](https://github.com/Capsize-Games/spikeforge) | ✓ | ✓ | [spikeforge examples](https://github.com/Capsize-Games/spikeforge/blob/main/examples/04_nir_export_validate.py) |
```

**Draft PR description:**

> Adds spikeforge to the framework support table.
>
> spikeforge (BSD-3-Clause, `pip install spikeforge`) reads and writes NIR
> through `spikeforge.nir_bridge`. Beyond export/import it ships an
> independent NIR interpreter that re-executes the exported graph and reports
> per-tensor numerical drift against the source model, so an export can be
> *validated* rather than assumed:
>
> ```bash
> pip install "spikeforge[nir]"
> spikeforge-verify validate --topology conv_net   # non-zero exit outside tolerance
> ```
>
> Runnable example: `examples/04_nir_export_validate.py`.
> Docs: <https://docs.spikeforge.net/interpreter-spine>.

**Before filing, check:** that `examples/04_nir_export_validate.py` still runs
end to end on a clean install, and that the drift claim in
[Interpreter spine](interpreter-spine.md) still matches the shipped
tolerance.

---

## 2. Open Neuromorphic's software guide

**Where:** <https://open-neuromorphic.org/neuromorphic-computing/software/> —
the de-facto directory for this field.

**Verified state (2026-09-14):** 27 SNN frameworks, 8 data tools, 1
simulator. Neither "spikeforge" nor "Capsize" appears. The page states:

> "Help us keep the software guide comprehensive and up-to-date. Suggest new
> frameworks, data tools, or corrections by opening an issue on our GitHub
> repository."

**Route:** an issue at
<https://github.com/open-neuromorphic/open-neuromorphic.github.io/issues/new/choose>.

**Draft issue body:**

> **Framework:** spikeforge
> **Repository:** <https://github.com/Capsize-Games/spikeforge>
> **Docs:** <https://docs.spikeforge.net/>
> **Install:** `pip install spikeforge`
> **License:** BSD-3-Clause
> **Language / backend:** Python, built on snnTorch and PyTorch
>
> **What it is:** an SNN toolkit that sits on top of snnTorch rather than
> replacing it. Rate/latency/delta/random spike encoding, LIF training across
> eight image datasets and four neuromorphic event datasets, NIR export with
> an independent interpreter that re-executes the exported graph and reports
> numerical drift, a per-target deployment capability matrix, an event-driven
> energy estimator (labelled as an estimate, not hardware-measured), and a
> live browser training/introspection dashboard.
>
> **Maturity:** pre-1.0, published on PyPI, CI green, reference accuracy
> numbers published at <https://docs.spikeforge.net/benchmarks>.

**Before filing, check:** the maturity sentence is still true, and that the
benchmarks page is live at that URL.

---

## What not to do

Neither submission is a place to claim more than the project delivers. The
capability matrix's spec-only targets (`spinnaker2`, `speck`, `xylo`) are
registered but have no installable SDK integration; the energy figure is an
estimate; the model hub carries a handful of this project's own reference
checkpoints, not a zoo. Every one of those is stated plainly on the project's
own surfaces, and the submissions must not soften any of it — a listing
earned by overclaiming is worse than no listing.

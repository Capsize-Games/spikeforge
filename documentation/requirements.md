# Requirements

- Python 3.10-3.13 (matching `python_requires` and CI)
- `torch`, `torchvision`, `snntorch`
- `matplotlib`, `Pillow`, `numpy`
- Node.js 18+ and npm (for the `client/` dashboard)
- `ffmpeg` (only when exporting MP4s)

From a clone, `./install.sh` installs every distribution in editable mode.
Otherwise install the library bundle and, for the dashboard, the server
distribution:

```bash
pip install "spikeforge[all]"   # core library + targets + hub
pip install spikeforge-server   # FastAPI dashboard/WebSocket server
```

### Optional event datasets (Tonic)

Neuromorphic/event datasets (N-MNIST, DVS128 Gesture, CIFAR10-DVS, Spiking
Speech Commands) are powered by [Tonic](https://tonic.readthedocs.io/) and
gated behind the optional `events` extra so the default image stays lean:

```bash
pip install "spikeforge[events]"                 # from PyPI
pip install -e "./packages/spikeforge[events]"   # from a clone
```

`tonic` is deliberately kept out of `requirements.txt`; the event loader and
its capability probe work without it, reporting the event datasets as
unavailable and raising a clear typed error that names the missing extra
until it is installed.

### Optional extras

Every capability beyond the core is an opt-in extra; each has an isolated
probe, so a missing package is *reported* rather than raising at import. The
`all` extra bundles the `spikeforge-targets` and `spikeforge-hub`
distributions. The dashboard/WebSocket server is **not** a core extra: it
ships as its own distribution, `spikeforge-server`. The model hub is likewise
its own distribution, `spikeforge-hub` (import root `spikeforge_hub`), whose
`huggingface_hub` dependency is a base dependency of that distribution rather
than a core `hub` extra. `norse` and `lava` are extras of
`spikeforge-targets`, not of core.

| Extra | Enables | Absent behavior |
|---|---|---|
| `all` | the `spikeforge-targets` + `spikeforge-hub` bundles | not installed |
| `nir` | NIR export, interpretation, `nirtorch` extraction | typed unavailable error |
| `events` | Tonic event datasets (+ event training) | datasets reported unavailable |
| `onnx` | ONNX export/import bridge | typed unavailable error |
| `norse` | real Norse simulator backend (targets extra) | `norse` target `available: false` |
| `lava` | Lava/Loihi 2 backend path (targets extra) | `lava_loihi2` target `available: false` |
| `tracking` | TensorBoard sink | local manifest remains the default |
| `tracking-wandb` | Weights & Biases sink | local manifest remains the default |
| `docs` | MkDocs Material for the docs site | `build_docs.sh` reports the gap |
| `dev` | `pytest`, `pytest-cov`, `ruff` | — |

```bash
pip install "spikeforge[all]"                       # core + targets + hub
pip install "spikeforge[nir,events,onnx,tracking,docs]"
pip install "spikeforge-targets[norse,lava]"        # backend extras
pip install spikeforge-server                       # dashboard/server
# Editable equivalent from a clone:
./install.sh --dev
```

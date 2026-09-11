"""Environment-driven runtime settings."""

import os
from pathlib import Path

# Repo-local build directory (gitignored) used when not in Docker.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Environment variables use the SPIKEFORGE_ prefix canonically; the legacy SNN_
# prefix is still read as a fallback and the new name wins when both are set.

# Override with SPIKEFORGE_DATA_DIR (Docker sets this to /data).
DATA_DIR = (
    os.environ.get("SPIKEFORGE_DATA_DIR")
    or os.environ.get("SNN_DATA_DIR")
    or str(_REPO_ROOT / "build")
)

MNIST_PATH = os.path.join(DATA_DIR, "mnist")

# Settings: where saved model checkpoints live (separate from datasets).
# The model manager's Load/Save read and write this directory. Change it here
# (or set SPIKEFORGE_MODEL_DIR / SPIKEFORGE_DATA_DIR) to relocate saved models.
MODEL_DIR = (
    os.environ.get("SPIKEFORGE_MODEL_DIR")
    or os.environ.get("SNN_MODEL_DIR")
    or os.path.join(DATA_DIR, "models")
)

# Model hub: offline cache for downloaded or imported artifacts, kept separate
# from the trained-model store so a foreign artifact is only promoted after a
# compatibility check. Override with SPIKEFORGE_HUB_DIR (Docker keeps it in
# /data).
HUB_CACHE_DIR = (
    os.environ.get("SPIKEFORGE_HUB_DIR")
    or os.environ.get("SNN_HUB_DIR")
    or os.path.join(DATA_DIR, "hub")
)

# Registry governance (PT-W7): the signable promotion registry that records
# stage (dev/staging/prod), approver, and lineage for an artifact. Kept
# separate from the model store and the hub cache so a promotion ledger is
# never confused with the artifact bytes. Override with
# SPIKEFORGE_REGISTRY_DIR.
REGISTRY_DIR = (
    os.environ.get("SPIKEFORGE_REGISTRY_DIR")
    or os.environ.get("SNN_REGISTRY_DIR")
    or os.path.join(DATA_DIR, "registry")
)

# Persisted metrics: JSON snapshots of the in-process registry, written only
# when persistence is opted into (SPIKEFORGE_METRICS_PERSIST). The root
# defaults to DATA_DIR/metrics and is relocatable with SPIKEFORGE_METRICS_DIR.
METRICS_DIR = (
    os.environ.get("SPIKEFORGE_METRICS_DIR")
    or os.environ.get("SNN_METRICS_DIR")
    or os.path.join(DATA_DIR, "metrics")
)

# External tracking: local staging for optional sink output (e.g. TensorBoard
# event files). Override with SPIKEFORGE_TRACKING_DIR; the local checkpoint
# manifest remains the default and is always written first.
TRACKING_DIR = (
    os.environ.get("SPIKEFORGE_TRACKING_DIR")
    or os.environ.get("SNN_TRACKING_DIR")
    or os.path.join(DATA_DIR, "tracking")
)

# Pinned dashboard bundle. When SPIKEFORGE_DASHBOARD_DIST points at a built
# dashboard `dist/`, e.g. a versioned artifact published by
# capsize-games/spikeforge-dashboard, the server serves it instead of the
# in-repo `client/dist`. Leaving it unset keeps the legacy behaviour, so the
# read-only `client/` mirror still works for a release.
DASHBOARD_DIST = (
    os.environ.get("SPIKEFORGE_DASHBOARD_DIST")
    or os.environ.get("SNN_DASHBOARD_DIST")
    or None
)

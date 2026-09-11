"""Environment-driven runtime settings."""

import os
from pathlib import Path

# Repo-local build directory (gitignored) used when not in Docker.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Override with SNN_DATA_DIR (Docker sets this to /data).
DATA_DIR = os.environ.get("SNN_DATA_DIR", str(_REPO_ROOT / "build"))

MNIST_PATH = os.path.join(DATA_DIR, "mnist")

# Settings: where saved model checkpoints live (separate from datasets).
# The model manager's Load/Save read and write this directory. Change it here
# (or set SNN_MODEL_DIR / SNN_DATA_DIR) to relocate saved models.
MODEL_DIR = os.environ.get("SNN_MODEL_DIR", os.path.join(DATA_DIR, "models"))

# Model hub: offline cache for downloaded or imported artifacts, kept separate
# from the trained-model store so a foreign artifact is only promoted after a
# compatibility check. Override with SNN_HUB_DIR (Docker keeps it in /data).
HUB_CACHE_DIR = os.environ.get("SNN_HUB_DIR", os.path.join(DATA_DIR, "hub"))

# Persisted metrics: JSON snapshots of the in-process registry, written only
# when persistence is opted into (SNN_METRICS_PERSIST). The root defaults to
# DATA_DIR/metrics and is relocatable with SNN_METRICS_DIR.
METRICS_DIR = os.environ.get(
    "SNN_METRICS_DIR", os.path.join(DATA_DIR, "metrics")
)

# External tracking: local staging for optional sink output (e.g. TensorBoard
# event files). Override with SNN_TRACKING_DIR; the local checkpoint manifest
# remains the default and is always written first.
TRACKING_DIR = os.environ.get(
    "SNN_TRACKING_DIR", os.path.join(DATA_DIR, "tracking")
)

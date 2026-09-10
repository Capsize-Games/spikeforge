"""Environment-driven runtime settings."""

import os
from pathlib import Path

# Repo-local build directory (gitignored) used when not in Docker.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Override with SNN_DATA_DIR (Docker sets this to /data).
DATA_DIR = os.environ.get("SNN_DATA_DIR", str(_REPO_ROOT / "build"))

MNIST_PATH = os.path.join(DATA_DIR, "mnist")

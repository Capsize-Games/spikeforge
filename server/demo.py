"""Deployment policy for the public dashboard demo."""

import os

READ_ONLY_ENV = "SPIKEFORGE_DASHBOARD_READ_ONLY"


def read_only() -> bool:
    """Return whether training and model changes are disabled."""
    return os.environ.get(READ_ONLY_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }

"""The registry entry builder shared by the governance tests."""

from typing import Any, Dict

from spikeforge_hub import registry as reg

_KEY = "test-registry-key"


def _entry(**overrides: Any) -> reg.RegistryEntry:
    """Return a valid dev/staging entry with optional overrides."""
    data: Dict[str, Any] = {
        "id": "streaming/anomaly",
        "name": "Streaming anomaly detector",
        "stage": "dev",
        "status": "active",
        "version": "0.1.0",
        "lineage": {
            "dataset": "uc1-telemetry",
            "model": "uc1_anomaly",
            "bundle": "uc1.spkf",
        },
    }
    data.update(overrides)
    return reg.RegistryEntry.from_dict(data)

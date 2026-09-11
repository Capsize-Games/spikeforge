"""Isolated wrappers over the optional ``huggingface_hub`` package.

Every call reaches the package through :mod:`snn_interpreter.hub.probe`, so
``huggingface_hub`` is imported in exactly one place. A missing dependency
raises :class:`~snn_interpreter.hub.errors.HubExtraMissingError` naming the
``hub`` extra rather than surfacing a bare ``ImportError``.
"""

from snn_interpreter.hub import probe
from snn_interpreter.hub.errors import HubExtraMissingError


def available() -> bool:
    """Return True when live Hugging Face access is usable."""
    return probe.available()


def download_repo(repo_id: str, dest: str) -> str:
    """Download a repo snapshot into ``dest`` and return the local path.

    Raises :class:`HubExtraMissingError` when the ``hub`` extra is absent, so
    the caller reports the missing capability instead of failing obscurely.
    """
    hub = probe.module()
    if hub is None:
        raise HubExtraMissingError(repo_id)
    return str(hub.snapshot_download(repo_id=repo_id, local_dir=dest))

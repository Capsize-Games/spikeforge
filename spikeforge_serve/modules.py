"""Install, list, and remove local spikeforge model modules.

A "module" is a named, installed ``.spkf`` bundle plus a thin CLI wrapper
script: the same idea as a Linux kernel module, one name that takes one
input and produces one output. Once installed, a module runs by name
(``digit-classifier < input.json``) and its JSON output on stdout can be
piped into the next module, so several installed models chain together
without either one knowing the other exists.
"""

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

from spikeforge.serving.bundle import DeploymentBundle

#: Root directory installed modules live under; override for tests/containers.
MODULES_HOME = Path(
    os.environ.get("SPIKEFORGE_MODULES_HOME")
    or (Path.home() / ".local" / "share" / "spikeforge" / "modules")
)

#: Directory wrapper scripts are written to; must be on PATH to run by name.
BIN_HOME = Path(
    os.environ.get("SPIKEFORGE_MODULES_BIN")
    or (Path.home() / ".local" / "bin")
)

_BUNDLE_NAME = "model.spkf"
_DESCRIPTOR_NAME = "module.json"

_WRAPPER_TEMPLATE = """#!/usr/bin/env bash
# Installed by `spikeforge-serve install`. Do not edit by hand --
# reinstalling the module regenerates this file.
exec python3 -m spikeforge_serve run "{name}" "$@"
"""


def safe_name(name: str) -> str:
    """Return ``name`` restricted to filesystem- and shell-safe characters."""
    cleaned = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in name.strip()
    )
    if not cleaned:
        raise ValueError("module name must not be empty")
    return cleaned


def module_dir(name: str) -> Path:
    """Return the install directory for ``name`` (not guaranteed to exist)."""
    return MODULES_HOME / safe_name(name)


def descriptor_from_bundle(
    bundle: DeploymentBundle, name: str
) -> Dict[str, Any]:
    """Return the small, human-readable summary written as ``module.json``."""
    manifest = bundle.manifest
    return {
        "name": name,
        "topology": manifest.get("topology"),
        "num_classes": manifest.get("num_classes"),
        "input_mode": manifest.get("input_mode"),
        "coding": manifest.get("coding"),
        "num_steps": manifest.get("num_steps"),
        "label_map": dict(manifest.get("label_map") or {}),
        "created_at": manifest.get("created_at"),
    }


def _write_wrapper(name: str) -> Path:
    """Write (or overwrite) the executable ``name`` command in ``BIN_HOME``."""
    BIN_HOME.mkdir(parents=True, exist_ok=True)
    wrapper_path = BIN_HOME / name
    wrapper_path.write_text(_WRAPPER_TEMPLATE.format(name=name))
    mode = wrapper_path.stat().st_mode
    wrapper_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper_path


def install(
    bundle_path: str, name: Optional[str] = None, make_wrapper: bool = True
) -> Path:
    """Install ``bundle_path`` as a named module; return its directory.

    Loading the bundle first (``strict=True``) means a corrupt or
    incompatible ``.spkf`` is refused before anything is written, rather
    than leaving a half-installed module behind.
    """
    bundle = DeploymentBundle.load(bundle_path, strict=True)
    resolved = safe_name(name or Path(bundle_path).stem)
    target = module_dir(resolved)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_path, target / _BUNDLE_NAME)
    descriptor = descriptor_from_bundle(bundle, resolved)
    (target / _DESCRIPTOR_NAME).write_text(json.dumps(descriptor, indent=2))
    if make_wrapper:
        _write_wrapper(resolved)
    return target


def uninstall(name: str) -> bool:
    """Remove an installed module and its wrapper script.

    Returns True when the module existed (and was removed), False when
    there was nothing to do.
    """
    resolved = safe_name(name)
    target = module_dir(resolved)
    existed = target.exists()
    shutil.rmtree(target, ignore_errors=True)
    (BIN_HOME / resolved).unlink(missing_ok=True)
    return existed


def list_modules() -> List[Dict[str, Any]]:
    """Return every installed module's descriptor, sorted by name."""
    if not MODULES_HOME.exists():
        return []
    modules = []
    for entry in sorted(MODULES_HOME.iterdir()):
        descriptor_path = entry / _DESCRIPTOR_NAME
        if descriptor_path.exists():
            modules.append(json.loads(descriptor_path.read_text()))
    return modules


def bundle_path_for(name: str) -> str:
    """Return the installed bundle path for ``name``.

    Raises ``FileNotFoundError`` when no such module is installed, so a
    caller can surface one clear message instead of a bare missing-file
    error from deeper in the load path.
    """
    resolved = safe_name(name)
    path = module_dir(resolved) / _BUNDLE_NAME
    if not path.exists():
        raise FileNotFoundError(f"no installed module named {resolved!r}")
    return str(path)

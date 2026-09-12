"""HTTP download of a saved checkpoint as a portable ``.spkf`` bundle.

Bundling is a plain GET, not a WebSocket action: the artifact is a binary
file a browser should save via its own download UI, not a JSON payload one
more hop needs to relay.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from spikeforge.network import model_store
from spikeforge.serving import bundle as bundle_mod

router = APIRouter()


@router.get("/api/bundle/{name}")
async def download_bundle(name: str) -> FileResponse:
    """Build ``name``'s checkpoint into a ``.spkf`` bundle and return it."""
    saved = {entry["name"] for entry in model_store.list_models()}
    if name not in saved:
        raise HTTPException(404, detail=f"no saved model named {name!r}")

    workdir = tempfile.mkdtemp(prefix="spikeforge-bundle-")
    out_path = str(Path(workdir) / f"{name}.spkf")
    try:
        bundle_mod.build(name, out=out_path)
    except Exception as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, detail=str(exc)) from exc

    return FileResponse(
        out_path,
        media_type="application/octet-stream",
        filename=f"{name}.spkf",
        background=BackgroundTask(shutil.rmtree, workdir, ignore_errors=True),
    )

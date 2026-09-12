"""HTTP download of a saved checkpoint as a portable ``.spkf`` bundle.

Bundling is a plain GET, not a WebSocket action: the artifact is a binary
file a browser should save via its own download UI, not a JSON payload one
more hop needs to relay.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from server.auth import authorized
from spikeforge.network import model_store
from spikeforge.serving import bundle as bundle_mod

router = APIRouter()


@router.get("/api/bundle/{name}")
async def download_bundle(
    name: str,
    token: Optional[str] = None,
    request: Request = None,
) -> FileResponse:
    """Build ``name``'s checkpoint into a ``.spkf`` bundle and return it.

    Gated by ``SPIKEFORGE_DASHBOARD_TOKEN`` (see ``server/auth.py``) via a
    ``token`` query param -- what the dashboard's plain ``<a download>``
    link can send -- or an ``Authorization: Bearer`` header for other
    clients. ``request`` is a real ``Request`` when routed through the app
    (FastAPI special-cases the bare ``Request`` annotation and injects it
    regardless of the default) and None when a test calls this coroutine
    directly (see ``tests/test_server_bundle_route.py``).
    """
    header = request.headers.get("authorization") if request else None
    if not authorized(token, header):
        raise HTTPException(401, detail="missing or invalid token")
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

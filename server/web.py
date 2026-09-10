"""Serve the built React client from FastAPI for single-port deploys."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# In Docker this is /app/client/dist; local fallback is client/dist.
_CLIENT_DIST = Path("/app/client/dist")
_LOCAL_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


def client_dist() -> Optional[Path]:
    """Return the client build directory if present, else None."""
    if _CLIENT_DIST.exists():
        return _CLIENT_DIST
    if _LOCAL_DIST.exists():
        return _LOCAL_DIST
    return None


def mount_client(app: FastAPI) -> None:
    """Serve static assets and an SPA fallback for the client build."""
    dist = client_dist()
    if dist is None:
        return
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    index = dist / "index.html"

    @app.get("/")
    async def index_page() -> FileResponse:
        # Always revalidate the HTML so a fresh build is picked up right
        # away; the content-hashed assets under /assets can stay cached.
        return FileResponse(
            index, headers={"Cache-Control": "no-cache, must-revalidate"}
        )

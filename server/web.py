"""Serve the built React client from FastAPI for single-port deploys."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# In Docker this is /app/client/dist; local fallback is client/dist.
_CLIENT_DIST = Path("/app/client/dist")
_LOCAL_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


def client_dist() -> Path:
    """Return the client build directory if present, else None."""
    return _CLIENT_DIST if _CLIENT_DIST.exists() else (
        _LOCAL_DIST if _LOCAL_DIST.exists() else None
    )


def mount_client(app: FastAPI):
    """Serve static assets and an SPA fallback for the client build."""
    dist = client_dist()
    if dist is None:
        return
    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")
    app.get("/")(lambda: FileResponse(dist / "index.html"))

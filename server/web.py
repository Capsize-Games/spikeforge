"""Serve the built React client from FastAPI for single-port deploys."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from spikeforge import config

# In Docker this is /app/client/dist; local fallback is client/dist.
_CLIENT_DIST = Path("/app/client/dist")
_LOCAL_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


def pinned_dist() -> Optional[Path]:
    """Return the pinned prebuilt dashboard bundle when it is present."""
    if config.DASHBOARD_DIST:
        candidate = Path(config.DASHBOARD_DIST)
        if candidate.is_dir():
            return candidate
    return None


def client_dist() -> Optional[Path]:
    """Return the dashboard build directory, preferring a pinned bundle.

    ``SPIKEFORGE_DASHBOARD_DIST`` (see :mod:`spikeforge.config`) lets a deploy
    serve a ``dist/`` published by
    ``capsize-games/spikeforge-dashboard`` without rebuilding ``client/``
    in-repo. When it is unset or absent, the legacy resolution --
    the Docker ``/app/client/dist`` then the in-repo ``client/dist`` -- is
    unchanged.
    """
    pinned = pinned_dist()
    if pinned is not None:
        return pinned
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

    # Root-level SEO files are outside /assets so crawlers and social
    # preview fetchers can reach them at their conventional URLs.
    for filename in ("robots.txt", "sitemap.xml", "spikeforge-social.png"):
        file_path = dist / filename
        if not file_path.is_file():
            continue

        async def static_file(path: Path = file_path) -> FileResponse:
            return FileResponse(path)

        app.add_api_route(
            f"/{filename}",
            static_file,
            methods=["GET"],
            include_in_schema=False,
        )

    index = dist / "index.html"

    @app.get("/")
    async def index_page() -> FileResponse:
        # Always revalidate the HTML so a fresh build is picked up right
        # away; the content-hashed assets under /assets can stay cached.
        return FileResponse(
            index, headers={"Cache-Control": "no-cache, must-revalidate"}
        )

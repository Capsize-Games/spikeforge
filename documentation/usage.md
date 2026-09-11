# Usage

### 🐳 Run everything with Docker (recommended)

The whole stack — React dashboard, FastAPI server, WebSocket streaming —
builds into a single container and is served from **one port**.

```bash
docker compose up --build
```

Or use the dedicated runner, which selects the CPU/GPU image, builds it,
waits for the server to report healthy, and follows the logs:

```bash
scripts/docker_server.sh            # build + run the CUDA image (default)
scripts/docker_server.sh --cpu      # build + run the CPU-only image
scripts/docker_server.sh --gpu      # explicit CUDA build
scripts/docker_server.sh -d --follow  # run detached, then tail logs
scripts/docker_server.sh --port 9000  # serve the dashboard on :9000
```

Run `scripts/docker_server.sh --help` for every option.

Then open **http://localhost:8877**. The dashboard auto-connects to the
WebSocket on the same host/port (no separate backend or proxy to run).
Datasets download on first use into a Docker volume, so they persist across
restarts.

> Port 8877 was chosen to avoid clashing with other apps (e.g., 8000 is
> commonly used by other dev servers). To change it, edit the
> `ports:` mapping in [`docker-compose.yml`](../docker-compose.yml).

### Compute device & resources

The training panel has a **Device** dropdown (CPU / GPU) that defaults to
GPU and falls back to CPU automatically when CUDA is unavailable. A live
**System resources** panel shows host CPU RAM and GPU VRAM
(used / total / free). The Docker image installs CUDA PyTorch (`cu132`) and
requests the host GPU, so `docker compose up --build` trains on the GPU out
of the box; build a smaller CPU-only image with:

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

#### Docker profiles (CPU and GPU)

The default `docker compose up --build` is unchanged: it builds the CUDA
image and serves the dashboard on port 8877. Two opt-in
[profiles](../docker-compose.yml) select explicit builds of the same service —
neither is started by a plain `up`:

```bash
docker compose --profile cpu up --build   # CPU-only torch (smaller image)
docker compose --profile gpu up --build   # explicit CUDA torch build
```

Because profiles are standard Compose, `scripts/dev.sh docker-up` honours
`COMPOSE_PROFILES` too:

```bash
COMPOSE_PROFILES=cpu scripts/dev.sh docker-up
```

### Local (non-Docker) development

#### CLI exports (rate pipeline)

```bash
python main.py
```

### Additional encoding demos (latency / delta / random)

```bash
python main_encodings.py
```

Both write PNGs/GIFs/MP4s into `build/`.

### Local dev (server + Vite)

```bash
# terminal 1 - FastAPI on the canonical dev port :8877 (same port as Docker)
venv/bin/python -m server   # defaults to 8877 with reload

# terminal 2 - Vite (proxy target in client/vite.config.ts must match :8877)
cd client && npm install && npm run dev
```

Open the printed `http://localhost:5173` URL. Pick a coding type (rate,
latency, delta, or random), tune parameters, then hit **Apply & Run** to
stream spike frames into the raster and image panels in real time. The Vite
dev server proxies `/ws` to the FastAPI server on port 8877 (see
[`client/vite.config.ts`](../client/vite.config.ts) and
[`server/__main__.py`](../server/__main__.py)).

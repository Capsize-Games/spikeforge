# syntax=docker/dockerfile:1
# ---------- Stage 1: build the React client ----------
FROM node:22-alpine AS client-build
WORKDIR /build
COPY client/package.json client/package-lock.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

# ---------- Stage 2: python runtime ----------
FROM python:3.13-slim AS runtime

# Avoid interactive prompts / bytecode noise; keep logs visible.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs for matplotlib/Pillow etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the repository so the server-chain distributions can be installed from
# their `packages/` pyproject.toml files. The core, targets, hub, serve, and
# server packages all resolve their import roots (`spikeforge/`, `server/`,
# `main*.py`) from the repo root.
#
# The install list below must be `spikeforge-server`'s *complete* local
# dependency closure. A distribution left out does not fail loudly -- pip
# quietly resolves it from PyPI instead, so the image would run a mix of this
# checkout and whatever was last published. `test_dockerfile_installs_the_
# server_dependency_closure` enforces the closure.
COPY . .

# Install the shared core runtime pins first (better layer caching).
# PyTorch build selector: default CUDA so the GPU device option works.
# For a smaller CPU-only image:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
RUN pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} torch torchvision \
    && pip install -r requirements.txt \
    && pip install ./packages/spikeforge \
        ./packages/spikeforge-targets \
        ./packages/spikeforge-hub \
        ./packages/spikeforge-serve \
        ./packages/spikeforge-server

# Built client bundle (the Dockerfile's own copy overrides client/dist).
COPY --from=client-build /build/dist /app/client/dist

# The exporter output and dataset cache live in /data (mounted in compose).
ENV SPIKEFORGE_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8877

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8877"]

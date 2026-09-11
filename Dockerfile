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

# Copy the repository so the two distributions can be installed from their
# `packages/` pyproject.toml files. Both resolve their import roots
# (`snn_interpreter/`, `server/`, `main*.py`) from the repo root.
COPY . .

# Install the shared core runtime pins first (better layer caching).
# PyTorch build selector: default CUDA so the GPU device option works.
# For a smaller CPU-only image:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
RUN pip install --upgrade pip \
    && pip install --index-url ${TORCH_INDEX_URL} torch torchvision \
    && pip install -r requirements.txt \
    && pip install ./packages/snn-interpreter \
        ./packages/snn-interpreter-server

# Built client bundle (the Dockerfile's own copy overrides client/dist).
COPY --from=client-build /build/dist /app/client/dist

# The exporter output and dataset cache live in /data (mounted in compose).
ENV SNN_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8877

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8877"]

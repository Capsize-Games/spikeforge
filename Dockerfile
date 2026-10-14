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
# dependency closure. The local distributions install with `--no-deps`,
# so a distribution left out cannot be quietly resolved from PyPI;
# instead `pip check` fails the build on the missing dependency, and the
# image never runs a mix of this checkout and whatever was last
# published. `test_install_lists_cover_the_server_dependency_closure`
# enforces the closure statically.
COPY . .

# Third-party closure: installed from a hashed lockfile, never floating
# pins, so two image builds install byte-identical versions. LOCKFILE
# must match TORCH_INDEX_URL (server.cpu.lock for the CPU index,
# server.cu132.lock for cu132): torch/torchvision carry a +cpu/+cu132
# local version segment, and the grep below fails the build on a
# mismatched pair instead of installing the wrong torch. For a smaller
# CPU-only image:
#   docker compose --profile cpu build
#
# Lockfile cadence: recompile monthly and on any CVE in the closure
# (see requirements/server.cpu.in), then copy the public versions back
# into requirements.txt so dev installs track the image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG LOCKFILE=requirements/server.cu132.lock
RUN pip install --upgrade pip \
    && grep -qF -- "--index-url ${TORCH_INDEX_URL}" ${LOCKFILE} \
    && pip install --require-hashes -r ${LOCKFILE} \
    && pip install --no-deps ./packages/spikeforge \
        ./packages/spikeforge-targets \
        ./packages/spikeforge-hub \
        ./packages/spikeforge-serve \
        ./packages/spikeforge-server \
    && pip check

# Built client bundle (the Dockerfile's own copy overrides client/dist).
COPY --from=client-build /build/dist /app/client/dist

# The exporter output and dataset cache live in /data (mounted in compose).
ENV SPIKEFORGE_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8877

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8877"]

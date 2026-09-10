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

# Install Python deps first (better layer caching).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy app source + built client.
COPY . .
COPY --from=client-build /build/dist /app/client/dist

# The exporter output and dataset cache live in /data (mounted in compose).
ENV SNN_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8877

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8877"]

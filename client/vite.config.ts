import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /ws, /health, and /api to the FastAPI dev server.
// Defaults to :8877, which is the port server/__main__.py (and the Docker
// mapping) uses. Override with SPIKEFORGE_DEV_BACKEND so a dev session can
// attach to a backend already running on another port — for example the
// end-to-end server on :8899 — without editing this file.
const backend = process.env.SPIKEFORGE_DEV_BACKEND ?? "127.0.0.1:8877";
const httpTarget = `http://${backend}`;
const wsTarget = `ws://${backend}`;

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    watch: {
      // `node_modules` is a symlink into the main checkout; following it makes
      // the watcher consume a file descriptor per module and hits EMFILE.
      // The generated and cache directories are not source, so they are not
      // watched either.
      followSymlinks: false,
      ignored: [
        "**/node_modules/**",
        "**/.git/**",
        "**/dist/**",
        "**/.e2e-data/**",
        "**/.headlesscode/**",
        "**/.worktrees/**",
        "**/.claude/**",
      ],
    },
    proxy: {
      "/ws": { target: wsTarget, ws: true },
      "/health": { target: httpTarget },
      "/api": { target: httpTarget },
    },
  },
});

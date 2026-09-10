import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /ws and /health to the FastAPI dev server on :8877.
// Must match the port in server/__main__.py (and the Docker mapping).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8877", ws: true },
      "/health": { target: "http://127.0.0.1:8877" },
    },
  },
});

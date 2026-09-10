import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxy /ws and /health to the FastAPI dev server on :8000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/health": { target: "http://127.0.0.1:8000" },
    },
  },
});

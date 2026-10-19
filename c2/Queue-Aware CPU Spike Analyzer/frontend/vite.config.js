import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output goes to ./dist, which the Python backend
// (research_framework/realtime_experiment.py) serves directly.
// Dev server proxies /api to the running backend.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://localhost:8081",
    },
  },
  preview: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8081",
    },
  },
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to `bsmapper serve` (127.0.0.1:8765).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1000 },
});

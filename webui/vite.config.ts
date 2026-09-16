import { defineConfig } from "vite";
import react, { reactCompilerPreset } from "@vitejs/plugin-react";
import babel from "@rolldown/plugin-babel";
import path from "node:path";

// https://vite.dev/config/
export default defineConfig({
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy: { "/api": { target: "http://127.0.0.1:8787", changeOrigin: true } } },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy: { "/api": { target: "http://127.0.0.1:8787", changeOrigin: true } } },
  plugins: [react(), babel({ presets: [reactCompilerPreset()] })],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});

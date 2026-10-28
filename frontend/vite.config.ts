import { readFileSync } from "node:fs";
import { join } from "node:path";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// The microphone AudioWorklet and its downsampler are shared with the classic UI in ../web, so
// there is one copy of the audio code that the Node tests in web/test already cover. They must be
// served as plain files (an AudioWorklet loads its module by URL), so this plugin serves them in
// development and emits them next to index.html in a build.
const SHARED = ["pcm-worklet.js", "downsampler.js"];
const sharedDir = join(import.meta.dirname, "..", "web");

function sharedAudioWorklet(): Plugin {
  return {
    name: "shared-audio-worklet",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const name = request.url?.split("?")[0]?.slice(1) ?? "";
        if (!SHARED.includes(name)) return next();
        response.setHeader("Content-Type", "text/javascript");
        response.end(readFileSync(join(sharedDir, name)));
      });
    },
    generateBundle() {
      for (const fileName of SHARED) {
        this.emitFile({ type: "asset", fileName, source: readFileSync(join(sharedDir, fileName)) });
      }
    },
  };
}

// In development the page is served by Vite and the API by the Python server on :8000.
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  base: "./",
  plugins: [react(), sharedAudioWorklet()],
  server: {
    proxy: {
      "/api": backend,
      "/health": backend,
      "/ready": backend,
      "/ws": { target: backend, ws: true },
    },
  },
  build: { target: "es2022", sourcemap: false },
});

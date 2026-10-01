import { defineConfig } from "vite";
export default defineConfig({
  base: process.env.SITE_BASE || "/PhiRIE/",
  build: { target: "es2022", chunkSizeWarningLimit: 2500 },
  server: { host: "127.0.0.1" },
  preview: { host: "127.0.0.1" },
});

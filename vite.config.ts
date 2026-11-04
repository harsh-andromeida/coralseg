
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import process from "node:process";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],

  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },

  clearScreen: false,

  server: {
    port: 1425,
    strictPort: true,

    // Allow access from Google Colab and Cloudflare
    host: "0.0.0.0",

    // Allow Cloudflare temporary domains
    allowedHosts: [".trycloudflare.com"],

    // Forward API requests to Python backend
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
      },
    },

    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
      "Pragma": "no-cache",
    },

    // Cloudflare-compatible HMR
    hmr: host
      ? {
        protocol: "ws",
        host,
        port: 1426,
      }
      : {
        protocol: "wss",
        clientPort: 443,
      },

    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
}));

// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";
import viteTsconfigPaths from "vite-tsconfig-paths";

import packageJson from "./package.json";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [tailwindcss(), react(), viteTsconfigPaths()],
  define: {
    "import.meta.env.VITE_APP_VERSION": JSON.stringify(packageJson.version),
  },
  resolve: {
    alias: {
      "@/": path.resolve(__dirname, "./src/"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          "intel-chatbot-toolkit-ui-auth": ["@intel-chatbot-toolkit-ui/auth"],
          "intel-chatbot-toolkit-ui-components": [
            "@intel-chatbot-toolkit-ui/components",
          ],
          "intel-chatbot-toolkit-ui-control-plane": [
            "@intel-chatbot-toolkit-ui/control-plane",
          ],
          "intel-chatbot-toolkit-ui-icons": ["@intel-chatbot-toolkit-ui/icons"],
          "intel-chatbot-toolkit-ui-input-validation": [
            "@intel-chatbot-toolkit-ui/input-validation",
          ],
          "intel-chatbot-toolkit-ui-layouts": [
            "@intel-chatbot-toolkit-ui/layouts",
          ],
          "intel-chatbot-toolkit-ui-markdown": [
            "@intel-chatbot-toolkit-ui/markdown",
          ],
          "intel-chatbot-toolkit-ui-utils": ["@intel-chatbot-toolkit-ui/utils"],
          "docx-vendor": ["docx"],
        },
      },
    },
  },
  server: {
    proxy: {
      "^/api/v1/docsum": {
        target: "https://inteltoolkits.ai",
        changeOrigin: true,
        secure: false,
      },
    },
  },
});

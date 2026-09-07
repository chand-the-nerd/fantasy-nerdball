import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    // No manual chunking: the Form page is hidden, so nothing imports the
    // charting library and forcing it into a chunk would only emit dead code.
    // Re-enabling Form with a lazy import gets it split automatically.
  },
  server: {
    port: 5173,
    // `npm run dev` talks to the FastAPI process running on 8000.
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
});

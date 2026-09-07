import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    sourcemap: false,
    rollupOptions: {
      output: {
        // The charting library is only needed on the Form page; keeping it in
        // its own chunk stops it delaying the first paint of the pitch.
        manualChunks: { charts: ["recharts"] },
      },
    },
  },
  server: {
    port: 5173,
    // `npm run dev` talks to the FastAPI process running on 8000.
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true } },
  },
});

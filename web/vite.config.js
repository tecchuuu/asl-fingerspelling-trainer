import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, API calls go to FastAPI. 127.0.0.1, not localhost: on this machine
// "localhost" tries IPv6 first and adds ~2 s per request.
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: { "/api": API },
  },
});

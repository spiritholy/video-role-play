import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// dev server 的 /api 代理目标端口，与 run.py 的默认端口保持一致（VRP_PORT 可覆盖）。
// 后端若因端口占用自动避让到了别的端口，dev 模式下需显式 VRP_PORT=<端口> npm run dev。
const backendPort = process.env.VRP_PORT || "8770";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": `http://127.0.0.1:${backendPort}`,
    },
  },
});

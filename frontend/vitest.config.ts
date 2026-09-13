import { defineConfig } from "vitest/config";

// 独立的测试配置：不改 vite.config.ts，避免把测试依赖带进 pnpm run build。
// 只测 src/lib 下的纯函数，无需 DOM，因此 environment 用 node。
export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});

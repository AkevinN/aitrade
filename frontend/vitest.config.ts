import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// 前端测试配置（示例 / 快照测试，使用 jsdom 环境）。
// 与 vite.config.ts 分离，避免影响生产构建。
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    // 全量并行运行时模块导入/CPU 争用较高，重型 RTL 用例（如 CNNTrain 画像抽屉）在默认
    // 阈值下偶发整测超时；放宽 testTimeout 给足并发下的完成余量（不影响断言逻辑）。
    testTimeout: 30000,
  },
})

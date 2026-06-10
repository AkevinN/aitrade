// 全局测试环境初始化（jsdom）。
// 注册 @testing-library/jest-dom 的断言匹配器，并为 antd / rc-* 组件
// 在 jsdom 中缺失的浏览器 API 提供最小桩实现。
import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup, configure } from '@testing-library/react'

// 全量并行运行时模块导入/渲染耗时较高，重渲染的 findBy*/waitFor 在默认 1s 下偶发
// 超时（如 CNNTrain 等重型用例）。放宽 RTL 异步工具超时以稳定回归套件（不影响断言逻辑）。
configure({ asyncUtilTimeout: 5000 })

// 每个测试结束后卸载已渲染的组件，避免相互污染。
afterEach(() => {
  cleanup()
})

// antd 的响应式逻辑依赖 window.matchMedia，jsdom 未实现 -> 提供桩。
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // 已废弃但部分库仍调用
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  }),
})

// rc-resize-observer 等组件依赖 ResizeObserver，jsdom 未实现 -> 提供桩。
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }).ResizeObserver =
  ResizeObserverStub

// 某些 antd 组件在测量时调用 getComputedStyle().transform 等，jsdom 返回空对象即可。

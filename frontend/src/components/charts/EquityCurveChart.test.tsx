// 示例测试：EquityCurveChart 通用组件。
// recharts 的 ResponsiveContainer 在 jsdom 下因容器宽高为 0 往往不渲染图表内容，
// 故将 ResponsiveContainer mock 成固定宽高的容器，保证子图表能挂载渲染。
// 验证：
//   1. 给定非空 points → 正常渲染、不报错，且不展示空状态「暂无净值数据」（Req 4.4）
//   2. 空 points → 渲染 antd Empty「暂无净值数据」（Req 4.4）
// _Requirements: 4.4_

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

// ---- mock recharts：保留真实实现，仅将 ResponsiveContainer 替换为固定尺寸容器 ----
// jsdom 无布局，ResponsiveContainer 量到的宽高为 0 会跳过渲染子图表；
// 用固定尺寸 div 包裹，让内部 ComposedChart 能正常挂载。
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts')
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 800, height: 300 }}>{children}</div>
    ),
  }
})

import EquityCurveChart from './EquityCurveChart'
import type { EquityPoint } from './types'

const SAMPLE_POINTS: EquityPoint[] = [
  { date: '2025-03-04', balance: 100000, drawdown: 0, ddpercent: 0, netPnl: 0 },
  { date: '2025-03-05', balance: 101200, drawdown: 0, ddpercent: 0, netPnl: 1200 },
  { date: '2025-03-06', balance: 99800, drawdown: -1400, ddpercent: -1.38, netPnl: -1400 },
]

describe('EquityCurveChart', () => {
  it('给定非空 points 时正常渲染且不展示空状态', () => {
    // 渲染过程不应抛错
    expect(() => render(<EquityCurveChart points={SAMPLE_POINTS} />)).not.toThrow()
    // 有数据时不应出现「暂无净值数据」空状态
    expect(screen.queryByText('暂无净值数据')).not.toBeInTheDocument()
  })

  it('空 points 时渲染 antd Empty 占位（Req 4.4）', () => {
    render(<EquityCurveChart points={[]} />)
    expect(screen.getByText('暂无净值数据')).toBeInTheDocument()
  })
})

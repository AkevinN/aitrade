// DecisionResultCard 示例测试：覆盖决策结果展示（Req 6.3）与
// 「仅提醒，不自动下单」提示始终存在（Req 7.4）。
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import DecisionResultCard from './DecisionResultCard'
import { makeTask, makeResult, makeDecision } from './testFixtures'
import type { Task } from '../../types/alpha'

// DecisionResultCard 内嵌 DecisionTracePanel（useQuery），需提供 QueryClient。
// 面板默认折叠不触网，仅为满足 react-query 上下文。
function renderCard(task: Task | null) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <DecisionResultCard task={task} />
    </QueryClientProvider>,
  )
}

describe('DecisionResultCard', () => {
  // Req 7.4：无论是否已有决策结果，始终展示「仅提醒，不自动下单」提示。
  it('任务为空时仍展示「仅提醒，不自动下单」提示', () => {
    renderCard(null)
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
  })

  it('失败态展示失败占位，且仍保留安全提示', () => {
    const task = makeTask({ status: 'failed' })
    renderCard(task)
    expect(screen.getByText(/决策任务失败/)).toBeInTheDocument()
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
  })

  // Req 6.3：完成后展示 action / 手数 / 价位 / 概率 / reason。
  it('完成态买入决策展示完整字段', () => {
    const task = makeTask({ status: 'completed', result: makeResult() })
    const { container } = renderCard(task)

    expect(screen.getByText('买入')).toBeInTheDocument()
    expect(screen.getByText('信号超过买入阈值')).toBeInTheDocument()
    // antd Statistic 将数值拆分为多个 span，断言整体文本包含格式化结果。
    // 信号概率 0.82 → 82.00%
    expect(container.textContent).toContain('82.00')
    // 建议手数 1000（antd Statistic 默认千分位 → 1,000）
    expect(container.textContent).toContain('1,000')
    // signal_id 透出
    expect(screen.getByText('2026-06-08:eod_buy_v1:demo@v1')).toBeInTheDocument()
    // 安全提示仍在
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
  })

  it('观望决策展示 hold 文案与拦截原因', () => {
    const decision = makeDecision({ action: 'hold', reason: '黑名单拦截', signal: 0.4 })
    const task = makeTask({
      status: 'completed',
      result: makeResult({ decision }),
    })
    renderCard(task)
    expect(screen.getByText('观望')).toBeInTheDocument()
    expect(screen.getByText('黑名单拦截')).toBeInTheDocument()
  })

  it('幂等命中时展示幂等标记', () => {
    const task = makeTask({
      status: 'completed',
      result: makeResult({ idempotent_hit: true, risk_detail: [] }),
    })
    renderCard(task)
    expect(screen.getByText(/幂等命中/)).toBeInTheDocument()
  })
})

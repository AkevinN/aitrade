// TradingConsole 页面级示例测试：页面渲染、进度联动与「仅提醒不自动下单」提示存在性。
// 通过 mock useTask 与 api 服务保持确定性、离线（不触网）。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import type { UseQueryResult } from '@tanstack/react-query'

import type { Task } from '../../types/alpha'
import { makeTask, makeResult } from './testFixtures'

// useTask 由页面直接调用以订阅任务进度；mock 它返回受控的 task。
const taskState: { data: Task | null } = { data: null }
vi.mock('../../hooks/useTask', () => ({
  useTask: () => taskState as unknown as UseQueryResult<Task>,
}))

// ConfigForm 拉取 CNN 模型列表；HistoryTable / liveService 调用决策接口。Mock 为离线。
vi.mock('../../api/cnn', () => ({
  cnnService: { listModels: vi.fn().mockResolvedValue([]) },
}))
vi.mock('../../api/liveApi', () => ({
  liveService: {
    startDecision: vi.fn(),
    listDecisions: vi.fn().mockResolvedValue([]),
    getDecision: vi.fn(),
  },
}))

import TradingConsole from './index'

function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <TradingConsole />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('TradingConsole 页面', () => {
  beforeEach(() => {
    taskState.data = null
  })

  it('无任务时正常渲染标题与各区块', () => {
    renderPage()
    expect(screen.getByText('交易操作台')).toBeInTheDocument()
    expect(screen.getByText('决策配置')).toBeInTheDocument()
    expect(screen.getByText('任务进度')).toBeInTheDocument()
    expect(screen.getByText('决策结果')).toBeInTheDocument()
    expect(screen.getByText('风控明细')).toBeInTheDocument()
    expect(screen.getByText('历史决策')).toBeInTheDocument()
  })

  // Req 7.4：页面始终展示「仅提醒，不自动下单」提示。
  it('始终展示「仅提醒，不自动下单」提示', () => {
    renderPage()
    // 页面顶部告警 + 决策结果卡片提示，至少出现一次。
    expect(screen.getAllByText('仅提醒，不自动下单').length).toBeGreaterThanOrEqual(1)
  })

  // Req 6.2：完成态任务时，决策结果联动展示。
  it('任务完成时联动展示决策结果', () => {
    taskState.data = makeTask({ status: 'completed', result: makeResult() })
    renderPage()
    expect(screen.getByText('买入')).toBeInTheDocument()
    expect(screen.getByText('信号超过买入阈值')).toBeInTheDocument()
  })
})

// DecisionTracePanel 示例测试：六段可折叠分组渲染、懒加载（展开才请求 /trace）、
// 折叠/展开交互，以及 404 显示「暂无过程档案」。（Req 8.6）
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'

import { makeTrace } from './testFixtures'

// mock liveApi 服务，使测试离线且可断言调用次数（懒加载）。
const getDecisionTrace = vi.fn()
vi.mock('../../api/liveApi', () => ({
  liveService: {
    getDecisionTrace: (signalId: string) => getDecisionTrace(signalId),
  },
}))

import DecisionTracePanel from './DecisionTracePanel'

function renderPanel(signalId: string | null = '2026-06-08:eod_buy_v1:demo@v1') {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <DecisionTracePanel signalId={signalId} />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('DecisionTracePanel', () => {
  beforeEach(() => {
    getDecisionTrace.mockReset()
  })

  it('渲染六个分组标题且默认折叠', () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    renderPanel()

    // 六个分组标题（运行头/推理段/取价段/决策逻辑段/风控段/结果段）。
    expect(screen.getByText('运行头')).toBeInTheDocument()
    expect(screen.getByText('推理段')).toBeInTheDocument()
    expect(screen.getByText('取价段')).toBeInTheDocument()
    expect(screen.getByText('决策逻辑段')).toBeInTheDocument()
    expect(screen.getByText('风控段')).toBeInTheDocument()
    expect(screen.getByText('结果段')).toBeInTheDocument()
  })

  // 懒加载：默认折叠时不应发起 /trace 请求。
  it('默认折叠时不请求 /trace（懒加载）', () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    renderPanel()
    expect(getDecisionTrace).not.toHaveBeenCalled()
  })

  // 懒加载 + 内容渲染：展开后才请求并渲染该段内容。
  it('展开分组后才请求 /trace 并渲染段内容', async () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    const user = userEvent.setup()
    renderPanel()

    expect(getDecisionTrace).not.toHaveBeenCalled()

    await user.click(screen.getByText('运行头'))

    await waitFor(() => {
      expect(getDecisionTrace).toHaveBeenCalledTimes(1)
    })
    expect(getDecisionTrace).toHaveBeenCalledWith('2026-06-08:eod_buy_v1:demo@v1')

    // 运行头段内容渲染（run_id、数据源类型）。
    expect(await screen.findByText('abcd1234')).toBeInTheDocument()
    expect(screen.getByText('upload')).toBeInTheDocument()
  })

  it('风控段复用 RiskDetailPanel 渲染逐项检查', async () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByText('风控段'))

    // RiskDetailPanel 将 check 映射为中文标签。
    expect(await screen.findByText('Kill-switch / 熔断')).toBeInTheDocument()
    expect(screen.getByText('黑名单')).toBeInTheDocument()
  })

  it('结果段展示 idempotent_hit / trace_persisted / abort_reason', async () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByText('结果段'))

    expect(await screen.findByText(/idempotent_hit/)).toBeInTheDocument()
    expect(screen.getByText(/trace_persisted/)).toBeInTheDocument()
    expect(screen.getByText(/abort_reason/)).toBeInTheDocument()
  })

  // 折叠/展开交互：点击标题切换 aria-expanded（antd 折叠时内容保留在 DOM 但隐藏）。
  it('点击标题可展开再折叠分组', async () => {
    getDecisionTrace.mockResolvedValue(makeTrace())
    const user = userEvent.setup()
    renderPanel()

    const header = screen.getByText('取价段').closest('.ant-collapse-header') as HTMLElement
    expect(header).toHaveAttribute('aria-expanded', 'false')

    await user.click(screen.getByText('取价段'))
    await waitFor(() => {
      expect(header).toHaveAttribute('aria-expanded', 'true')
    })
    expect(await screen.findByText('取价周期')).toBeInTheDocument()

    await user.click(screen.getByText('取价段'))
    await waitFor(() => {
      expect(header).toHaveAttribute('aria-expanded', 'false')
    })
  })

  // Req 8.6：404（无过程档案）时显示「暂无过程档案」。
  it('404 时显示「暂无过程档案」', async () => {
    getDecisionTrace.mockRejectedValue({ response: { status: 404 } })
    const user = userEvent.setup()
    renderPanel()

    await user.click(screen.getByText('运行头'))

    expect(await screen.findByText('暂无过程档案')).toBeInTheDocument()
  })

  it('无 signalId 时给出占位且不请求', () => {
    renderPanel(null)
    expect(screen.getByText(/完成决策后将在此展示/)).toBeInTheDocument()
    expect(getDecisionTrace).not.toHaveBeenCalled()
  })
})

// SchedulerRunsCard 示例测试：事件列表渲染、reason 中文映射、计划过滤（TSO Wave 4 / R6.4）。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import type { SchedulerRunEvent } from '../../types/live'
import { makePlanSummary } from './testFixtures'

const mockGetSchedulerRuns = vi.fn()

vi.mock('../../api/liveApi', () => ({
  liveService: {
    getSchedulerRuns: (...args: unknown[]) => mockGetSchedulerRuns(...args),
  },
}))

import SchedulerRunsCard from './SchedulerRunsCard'

function makeEvent(overrides: Partial<SchedulerRunEvent> = {}): SchedulerRunEvent {
  return {
    ts: '2026-06-12T09:30:05+08:00',
    event: 'skip',
    plan_id: 'plan-1',
    reason: 'not_trading_day',
    detail: '',
    ...overrides,
  }
}

function renderCard(plans = [makePlanSummary()]) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <SchedulerRunsCard plans={plans} />
    </QueryClientProvider>,
  )
}

describe('SchedulerRunsCard', () => {
  beforeEach(() => {
    mockGetSchedulerRuns.mockClear()
  })

  it('渲染 skip 事件并显示中文原因（非交易日）', async () => {
    mockGetSchedulerRuns.mockResolvedValue([makeEvent()])
    renderCard()
    expect(await screen.findByText('skip')).toBeInTheDocument()
    // 原因中文映射 not_trading_day → 非交易日
    expect(screen.getByText('非交易日')).toBeInTheDocument()
  })

  it('渲染 trigger 事件并展示时点', async () => {
    mockGetSchedulerRuns.mockResolvedValue([
      makeEvent({ event: 'trigger', slot: '15:05', reason: undefined }),
    ])
    renderCard()
    expect(await screen.findByText('trigger')).toBeInTheDocument()
    expect(screen.getByText('时点 15:05')).toBeInTheDocument()
  })

  it('渲染 error 事件并展示错误摘要', async () => {
    mockGetSchedulerRuns.mockResolvedValue([
      makeEvent({ event: 'error', error: 'RuntimeError: 无信号', reason: undefined }),
    ])
    renderCard()
    expect(await screen.findByText('error')).toBeInTheDocument()
    expect(screen.getByText('RuntimeError: 无信号')).toBeInTheDocument()
  })

  it('各 skip reason 映射：schedule_gate/already_done/degraded/data_lag/disabled', async () => {
    const reasons = [
      { reason: 'schedule_gate', label: '未到调度日' },
      { reason: 'already_done', label: '今日已触发' },
      { reason: 'degraded', label: '降级暂停' },
      { reason: 'data_lag', label: '行情滞后' },
      { reason: 'disabled', label: '计划停用' },
    ]
    for (const { reason, label } of reasons) {
      mockGetSchedulerRuns.mockResolvedValue([makeEvent({ reason })])
      const { unmount } = renderCard()
      expect(await screen.findByText(label)).toBeInTheDocument()
      unmount()
    }
  })

  it('空数据显示「暂无调度日志」', async () => {
    mockGetSchedulerRuns.mockResolvedValue([])
    renderCard()
    expect(await screen.findByText('暂无调度日志')).toBeInTheDocument()
  })

  it('选择计划过滤时传入 plan_id 查询参数', async () => {
    mockGetSchedulerRuns.mockResolvedValue([])
    renderCard([makePlanSummary({ plan_id: 'plan-1', name: '尾盘买入计划' })])
    // 等待初始渲染完成
    await screen.findByText('暂无调度日志')

    // 点击 Select 并选择计划
    const select = screen.getByRole('combobox')
    fireEvent.mouseDown(select)
    const option = await screen.findByTitle('尾盘买入计划')
    fireEvent.click(option)

    await waitFor(() => {
      // 选择计划后，getSchedulerRuns 应被调用过，且某次调用携带 plan_id
      const calls = mockGetSchedulerRuns.mock.calls
      expect(calls.some((c) => c[0]?.plan_id === 'plan-1')).toBe(true)
    })
  })

  it('点击刷新按钮重新请求数据', async () => {
    mockGetSchedulerRuns.mockResolvedValue([makeEvent()])
    renderCard()
    await screen.findByText('skip')

    const refreshBtn = screen.getByRole('button', { name: /刷新/ })
    fireEvent.click(refreshBtn)

    await waitFor(() => {
      expect(mockGetSchedulerRuns).toHaveBeenCalledTimes(2)
    })
  })
})

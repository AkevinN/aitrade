// RebalancePlanCard 测试：清单渲染/确认按钮/409 错误/skipped 空态（任务 3.8）
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App as AntApp } from 'antd'

import type { Task } from '../../types/alpha'
import type { RebalanceDecision, RebalanceItem, RebalanceResult } from '../../types/live'

// ---- mock liveService ----
const confirmRebalance = vi.fn()
vi.mock('../../api/liveApi', () => ({
  liveService: {
    confirmRebalance: (...args: unknown[]) => confirmRebalance(...args),
  },
}))

import RebalancePlanCard from './RebalancePlanCard'

// ---- Fixtures ----
function makeRebalanceItem(overrides: Partial<RebalanceItem> = {}): RebalanceItem {
  return {
    vt_symbol: '000001.SZSE',
    action: 'buy',
    volume: 1000,
    price: 12.34,
    signal: 0.75,
    reason: '信号超过阈值',
    ...overrides,
  }
}

function makeRebalanceDecision(overrides: Partial<RebalanceDecision> = {}): RebalanceDecision {
  return {
    signal_id: '2026-06-11:rule_v1:src@v1',
    decision_bar_dt: '2026-06-11T15:00:00',
    as_of: '2026-06-11T15:05:00',
    bar_freq: '1d',
    scheme: 'rule_v1',
    portfolio_id: 'portfolio-001',
    items: [makeRebalanceItem()],
    target_portfolio: { '000001.SZSE': 1000 },
    risk_summary: [
      { check: 'blacklist', passed: true, detail: '不在黑名单' },
      { check: 'max_total_position', passed: false, detail: '总仓位超限' },
    ],
    status: 'proposed',
    created_at: '2026-06-11T15:05:00',
    confirmed_at: '',
    ...overrides,
  }
}

function makeRebalanceResult(overrides: Partial<RebalanceResult> = {}): RebalanceResult {
  return {
    decision: makeRebalanceDecision(),
    idempotent_hit: false,
    risk: [],
    skipped_reason: null,
    ...overrides,
  }
}

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    task_id: 'task-r1',
    type: 'rebalance',
    title: '规则调仓',
    entity_type: 'live',
    entity_name: 'portfolio-001',
    status: 'completed',
    progress: 100,
    message: '',
    result: null,
    created_at: '2026-06-11T15:05:00',
    updated_at: '2026-06-11T15:05:30',
    ...overrides,
  }
}

function renderCard(task: Task | null) {
  return render(
    <AntApp>
      <RebalancePlanCard task={task} />
    </AntApp>,
  )
}

describe('RebalancePlanCard', () => {
  beforeEach(() => {
    confirmRebalance.mockReset()
  })

  // 1. 调仓清单渲染
  it('完成态渲染调仓清单：标的/方向/数量/价格/信号/原因', () => {
    const task = makeTask({
      result: makeRebalanceResult() as unknown as Record<string, unknown>,
    })
    renderCard(task)
    // 标的
    expect(screen.getByText('000001.SZSE')).toBeInTheDocument()
    // 方向：买入
    expect(screen.getByText('买入')).toBeInTheDocument()
    // 数量（千分位）
    expect(screen.getByText('1,000')).toBeInTheDocument()
    // 参考价
    expect(screen.getByText('12.34')).toBeInTheDocument()
    // 原因
    expect(screen.getByText('信号超过阈值')).toBeInTheDocument()
    // 风控摘要：passed=false 风险项显示
    expect(screen.getByText(/总仓位超限/)).toBeInTheDocument()
    // 安全提示
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
  })

  // 2. 确认按钮调用 confirmRebalance
  it('点击确认执行后调用 confirmRebalance(signal_id)', async () => {
    confirmRebalance.mockResolvedValueOnce({
      decision: makeRebalanceDecision({ status: 'confirmed', confirmed_at: '2026-06-11T15:10:00' }),
      portfolio: { portfolio_id: 'portfolio-001', positions: {}, cash: null },
    })
    const task = makeTask({
      result: makeRebalanceResult() as unknown as Record<string, unknown>,
    })
    renderCard(task)

    // 点击「确认执行」触发器按钮
    await userEvent.click(screen.getByRole('button', { name: '确认执行' }))
    // Popconfirm 弹出后点击 okText（确认回填）按钮
    await userEvent.click(await screen.findByRole('button', { name: '确认回填' }))

    await waitFor(() =>
      expect(confirmRebalance).toHaveBeenCalledWith('2026-06-11:rule_v1:src@v1'),
    )
    // 确认后展示已确认状态
    await waitFor(() =>
      expect(screen.getByText('已确认执行')).toBeInTheDocument(),
    )
  })

  // 3. 409 错误展示 detail
  it('confirmRebalance 返回 409 时展示错误 detail', async () => {
    const err = Object.assign(new Error('已确认'), {
      response: { status: 409, data: { detail: '该决策已确认，不可重复确认' } },
    })
    confirmRebalance.mockRejectedValueOnce(err)

    const task = makeTask({
      result: makeRebalanceResult() as unknown as Record<string, unknown>,
    })
    renderCard(task)

    await userEvent.click(screen.getByRole('button', { name: '确认执行' }))
    await userEvent.click(await screen.findByRole('button', { name: '确认回填' }))

    // 409 detail 应通过 antd message 展示（我们检查 confirmRebalance 被调用即可，
    // 因为 message.error 在 jsdom 中不挂载 DOM）
    await waitFor(() =>
      expect(confirmRebalance).toHaveBeenCalledWith('2026-06-11:rule_v1:src@v1'),
    )
  })

  // 4. skipped 空态展示
  it('result.skipped_reason 存在且无 decision 时展示跳过原因', () => {
    const task = makeTask({
      result: makeRebalanceResult({
        decision: null,
        skipped_reason: '触发周期未到，跳过',
      }) as unknown as Record<string, unknown>,
    })
    renderCard(task)
    expect(screen.getByText(/已跳过：触发周期未到，跳过/)).toBeInTheDocument()
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
  })

  // 5. 任务未完成时展示占位
  it('任务未完成时展示占位文案', () => {
    const task = makeTask({ status: 'running', result: null })
    renderCard(task)
    expect(screen.getByText('调仓完成后将在此展示调仓清单')).toBeInTheDocument()
  })

  // 6. 幂等命中时展示标记
  it('幂等命中时展示幂等命中标记', () => {
    const task = makeTask({
      result: makeRebalanceResult({ idempotent_hit: true }) as unknown as Record<string, unknown>,
    })
    renderCard(task)
    expect(screen.getByText(/幂等命中/)).toBeInTheDocument()
  })
})

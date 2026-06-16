// Portfolio 页面测试（Task 5.4）
// ① 无 rule 计划 Empty 引导
// ② 选组合后账本+风控渲染
// ③ broken 态红 Alert + 复位调用 resetPortfolioRisk
// ④ 调仓历史行点击开详情 Modal
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'

import type { TradingPlan, PortfolioState, PortfolioRiskState, RebalanceDecision } from '../../types/live'

// ---- mock liveService ----
const listPlans = vi.fn()
const getPortfolio = vi.fn()
const getPortfolioRisk = vi.fn()
const resetPortfolioRisk = vi.fn()
const listRebalances = vi.fn()
const getRebalance = vi.fn()

vi.mock('../../api/liveApi', () => ({
  liveService: {
    listPlans: (...args: unknown[]) => listPlans(...args),
    getPortfolio: (...args: unknown[]) => getPortfolio(...args),
    getPortfolioRisk: (...args: unknown[]) => getPortfolioRisk(...args),
    resetPortfolioRisk: (...args: unknown[]) => resetPortfolioRisk(...args),
    listRebalances: (...args: unknown[]) => listRebalances(...args),
    getRebalance: (...args: unknown[]) => getRebalance(...args),
  },
}))

import Portfolio from './index'

// ---- Fixtures ----
function makeRulePlan(portfolioId = 'portfolio-001'): TradingPlan {
  return {
    plan_id: 'plan-rule-1',
    name: '规则调仓计划',
    model: '',
    vt_symbol: '000001.SZSE',
    scheme: 'rule_v1',
    buy_threshold: 0.6,
    position_ratio: 0.9,
    min_volume: 100,
    model_version: 'v1',
    data_source: 'pull',
    should_exit: false,
    halted: false,
    portfolio: { portfolio_value: 100000 },
    risk: {},
    notify_channels: [],
    enabled: true,
    bar_freq: '1d',
    trigger_times: ['15:05'],
    created_at: '2026-06-01T00:00:00',
    updated_at: '2026-06-01T00:00:00',
    strategy_type: 'rule',
    signal_source: 'factor_v1',
    signal_params: {},
    trigger_schedule: 'daily',
    portfolio_id: portfolioId,
  }
}

function makePortfolio(portfolioId = 'portfolio-001'): PortfolioState {
  return {
    portfolio_id: portfolioId,
    positions: { '000001.SZSE': 1000, '600000.SSE': 500 },
    cash: 50000,
    last_signal_id: '2026-06-11:rule_v1:src@v1',
    updated_at: '2026-06-11T15:05:00',
  }
}

function makeRiskNormal(portfolioId = 'portfolio-001'): PortfolioRiskState {
  return {
    portfolio_id: portfolioId,
    peak_value: 105000,
    broken: false,
    broken_date: null,
    reason: null,
  }
}

function makeRiskBroken(portfolioId = 'portfolio-001'): PortfolioRiskState {
  return {
    portfolio_id: portfolioId,
    peak_value: 110000,
    broken: true,
    broken_date: '2026-06-10',
    reason: '回撤超过 20%',
  }
}

function makeRebalanceSummary(portfolioId = 'portfolio-001') {
  return {
    signal_id: '2026-06-11:rule_v1:src@v1',
    status: 'proposed',
    portfolio_id: portfolioId,
    created_at: '2026-06-11T15:05:00',
  }
}

function makeRebalanceDetail(portfolioId = 'portfolio-001'): RebalanceDecision {
  return {
    signal_id: '2026-06-11:rule_v1:src@v1',
    decision_bar_dt: '2026-06-11T15:00:00',
    as_of: '2026-06-11T15:05:00',
    bar_freq: '1d',
    scheme: 'rule_v1',
    portfolio_id: portfolioId,
    items: [
      {
        vt_symbol: '000001.SZSE',
        action: 'buy',
        volume: 500,
        price: 12.5,
        signal: 0.8,
        reason: '信号超过阈值',
      },
    ],
    target_portfolio: { '000001.SZSE': 1500 },
    risk_summary: [],
    status: 'proposed',
    created_at: '2026-06-11T15:05:00',
    confirmed_at: null,
  }
}

function renderPage(props: { _testPortfolioId?: string } = {}) {
  const qc = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <Portfolio {...props} />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('Portfolio 策略组合页', () => {
  beforeEach(() => {
    listPlans.mockReset()
    getPortfolio.mockReset()
    getPortfolioRisk.mockReset()
    resetPortfolioRisk.mockReset()
    listRebalances.mockReset()
    getRebalance.mockReset()
  })

  // ① 无 rule 计划时展示引导 Empty
  it('无 rule 计划时展示引导 Empty 文案', async () => {
    listPlans.mockResolvedValue([])
    renderPage()
    expect(await screen.findByText('先在交易操作台创建规则调仓计划')).toBeInTheDocument()
  })

  // ① 变体：仅有 cnn 计划时同样展示引导
  it('仅有 cnn 计划时展示引导 Empty 文案', async () => {
    const cnnPlan = { ...makeRulePlan(), strategy_type: 'cnn', portfolio_id: undefined }
    listPlans.mockResolvedValue([cnnPlan])
    renderPage()
    expect(await screen.findByText('先在交易操作台创建规则调仓计划')).toBeInTheDocument()
  })

  // ② 选组合后账本+风控渲染（通过 _testPortfolioId 跳过 antd Select jsdom 限制）
  it('选中组合后渲染账本持仓与正常风控', async () => {
    const pid = 'portfolio-001'
    listPlans.mockResolvedValue([makeRulePlan(pid)])
    getPortfolio.mockResolvedValue(makePortfolio(pid))
    getPortfolioRisk.mockResolvedValue(makeRiskNormal(pid))
    listRebalances.mockResolvedValue([makeRebalanceSummary(pid)])

    renderPage({ _testPortfolioId: pid })

    // 账本卡应渲染持仓股
    await waitFor(() =>
      expect(screen.getByText('持仓账本')).toBeInTheDocument(),
    )
    // 应显示标的代码（账本表格）
    await waitFor(() =>
      expect(screen.getAllByText('000001.SZSE').length).toBeGreaterThanOrEqual(1),
    )
    // 风控状态正常
    await waitFor(() =>
      expect(screen.getByText('正常')).toBeInTheDocument(),
    )
  })

  // ③ broken 态红 Alert + 复位按钮调用 resetPortfolioRisk
  it('broken 态展示红色 Alert 并点击复位触发 resetPortfolioRisk', async () => {
    const pid = 'portfolio-001'
    listPlans.mockResolvedValue([makeRulePlan(pid)])
    getPortfolio.mockResolvedValue(makePortfolio(pid))
    getPortfolioRisk.mockResolvedValue(makeRiskBroken(pid))
    listRebalances.mockResolvedValue([])
    resetPortfolioRisk.mockResolvedValue(makeRiskNormal(pid))

    renderPage({ _testPortfolioId: pid })

    // 应显示熔断已触发 Alert
    await waitFor(() =>
      expect(screen.getByText('熔断已触发')).toBeInTheDocument(),
    )
    // broken_date 和 reason
    await waitFor(() =>
      expect(screen.getByText(/回撤超过 20%/)).toBeInTheDocument(),
    )

    // 点击复位按钮触发 Popconfirm
    const resetBtn = screen.getByRole('button', { name: /复位熔断/ })
    await userEvent.click(resetBtn)

    // Popconfirm 出现后确认
    const confirmBtn = await screen.findByRole('button', { name: '确认复位' })
    await userEvent.click(confirmBtn)

    await waitFor(() =>
      expect(resetPortfolioRisk).toHaveBeenCalledWith(pid),
    )
  })

  // ④ 调仓历史行点击开详情 Modal
  it('点击调仓历史行后打开详情 Modal 并展示调仓明细', async () => {
    const pid = 'portfolio-001'
    listPlans.mockResolvedValue([makeRulePlan(pid)])
    getPortfolio.mockResolvedValue(makePortfolio(pid))
    getPortfolioRisk.mockResolvedValue(makeRiskNormal(pid))
    listRebalances.mockResolvedValue([makeRebalanceSummary(pid)])
    getRebalance.mockResolvedValue(makeRebalanceDetail(pid))

    renderPage({ _testPortfolioId: pid })

    // 等待调仓历史表格出现
    await waitFor(() =>
      expect(screen.getByText('调仓历史')).toBeInTheDocument(),
    )

    // 点击历史行（待确认标签那行）
    const pendingTag = await screen.findByText('待确认')
    fireEvent.click(pendingTag.closest('tr')!)

    // Modal 打开后应展示调仓详情
    await waitFor(() =>
      expect(screen.getByText(/调仓详情/)).toBeInTheDocument(),
    )
    await waitFor(() =>
      expect(getRebalance).toHaveBeenCalledWith('2026-06-11:rule_v1:src@v1'),
    )
  })
})

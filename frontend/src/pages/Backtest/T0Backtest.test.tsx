// T0Backtest 页面 RTL：默认单固定档 payload、加策略、label 重复拦截、多策略结果渲染。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'

const mockRunBacktest = vi.fn()
const mockProfile = vi.fn()
const mockProfileSegmented = vi.fn()
const mockListSignals = vi.fn<() => Promise<string[]>>()
const mockGetResources = vi.fn()

vi.mock('../../api/t0', () => ({
  t0Service: {
    runBacktest: (...a: unknown[]) => mockRunBacktest(...a),
    profile: (...a: unknown[]) => mockProfile(...a),
    profileSegmented: (...a: unknown[]) => mockProfileSegmented(...a),
    listSignals: () => mockListSignals(),
  },
}))
vi.mock('../../api/alpha', () => ({
  alphaService: { getDataResources: () => mockGetResources() },
}))

import T0Backtest, { fillLabel, validatePolicies } from './T0Backtest'
import type { TickPolicyCfg, RuleCfg } from '../../types/t0'

const REPORT = {
  symbol: '000415.SZSE', eval_window: ['2023-01-01', '2025-12-31'],
  fill_sensitivity: [
    { tick_label: '固定2分', fill: { penetration: 0, ratio: 1 }, total_return: 0.1, sharpe: 1, max_drawdown: -0.05 },
    { tick_label: '高低开', fill: { penetration: 0, ratio: 1 }, total_return: 0.2, sharpe: 1.2, max_drawdown: -0.06 },
  ],
  results: [
    { tick_label: '固定2分', fill: { penetration: 0, ratio: 1 }, total_return: 0.1, cagr: 0.1, sharpe: 1, max_drawdown: -0.05, turnover_annual: 50, yearly: [{ year: 2024, strat: 0.1, bh: 0.05, half_bh: 0.04, excess_vs_bh: 0.05, excess_vs_half_bh: 0.06 }], monthly_excess: [], hit_dist: { both: 0.3, onlyS: 0.2, onlyB: 0.2, none: 0.3 } },
    { tick_label: '高低开', fill: { penetration: 0, ratio: 1 }, total_return: 0.2, cagr: 0.2, sharpe: 1.2, max_drawdown: -0.06, turnover_annual: 80, yearly: [{ year: 2024, strat: 0.2, bh: 0.05, half_bh: 0.04, excess_vs_bh: 0.15, excess_vs_half_bh: 0.16 }], monthly_excess: [], hit_dist: { both: 0.4, onlyS: 0.2, onlyB: 0.2, none: 0.2 } },
  ],
}

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp><T0Backtest /></AntApp>
    </QueryClientProvider>,
  )
}

const runBtn = () => screen.getByRole('button', { name: /运行回测/ })

describe('T0Backtest 多策略配置', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetResources.mockResolvedValue({ raw_bars: [], raw_ticks: [], derived_bars: [] })
    mockListSignals.mockResolvedValue(['mdl_prob'])
    mockRunBacktest.mockResolvedValue(REPORT)
    mockProfile.mockResolvedValue({ symbol: 'X', window: ['2023-01-01', '2023-06-30'], rows: [],
      suggested_sell_tick: 0.05, suggested_buy_tick: 0.03, note: '' })
    mockProfileSegmented.mockResolvedValue({ symbol: 'X', thresh: 0.003, segments: [] })
  })

  it('默认发送单个固定档策略（向后兼容等价现状）', async () => {
    renderPage()
    expect((await screen.findByLabelText('策略0名称') as HTMLInputElement).value).toBe('固定2分')
    fireEvent.click(runBtn())
    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalledTimes(1))
    const req = mockRunBacktest.mock.calls[0][0]
    expect(req.tick_policies).toHaveLength(1)
    expect(req.tick_policies[0]).toMatchObject({ kind: 'fixed', label: '固定2分', sell_tick: 0.02, buy_tick: 0.02 })
  })

  it('加一个策略 → payload 含两个策略', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    fireEvent.click(screen.getByText('添加档位策略'))
    fireEvent.click(runBtn())
    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalledTimes(1))
    expect(mockRunBacktest.mock.calls[0][0].tick_policies).toHaveLength(2)
  })

  it('策略名称重复 → 拦截运行（不发请求）', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    fireEvent.click(screen.getByText('添加档位策略'))
    // 把第二个策略改名为与第一个相同
    fireEvent.change(await screen.findByLabelText('策略1名称'), { target: { value: '固定2分' } })
    fireEvent.click(runBtn())
    await waitFor(() => expect(screen.getAllByText('名称重复').length).toBeGreaterThan(0))
    expect(mockRunBacktest).not.toHaveBeenCalled()
  })

  it('多策略结果：成交敏感性按策略标注、逐年默认看第一个策略', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    fireEvent.click(runBtn())
    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalled())
    // 两个策略标签都出现在成交敏感性区
    expect(await screen.findAllByText('固定2分')).not.toHaveLength(0)
    expect(await screen.findAllByText('高低开')).not.toHaveLength(0)
    // 逐年"当前口径"默认看第一个策略
    expect(await screen.findByText(/当前口径/)).toBeInTheDocument()
  })

  it('重复成交假设在提交前去重', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    fireEvent.click(screen.getByText('添加成交假设'))   // 加一个与默认首行相同的 0分/100%
    fireEvent.click(runBtn())
    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalled())
    // 默认 3 个(0/100,1/100,0/50)，新增的 0/100 与首行重复被去掉 → 仍 3 个
    expect(mockRunBacktest.mock.calls[0][0].fill_grid).toHaveLength(3)
  })

  it('提交前 trim 策略名称（不发首尾空白）', async () => {
    renderPage()
    fireEvent.change(await screen.findByLabelText('策略0名称'), { target: { value: ' 我的策略 ' } })
    fireEvent.click(runBtn())
    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalled())
    expect(mockRunBacktest.mock.calls[0][0].tick_policies[0].label).toBe('我的策略')
  })

  it('每个策略卡片可触发画像标定抽屉', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    fireEvent.click(screen.getByLabelText('策略0画像'))
    expect(await screen.findByText(/标定：固定2分/)).toBeInTheDocument()
  })

  it('已移除顶部全局画像区', async () => {
    renderPage()
    await screen.findByLabelText('策略0名称')
    expect(screen.queryByText('统计当前标的/区间')).toBeNull()
  })
})

describe('T0Backtest 导出工具', () => {
  it('fillLabel 叠加穿越与部分成交', () => {
    expect(fillLabel({ penetration: 0, ratio: 1 })).toBe('理想撮合（触价即成交）')
    expect(fillLabel({ penetration: 0.01, ratio: 1 })).toBe('穿越 1 分')
    expect(fillLabel({ penetration: 0, ratio: 0.5 })).toBe('部分成交 50%')
    expect(fillLabel({ penetration: 0.01, ratio: 0.5 })).toBe('穿越 1 分 + 部分成交 50%')
  })

  it('validatePolicies 拦空名/重名/空规则/缺信号名', () => {
    const fixed = (l: string): TickPolicyCfg => ({ kind: 'fixed', label: l, sell_tick: 0.02, buy_tick: 0.02 })
    const cond = (rules: RuleCfg[]): TickPolicyCfg =>
      ({ kind: 'conditional', label: 'c', rules, default_sell_tick: 0.02, default_buy_tick: 0.02, pricetick: 0.01 })
    expect(validatePolicies([fixed('a')])).toBeNull()
    expect(validatePolicies([fixed('a'), fixed('a')])).toMatch(/唯一/)
    expect(validatePolicies([fixed('  ')])).toMatch(/名称/)
    expect(validatePolicies([cond([])])).toMatch(/至少需要一条规则/)
    expect(validatePolicies([cond([{ lhs: 'signal', op: 'gt', threshold: 0, sell_tick: 0.02, buy_tick: 0.02 }])]))
      .toMatch(/未选信号名/)
  })
})

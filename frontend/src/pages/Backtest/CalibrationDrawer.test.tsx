// CalibrationDrawer RTL：按 kind 画像、小样本标注、应用回填、波动/趋势仅参考。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import dayjs from 'dayjs'

const mockProfile = vi.fn()
const mockProfileSegmented = vi.fn()
vi.mock('../../api/t0', () => ({
  t0Service: {
    profile: (...a: unknown[]) => mockProfile(...a),
    profileSegmented: (...a: unknown[]) => mockProfileSegmented(...a),
  },
}))

import CalibrationDrawer from './CalibrationDrawer'
import type { TickPolicyCfg } from '../../types/t0'

const prof = (sell: number, buy: number) => ({
  symbol: 'X', window: ['2023-01-01', '2023-06-30'], rows: [
    { x_fen: 2, sell_fill: 0.6, sell_edge_fen: 0.3, buy_fill: 0.7, buy_edge_fen: 0.4, day_pnl_fen: 0.5 },
  ], suggested_sell_tick: sell, suggested_buy_tick: buy, note: '理想撮合前提',
})

const SEG = {
  symbol: 'X', thresh: 0.003, segments: [
    { regime: 'high', label: '高开', n_days: 40, profile: prof(0.07, 0.01) },
    { regime: 'low', label: '低开', n_days: 2, profile: prof(0.09, 0.01) },   // 样本不足
    { regime: 'flat', label: '平开', n_days: 100, profile: prof(0.03, 0.03) },
  ],
}

const COND: TickPolicyCfg = {
  kind: 'conditional', label: '高低开', rules: [
    { name: '高开', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.02, buy_tick: 0.02 },
    { name: '低开', lhs: 'gap', op: 'lt', threshold: -0.003, sell_tick: 0.02, buy_tick: 0.02 },
  ], default_sell_tick: 0.02, default_buy_tick: 0.02, pricetick: 0.01,
}
const FIXED: TickPolicyCfg = { kind: 'fixed', label: '固定', sell_tick: 0.02, buy_tick: 0.02 }
const VOL: TickPolicyCfg = { kind: 'vol_scaled', label: '波动', k: 0.4, n: 20, fallback: 0.02 }

function renderDrawer(policy: TickPolicyCfg) {
  const onApply = vi.fn()
  const onClose = vi.fn()
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <CalibrationDrawer open policy={policy} evalWindow={[dayjs('2024-01-01'), dayjs('2025-12-31')]}
          symbol="X" commissionRate={0.0001} stampDuty={0.0003} xMaxFen={15}
          onApply={onApply} onClose={onClose} />
      </AntApp>
    </QueryClientProvider>,
  )
  return { onApply, onClose }
}

describe('CalibrationDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockProfile.mockResolvedValue(prof(0.05, 0.03))
    mockProfileSegmented.mockResolvedValue(SEG)
  })

  it('默认标定窗早于评估窗：无重叠告警', () => {
    renderDrawer(COND)
    expect(screen.queryByText('标定窗与评估窗重叠')).toBeNull()
  })

  it('条件策略：分场景画像 + 小样本标注 + 应用回填', async () => {
    const { onApply, onClose } = renderDrawer(COND)
    fireEvent.click(screen.getByText('统计画像'))
    expect(await screen.findByText('高开')).toBeInTheDocument()
    expect(screen.getByText('低开')).toBeInTheDocument()
    expect(screen.getByText('平开')).toBeInTheDocument()
    // 低开样本 2 天 < 5 → 标注样本不足
    expect(screen.getAllByText(/样本不足/).length).toBeGreaterThan(0)
    // 应用：高开建议填首个 gap> 规则，低开样本不足跳过，平开→默认档
    fireEvent.click(screen.getByText('应用到本策略'))
    expect(onApply).toHaveBeenCalledTimes(1)
    const next = onApply.mock.calls[0][0]
    expect(next.rules[0]).toMatchObject({ op: 'gt', sell_tick: 0.07, buy_tick: 0.01 })  // 高开回填
    expect(next.rules[1].sell_tick).toBe(0.02)        // 低开样本不足，不回填
    expect(next.default_sell_tick).toBe(0.03)         // 平开→默认档
    expect(onClose).toHaveBeenCalled()
  })

  it('固定档：全窗画像 + 应用一对档', async () => {
    const { onApply } = renderDrawer(FIXED)
    fireEvent.click(screen.getByText('统计画像'))
    await waitFor(() => expect(mockProfile).toHaveBeenCalled())
    expect(await screen.findByText(/建议 卖/)).toBeInTheDocument()
    fireEvent.click(screen.getByText('应用到本策略'))
    expect(onApply.mock.calls[0][0]).toMatchObject({ kind: 'fixed', sell_tick: 0.05, buy_tick: 0.03 })
  })

  it('波动策略：标"仅供参考"', async () => {
    renderDrawer(VOL)
    fireEvent.click(screen.getByText('统计画像'))
    expect(await screen.findByText('仅供参考')).toBeInTheDocument()
  })
})

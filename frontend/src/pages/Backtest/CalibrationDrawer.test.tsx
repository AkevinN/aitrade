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

// 抽屉内嵌 KLineChart 用 lightweight-charts；jsdom 无 canvas，mock 掉避免初始化报错。
vi.mock('lightweight-charts', () => ({
  createChart: vi.fn(() => ({
    addSeries: () => ({ setData: vi.fn(), createPriceLine: vi.fn(), priceScale: () => ({ applyOptions: vi.fn() }) }),
    applyOptions: vi.fn(),
    timeScale: () => ({ fitContent: vi.fn() }),
    subscribeCrosshairMove: vi.fn(),
    remove: vi.fn(),
  })),
  createSeriesMarkers: vi.fn(),
  CandlestickSeries: 'CandlestickSeries',
  HistogramSeries: 'HistogramSeries',
}))

const BARS = [
  { d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 },
  { d: '2025-01-03', open: 10.1, high: 10.2, low: 10.0, close: 10.15 },
]

import CalibrationDrawer, { defaultCalibWindow, clampDrawerWidth } from './CalibrationDrawer'
import { createChart as mockedCreateChart } from 'lightweight-charts'
import type { TickPolicyCfg } from '../../types/t0'

describe('clampDrawerWidth', () => {
  it('夹在 [min, 视口−margin] 内', () => {
    expect(clampDrawerWidth(720, 1440)).toBe(720)        // 正常区间原样
    expect(clampDrawerWidth(100, 1440)).toBe(480)        // < 最小 → 480
    expect(clampDrawerWidth(5000, 1440)).toBe(1360)      // > 视口−80 → 1360
  })
  it('窄视口下视口边界优先（不溢出）', () => {
    expect(clampDrawerWidth(800, 500)).toBe(420)         // min(480) 也让位给 视口−80=420
    expect(clampDrawerWidth(800, 450)).toBe(370)         // 450−80=370 < 视口，仍不溢出
  })
})

describe('defaultCalibWindow', () => {
  it('无本地区间：评估窗起点前一年 → 前一日', () => {
    const [s, e] = defaultCalibWindow(dayjs('2024-01-01'), null)
    expect(s.format('YYYY-MM-DD')).toBe('2023-01-01')
    expect(e.format('YYYY-MM-DD')).toBe('2023-12-31')
  })
  it('默认窗起点早于本地数据起点 → 夹到数据起点', () => {
    const [s, e] = defaultCalibWindow(dayjs('2024-01-01'), { start: '2023-06-01', end: '2025-12-31' })
    expect(s.format('YYYY-MM-DD')).toBe('2023-06-01')   // 2023-01-01 被夹到 2023-06-01
    expect(e.format('YYYY-MM-DD')).toBe('2023-12-31')
  })
  it('数据在评估窗内 → 默认窗不被截断', () => {
    const [s, e] = defaultCalibWindow(dayjs('2024-06-01'), { start: '2020-01-01', end: '2025-12-31' })
    expect(s.format('YYYY-MM-DD')).toBe('2023-06-01')
    expect(e.format('YYYY-MM-DD')).toBe('2024-05-31')
  })
})

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
    mockProfile.mockResolvedValue({ ...prof(0.05, 0.03), bars: BARS })
    mockProfileSegmented.mockResolvedValue({ ...SEG, bars: BARS })
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

  it('统计画像后渲染「K 线与买卖腿」区', async () => {
    renderDrawer(FIXED)
    fireEvent.click(screen.getByText('统计画像'))
    expect(await screen.findByText('K 线与买卖腿')).toBeInTheDocument()
  })

  it('表格含更多字段（成交率 / 全日期望）', async () => {
    renderDrawer(FIXED)
    fireEvent.click(screen.getByText('统计画像'))
    // antd Table scroll.x 会把表头渲染到额外的 sticky 子表 → 同名表头可能多于 1 个，用 All
    expect((await screen.findAllByText('卖成交率')).length).toBeGreaterThan(0)
    expect(screen.getAllByText('买成交率').length).toBeGreaterThan(0)
    expect(screen.getAllByText('全日期望(分)').length).toBeGreaterThan(0)
  })

  it('打开时渲染可拖拽宽度手柄', () => {
    renderDrawer(FIXED)
    expect(screen.getByLabelText('拖拽调整标定抽屉宽度')).toBeInTheDocument()
  })

  it('拖拽调宽：直接写 DOM 宽度、全程不重建 K 线图（丝滑）', async () => {
    // rAF 同步执行，便于断言拖拽期间的直写效果
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame')
      .mockImplementation((cb: FrameRequestCallback) => { cb(0); return 1 })
    const chartCalls = () => (mockedCreateChart as unknown as { mock: { calls: unknown[] } }).mock.calls.length
    renderDrawer(FIXED)
    fireEvent.click(screen.getByText('统计画像'))
    await screen.findByText('K 线与买卖腿')
    const before = chartCalls()
    expect(before).toBeGreaterThan(0)               // 图已建一次

    const handle = screen.getByLabelText('拖拽调整标定抽屉宽度')
    fireEvent.mouseDown(handle, { clientX: 800 })
    fireEvent.mouseMove(document, { clientX: 700, buttons: 1 })
    fireEvent.mouseMove(document, { clientX: 600, buttons: 1 })
    // 拖拽期间直接写 antd 抽屉 DOM 宽度（jsdom innerWidth=1024 → clamp(1024−600,1024)=480）
    const wrapper = document.querySelector('.t0-calib-drawer .ant-drawer-content-wrapper') as HTMLElement
    expect(wrapper.style.width).toBe('480px')
    fireEvent.mouseUp(document, { clientX: 600, buttons: 0 })

    expect(chartCalls()).toBe(before)               // 全程未重建图表（createChart 调用数不变）
    rafSpy.mockRestore()
  })

  it('波动策略：K 线区标"暂不标买卖腿"', async () => {
    renderDrawer(VOL)
    fireEvent.click(screen.getByText('统计画像'))
    expect(await screen.findByText(/暂不标买卖腿/)).toBeInTheDocument()
  })

  it('重开抽屉清空上次结果：应用需对当前策略重新统计才可用', async () => {
    const onApply = vi.fn()
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const FIXED2: TickPolicyCfg = { kind: 'fixed', label: '固定B', sell_tick: 0.02, buy_tick: 0.02 }
    const ui = (open: boolean, policy: TickPolicyCfg | null) => (
      <QueryClientProvider client={qc}>
        <AntApp>
          <CalibrationDrawer open={open} policy={policy} evalWindow={[dayjs('2024-01-01'), dayjs('2025-12-31')]}
            symbol="X" commissionRate={0.0001} stampDuty={0.0003} xMaxFen={15} onApply={onApply} onClose={vi.fn()} />
        </AntApp>
      </QueryClientProvider>
    )
    const { rerender } = render(ui(true, FIXED))
    fireEvent.click(screen.getByText('统计画像'))
    await screen.findByText(/建议 卖/)
    expect(screen.getByRole('button', { name: '应用到本策略' })).toBeEnabled()
    // 关闭 → 为另一策略重开：旧结果应被清空、应用按钮重新禁用
    rerender(ui(false, null))
    rerender(ui(true, FIXED2))
    await waitFor(() => expect(screen.getByRole('button', { name: '应用到本策略' })).toBeDisabled())
    expect(screen.queryByText(/建议 卖/)).toBeNull()
  })
})

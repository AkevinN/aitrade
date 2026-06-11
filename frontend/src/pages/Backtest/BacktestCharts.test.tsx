// 示例测试：BacktestCharts 业务接线组件。
// 该组件用 useQuery 调用 alphaService.getBarDataDetail 拉取 K 线行情，
// 上方渲染「净值 / 回撤曲线」卡片（EquityCurveChart），下方渲染「K 线与买卖点」卡片（KLineChart）。
// 关键契约：K 线行情失败仅在 K 线卡片内呈现错误，绝不影响净值曲线（Req 6.6、6.7）；
//          无成交且无标的时 K 线区给空状态「本次回测无成交」（Req 6.4）。
//
// jsdom 下无法真正渲染 canvas / 量取布局宽高，故：
//   - mock lightweight-charts（同 KLineChart.test.tsx），让 KLineChart 能在 jsdom 中挂载；
//   - mock recharts 的 ResponsiveContainer 为固定尺寸容器（同 EquityCurveChart.test.tsx）；
//   - mock '../../api/alpha'，逐用例控制 getBarDataDetail 的成功 / 失败。
// _Requirements: 6.4, 6.6, 6.7_

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// ---- mock lightweight-charts：jsdom 无 canvas，桩化图表实例 ----
const candleSeries = {
  setData: vi.fn(),
  createPriceLine: vi.fn(),
  priceScale: vi.fn(() => ({ applyOptions: vi.fn() })),
}
const volumeSeries = {
  setData: vi.fn(),
  priceScale: vi.fn(() => ({ applyOptions: vi.fn() })),
}
const createChartMock = vi.fn(() => ({
  addSeries: vi.fn((def: unknown) =>
    def === 'CandlestickSeries' ? candleSeries : volumeSeries,
  ),
  applyOptions: vi.fn(),
  timeScale: vi.fn(() => ({ fitContent: vi.fn() })),
  remove: vi.fn(),
}))

vi.mock('lightweight-charts', () => ({
  createChart: (...args: unknown[]) => createChartMock(...(args as [])),
  createSeriesMarkers: vi.fn(),
  CandlestickSeries: 'CandlestickSeries',
  HistogramSeries: 'HistogramSeries',
}))

// ---- mock recharts：ResponsiveContainer 在 jsdom 下量到 0 宽高会跳过渲染，换成固定尺寸容器 ----
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts')
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div style={{ width: 800, height: 300 }}>{children}</div>
    ),
  }
})

// ---- mock alpha 服务：逐用例控制 getBarDataDetail 行为 ----
vi.mock('../../api/alpha', () => ({
  alphaService: {
    getBarDataDetail: vi.fn(),
  },
}))

import { alphaService } from '../../api/alpha'
import BacktestCharts from './BacktestCharts'
import type { BacktestResultPayload } from '../../types/alpha'

const getBarDataDetailMock = vi.mocked(alphaService.getBarDataDetail)

// 构造一个最小可渲染净值曲线的逐日净值序列
const SAMPLE_EQUITY = [
  { date: '2025-01-02', balance: 100000, drawdown: 0, ddpercent: 0, net_pnl: 0 },
  { date: '2025-01-03', balance: 101000, drawdown: 0, ddpercent: 0, net_pnl: 1000 },
]

// 构造一笔成交（offset=OPEN → 买入买卖点）
const SAMPLE_TRADES = [
  {
    datetime: '2025-01-02',
    vt_symbol: '600000.SSE',
    direction: 'LONG',
    offset: 'OPEN',
    price: 10.5,
    volume: 1000,
  },
]

// getBarDataDetail 成功返回的行情明细（含开高低收，能被适配器转成 OHLCBar）
const SAMPLE_BAR_DETAIL = {
  vt_symbol: '600000.SSE',
  interval: 'd',
  row_count: 2,
  start: '2025-01-02',
  end: '2025-01-03',
  columns: ['datetime', 'open', 'high', 'low', 'close', 'volume'],
  preview: [
    { datetime: '2025-01-02', open: 10, high: 11, low: 9.8, close: 10.5, volume: 1000 },
    { datetime: '2025-01-03', open: 10.5, high: 12, low: 10.3, close: 11.8, volume: 2000 },
  ],
  loaded_count: 2,
  has_more: false,
  next_before: null,
}

// 渲染辅助：每个用例使用全新的 QueryClient（retry: false，避免失败用例重试拖慢测试）
function renderWithClient(result?: BacktestResultPayload) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <BacktestCharts result={result} interval="d" start="2025-01-01" end="2025-03-01" />
    </QueryClientProvider>,
  )
}

describe('BacktestCharts', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('含 trades+equity_curve 且行情请求成功时，净值图与 K 线图同时渲染、无错误提示', async () => {
    getBarDataDetailMock.mockResolvedValue(SAMPLE_BAR_DETAIL)

    renderWithClient({
      target_symbol: '600000.SSE',
      trades: SAMPLE_TRADES,
      equity_curve: SAMPLE_EQUITY,
    })

    // 两张卡片标题都应在
    expect(screen.getByText('净值 / 回撤曲线')).toBeInTheDocument()
    expect(screen.getByText('K 线与买卖点')).toBeInTheDocument()

    // 行情成功解析后，KLineChart 应完成图表初始化
    await waitFor(() => {
      expect(createChartMock).toHaveBeenCalledTimes(1)
    })

    // 净值曲线有数据，不应展示空状态；K 线卡片不应出现错误提示
    expect(screen.queryByText('暂无净值数据')).not.toBeInTheDocument()
    expect(screen.queryByText('K 线行情获取失败')).not.toBeInTheDocument()
  })

  it('成交为空且无标的时，K 线区显示空状态「本次回测无成交」，净值卡片仍渲染（Req 6.4）', () => {
    renderWithClient({
      trades: [],
      equity_curve: SAMPLE_EQUITY,
    })

    // 无 vtSymbol → 不应发起行情请求，也不应初始化图表
    expect(getBarDataDetailMock).not.toHaveBeenCalled()
    expect(createChartMock).not.toHaveBeenCalled()

    // K 线区给空状态而非报错
    expect(screen.getByText('本次回测无成交')).toBeInTheDocument()
    expect(screen.queryByText('K 线行情获取失败')).not.toBeInTheDocument()

    // 净值卡片仍在且有数据
    expect(screen.getByText('净值 / 回撤曲线')).toBeInTheDocument()
    expect(screen.queryByText('暂无净值数据')).not.toBeInTheDocument()
  })

  it('行情请求失败时，K 线区显示错误 Alert，净值曲线卡片不受影响（Req 6.6、6.7）', async () => {
    getBarDataDetailMock.mockRejectedValue(new Error('network error'))

    renderWithClient({
      target_symbol: '600000.SSE',
      trades: SAMPLE_TRADES,
      equity_curve: SAMPLE_EQUITY,
    })

    // 行情失败后，K 线卡片内出现错误提示
    await waitFor(() => {
      expect(screen.getByText('K 线行情获取失败')).toBeInTheDocument()
    })

    // 净值曲线卡片依旧存在且正常（错误被隔离在 K 线卡片内）
    expect(screen.getByText('净值 / 回撤曲线')).toBeInTheDocument()
    expect(screen.queryByText('暂无净值数据')).not.toBeInTheDocument()
  })
})

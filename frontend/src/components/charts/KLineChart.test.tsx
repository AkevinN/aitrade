// 示例测试：KLineChart 通用组件。
// lightweight-charts 在 jsdom 下无法真正渲染 canvas，故对其做 mock，
// 验证：给定 bars 时调用 createChart/addSeries/setData，卸载时调用 chart.remove()（Req 3.7），
// 未提供 markers/overlays 时仅渲染蜡烛图而不报错（Req 3.5），
// 空 bars 渲染 antd Empty，loading 渲染 Spin。
// 注：组件基于 lightweight-charts v5，买卖点标注走 createSeriesMarkers（而非 v4 的 setMarkers），
//     故任务 6.2「setMarkers 被调用」映射为断言 createSeriesMarkers。
// _Requirements: 3.1, 3.2, 3.3, 3.5, 3.7_

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'

// ---- mock lightweight-charts ----
const candleSeries = {
  setData: vi.fn(),
  createPriceLine: vi.fn(),
  priceScale: vi.fn(() => ({ applyOptions: vi.fn() })),
}
const volumeSeries = {
  setData: vi.fn(),
  priceScale: vi.fn(() => ({ applyOptions: vi.fn() })),
}
const chartRemove = vi.fn()
const fitContent = vi.fn()
let crosshairHandler: ((p: unknown) => void) | null = null
const createChartMock = vi.fn(() => ({
  addSeries: vi.fn((def: unknown) =>
    def === 'CandlestickSeries' ? candleSeries : volumeSeries,
  ),
  applyOptions: vi.fn(),
  timeScale: vi.fn(() => ({ fitContent })),
  subscribeCrosshairMove: vi.fn((cb: (p: unknown) => void) => { crosshairHandler = cb }),
  remove: chartRemove,
}))
const createSeriesMarkersMock = vi.fn()

vi.mock('lightweight-charts', () => ({
  createChart: (...args: unknown[]) => createChartMock(...(args as [])),
  createSeriesMarkers: (...args: unknown[]) => createSeriesMarkersMock(...(args as [])),
  CandlestickSeries: 'CandlestickSeries',
  HistogramSeries: 'HistogramSeries',
}))

import KLineChart from './KLineChart'
import type { OHLCBar, TradeMarker } from './types'

const SAMPLE_BARS: OHLCBar[] = [
  { time: '2025-03-04', open: 10, high: 11, low: 9.5, close: 10.5, volume: 1000 },
  { time: '2025-03-05', open: 10.5, high: 12, low: 10.2, close: 11.8, volume: 2000 },
]

describe('KLineChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    crosshairHandler = null
  })

  it('提供 tooltipFormatter：订阅 crosshair 并按 hover 的 bar 渲染明细', () => {
    const fmt = vi.fn((b: OHLCBar) => `详情 ${b.time} 收${b.close}`)
    const { container } = render(<KLineChart bars={SAMPLE_BARS} tooltipFormatter={fmt} />)
    expect(crosshairHandler).toBeTypeOf('function')
    // 模拟 hover 到第一根（日线时间为字符串）
    crosshairHandler!({ time: '2025-03-04', point: { x: 10, y: 10 } })
    expect(fmt).toHaveBeenCalledWith(SAMPLE_BARS[0])
    expect(container.textContent).toContain('详情 2025-03-04 收10.5')
  })

  it('不提供 tooltipFormatter：不订阅 crosshair（行为不变）', () => {
    render(<KLineChart bars={SAMPLE_BARS} />)
    expect(crosshairHandler).toBeNull()
  })

  it('空 bars 渲染 Empty，不初始化图表', () => {
    render(<KLineChart bars={[]} emptyText="本次回测无成交" />)
    expect(screen.getByText('本次回测无成交')).toBeInTheDocument()
    expect(createChartMock).not.toHaveBeenCalled()
  })

  it('loading 时渲染加载态，不初始化图表', () => {
    render(<KLineChart bars={SAMPLE_BARS} loading />)
    expect(createChartMock).not.toHaveBeenCalled()
  })

  it('给定 bars 时创建图表并写入蜡烛数据', () => {
    render(<KLineChart bars={SAMPLE_BARS} />)
    expect(createChartMock).toHaveBeenCalledTimes(1)
    expect(candleSeries.setData).toHaveBeenCalledTimes(1)
    const data = candleSeries.setData.mock.calls[0][0]
    expect(data).toHaveLength(2)
  })

  it('未提供 markers/overlays 时仅渲染蜡烛图、不报错（Req 3.5）', () => {
    // 不传 markers / overlays：应正常创建图表并写入蜡烛数据，
    // 且不触发买卖点标注（createSeriesMarkers），也不创建价位线（createPriceLine）。
    expect(() => render(<KLineChart bars={SAMPLE_BARS} />)).not.toThrow()
    expect(createChartMock).toHaveBeenCalledTimes(1)
    expect(candleSeries.setData).toHaveBeenCalledTimes(1)
    expect(createSeriesMarkersMock).not.toHaveBeenCalled()
    expect(candleSeries.createPriceLine).not.toHaveBeenCalled()
  })

  it('给定 markers 时调用 createSeriesMarkers', () => {
    const markers: TradeMarker[] = [
      { time: '2025-03-04', side: 'buy', price: 10.5, text: '买 1000@10.5' },
      { time: '2025-03-05', side: 'sell', price: 11.8, text: '卖 1000@11.8' },
    ]
    render(<KLineChart bars={SAMPLE_BARS} markers={markers} />)
    expect(createSeriesMarkersMock).toHaveBeenCalledTimes(1)
    const passedMarkers = createSeriesMarkersMock.mock.calls[0][1]
    expect(passedMarkers).toHaveLength(2)
    expect(passedMarkers[0]).toMatchObject({ shape: 'arrowUp', position: 'belowBar' })
    expect(passedMarkers[1]).toMatchObject({ shape: 'arrowDown', position: 'aboveBar' })
  })

  it('卸载时调用 chart.remove() 释放实例', () => {
    const { unmount } = render(<KLineChart bars={SAMPLE_BARS} />)
    expect(chartRemove).not.toHaveBeenCalled()
    unmount()
    expect(chartRemove).toHaveBeenCalledTimes(1)
  })
})

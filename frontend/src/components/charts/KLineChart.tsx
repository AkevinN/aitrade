// 通用 K 线图组件，基于 lightweight-charts v5 实现。
// 纯展示组件：仅消费纯数据 props（bars/markers/overlays），不含任何回测/决策业务语义。
// 业务数据 → 图表数据的转换由 chartAdapters 纯函数完成。
import { useEffect, useRef } from 'react'
import { Empty, Spin } from 'antd'
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type CandlestickData,
  type HistogramData,
  type MouseEventParams,
  type Time,
} from 'lightweight-charts'
import type {
  OHLCBar,
  TradeMarker,
  ChartOverlay,
  KLineColorScheme,
} from './types'

/**
 * 把 lightweight-charts 的 Time 归一成可比对的字符串键。
 *
 * 日线传入 `'YYYY-MM-DD'` 字符串后，crosshair 回传的可能是 `{year,month,day}` BusinessDay；
 * 本函数统一成 `'YYYY-MM-DD'`，使「hover 时间 → 原始 bar」查表稳定。
 *
 * @param t - lightweight-charts 时间值（字符串 / 秒级时间戳 / BusinessDay）
 * @returns 归一化字符串键
 */
export function timeKey(t: unknown): string {
  if (typeof t === 'string') return t
  if (typeof t === 'number') return String(t)
  if (t && typeof t === 'object' && 'year' in t) {
    const d = t as { year: number; month: number; day: number }
    const p = (n: number) => String(n).padStart(2, '0')
    return `${d.year}-${p(d.month)}-${p(d.day)}`
  }
  return String(t)
}

/**
 * {@link KLineChart} 组件 props。
 */
export interface KLineChartProps {
  /** K 线数据（升序）；空数组时渲染 {@link emptyText} 占位。 */
  bars: OHLCBar[]
  /** 买卖点标注列表（可选）。 */
  markers?: TradeMarker[]
  /** 叠加到主图的价位线等图层（可选）。 */
  overlays?: ChartOverlay[]
  /** 是否显示成交量副图，默认 true */
  showVolume?: boolean
  /** 图表高度（px），默认 420 */
  height?: number
  /** 配色覆盖（默认遵循 A 股惯例：涨红跌绿） */
  colors?: Partial<KLineColorScheme>
  /** 加载态 */
  loading?: boolean
  /** 空数据提示文案 */
  emptyText?: string
  /**
   * 可选 hover 明细格式化器：鼠标移到某根 K 线时，浮层显示其返回的多行文本
   * （`\n` 换行，纯文本无 HTML 注入）。不传则无 hover 浮层（行为不变）。
   */
  tooltipFormatter?: (bar: OHLCBar) => string
}

// A 股默认配色：涨红 / 跌绿（与国际惯例相反），买点红、卖点绿
const DEFAULT_COLORS: KLineColorScheme = {
  up: '#dc4446',
  down: '#49aa19',
  buyMarker: '#dc4446',
  sellMarker: '#49aa19',
}

/**
 * 通用 K 线图组件（基于 lightweight-charts v5）。
 *
 * 纯展示组件：仅消费纯数据 props，不含回测/决策等业务语义。
 * 支持成交量副图、买卖点标注（`markers`）、价位线叠加（`overlays`）和 ResizeObserver 自适应宽度。
 * 组件卸载时自动清理图表实例，防止内存泄漏。
 *
 * @remarks
 * 业务数据 → 图表数据的转换应由 {@link ../charts/chartAdapters} 中的纯函数完成后传入。
 */
export default function KLineChart({
  bars,
  markers,
  overlays,
  showVolume = true,
  height = 420,
  colors,
  loading = false,
  emptyText = '暂无数据',
  tooltipFormatter,
}: KLineChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const hasBars = bars.length > 0

  useEffect(() => {
    // 空数据不初始化图表（改由 antd Empty 占位）
    const container = containerRef.current
    if (!container || !hasBars) return

    const scheme: KLineColorScheme = { ...DEFAULT_COLORS, ...colors }

    // 1. 创建图表实例
    const chart: IChartApi = createChart(container, {
      width: container.clientWidth,
      height,
      layout: {
        background: { color: '#ffffff' },
        textColor: '#333333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: { borderColor: '#d9d9d9' },
      timeScale: { borderColor: '#d9d9d9', timeVisible: true, secondsVisible: false },
    })

    // 2. 蜡烛主图（v5：addSeries 传入序列定义）
    const candleSeries: ISeriesApi<'Candlestick'> = chart.addSeries(CandlestickSeries, {
      upColor: scheme.up,
      downColor: scheme.down,
      borderUpColor: scheme.up,
      borderDownColor: scheme.down,
      wickUpColor: scheme.up,
      wickDownColor: scheme.down,
    })
    const candleData: CandlestickData<Time>[] = bars.map((b) => ({
      time: b.time as Time,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }))
    candleSeries.setData(candleData)

    // 3. 成交量副图：独立 priceScale，占据底部约 20% 高度
    if (showVolume) {
      const volumeSeries: ISeriesApi<'Histogram'> = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
      const volumeData: HistogramData<Time>[] = bars
        .filter((b) => b.volume != null)
        .map((b) => ({
          time: b.time as Time,
          value: b.volume as number,
          // 量柱配色跟随当根涨跌
          color: b.close >= b.open ? scheme.up : scheme.down,
        }))
      volumeSeries.setData(volumeData)
    }

    // 4. 买卖点标注（v5：createSeriesMarkers）
    if (markers && markers.length > 0) {
      const seriesMarkers: SeriesMarker<Time>[] = markers.map((m) => {
        if (m.side === 'buy') {
          return {
            time: m.time as Time,
            position: 'belowBar',
            shape: 'arrowUp',
            color: scheme.buyMarker,
            text: m.text,
          }
        }
        return {
          time: m.time as Time,
          position: 'aboveBar',
          shape: 'arrowDown',
          color: scheme.sellMarker,
          text: m.text,
        }
      })
      // markers 需按时间升序
      seriesMarkers.sort((a, b) =>
        a.time < b.time ? -1 : a.time > b.time ? 1 : 0,
      )
      createSeriesMarkers(candleSeries, seriesMarkers)
    }

    // 5. 叠加价位线（阈值 / 决策价）
    if (overlays && overlays.length > 0) {
      overlays.forEach((o) => {
        candleSeries.createPriceLine({
          price: o.price,
          color: o.color ?? '#888888',
          title: o.title ?? '',
          lineWidth: 1,
        })
      })
    }

    // 5b. 可选 hover 明细浮层：crosshair 移到某根 K 线时显示该根明细
    let tooltipEl: HTMLDivElement | null = null
    if (tooltipFormatter) {
      container.style.position = 'relative'
      tooltipEl = document.createElement('div')
      tooltipEl.style.cssText =
        'position:absolute;display:none;pointer-events:none;z-index:20;background:rgba(255,255,255,0.96);' +
        'border:1px solid #d9d9d9;border-radius:4px;padding:6px 8px;font-size:12px;line-height:1.6;' +
        'white-space:pre-line;box-shadow:0 2px 8px rgba(0,0,0,0.12)'
      container.appendChild(tooltipEl)
      const barByTime = new Map<string, OHLCBar>(bars.map((b) => [timeKey(b.time), b]))
      chart.subscribeCrosshairMove((param: MouseEventParams) => {
        const el = tooltipEl as HTMLDivElement
        if (param.time == null || param.point == null) { el.style.display = 'none'; return }
        const bar = barByTime.get(timeKey(param.time))
        if (!bar) { el.style.display = 'none'; return }
        el.textContent = tooltipFormatter(bar)
        el.style.display = 'block'
        el.style.left = `${Math.min(param.point.x + 12, container.clientWidth - el.clientWidth - 8)}px`
        el.style.top = `${Math.max(8, param.point.y - 8)}px`
      })
    }

    chart.timeScale().fitContent()

    // 6. ResizeObserver 适配容器宽度
    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) {
        chart.applyOptions({ width: entry.contentRect.width })
      }
    })
    resizeObserver.observe(container)

    // 7. 卸载清理：释放图表实例，避免内存泄漏（Req 3.7）
    return () => {
      resizeObserver.disconnect()
      if (tooltipEl && tooltipEl.parentNode) tooltipEl.parentNode.removeChild(tooltipEl)
      chart.remove()
    }
  }, [bars, markers, overlays, showVolume, height, colors, tooltipFormatter])

  if (loading) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Spin />
      </div>
    )
  }

  if (!hasBars) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Empty description={emptyText} />
      </div>
    )
  }

  return <div ref={containerRef} style={{ width: '100%', height }} />
}

// buildLegChart 测试：按配置档标买卖腿、条件策略按当日场景取档、动态档不标、无前视。
import { describe, it, expect } from 'vitest'

import { buildLegChart } from './t0LegMarkers'
import type { TickPolicyCfg, T0DailyBar } from '../../types/t0'

const FIXED: TickPolicyCfg = { kind: 'fixed', label: 'f', sell_tick: 0.02, buy_tick: 0.02 }

describe('buildLegChart 固定档', () => {
  it('触价两腿都标', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 }]
    const { markers, bars: cb } = buildLegChart(bars, FIXED)
    expect(cb[0]).toMatchObject({ time: '2025-01-02', open: 10 })
    expect(markers.filter((m) => m.side === 'sell')).toHaveLength(1)   // 10.05>=10.02
    expect(markers.filter((m) => m.side === 'buy')).toHaveLength(1)    // 9.95<=9.98
  })

  it('未触价不标', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.01, low: 9.99, close: 10.0 }]
    expect(buildLegChart(bars, FIXED).markers).toHaveLength(0)
  })
})

describe('buildLegChart 条件策略（按当日场景取配置档）', () => {
  const COND: TickPolicyCfg = {
    kind: 'conditional', label: 'c', rules: [
      { name: '高开', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.50, buy_tick: 0.01 },  // 卖档巨大→高开不触卖
      { name: '低开', lhs: 'gap', op: 'lt', threshold: -0.003, sell_tick: 0.01, buy_tick: 0.50 },
    ], default_sell_tick: 0.02, default_buy_tick: 0.02, pricetick: 0.01,
  }

  it('高开日用高开规则的档（卖档0.5→不触卖、买档0.01→触买）', () => {
    const bars: T0DailyBar[] = [
      { d: '2025-01-02', open: 10.0, high: 10.05, low: 9.95, close: 10.0 },   // 首日 gap=0 → 默认档
      { d: '2025-01-03', open: 10.10, high: 10.30, low: 10.05, close: 10.10 }, // gap=+1%→高开
    ]
    const { markers, details } = buildLegChart(bars, COND)
    const d2Sell = markers.filter((m) => m.time === '2025-01-03' && m.side === 'sell')
    const d2Buy = markers.filter((m) => m.time === '2025-01-03' && m.side === 'buy')
    expect(d2Sell).toHaveLength(0)   // 高开卖档 0.50 → 10.60 未达
    expect(d2Buy).toHaveLength(1)    // 高开买档 0.01 → 10.09，low 10.05<=10.09 触发
    expect(details.get('2025-01-03')).toContain('高开')
  })

  it('首日 gap 用昨收（无昨收→0→默认档），无前视', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10.0, high: 10.05, low: 9.95, close: 10.0 }]
    const { details } = buildLegChart(bars, COND)
    expect(details.get('2025-01-02')).toContain('默认')
    expect(details.get('2025-01-02')).toContain('首日无昨收')
  })
})

describe('buildLegChart 动态档', () => {
  it('波动/趋势策略：不标腿、supported=false', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.5, low: 9.5, close: 10.0 }]
    const vol: TickPolicyCfg = { kind: 'vol_scaled', label: 'v', k: 0.4, n: 20, fallback: 0.02 }
    const out = buildLegChart(bars, vol)
    expect(out.supported).toBe(false)
    expect(out.markers).toHaveLength(0)
    expect(out.details.get('2025-01-02')).toContain('动态')
  })
})

describe('buildLegChart 明细', () => {
  it('每日都有明细，含 OHLC 与跳空', () => {
    const bars: T0DailyBar[] = [
      { d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 },
      { d: '2025-01-03', open: 10.1, high: 10.2, low: 10.0, close: 10.15 },
    ]
    const { details } = buildLegChart(bars, FIXED)
    expect(details.size).toBe(2)
    expect(details.get('2025-01-03')).toContain('跳空')
    expect(details.get('2025-01-03')).toContain('开 10.1')
  })
})

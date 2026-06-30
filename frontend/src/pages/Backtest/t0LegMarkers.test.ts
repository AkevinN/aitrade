// buildLegChart 测试：按解析器逐日档位标买卖腿；建议档解析器与表格建议一致；条件按场景；动态档不标。
import { describe, it, expect } from 'vitest'

import {
  buildLegChart, fixedSuggestionResolver, gapSuggestionResolver, noMarkerResolver,
} from './t0LegMarkers'
import type { T0DailyBar, T0Profile, T0SegmentedProfile } from '../../types/t0'

const prof = (sell: number, buy: number): T0Profile => ({
  symbol: 'X', window: ['2024-01-01', '2024-12-31'], rows: [],
  suggested_sell_tick: sell, suggested_buy_tick: buy, note: '',
})

describe('buildLegChart + fixedSuggestionResolver（与表格建议档一致）', () => {
  it('每天都用全窗建议档判定触发', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 }]
    const { markers } = buildLegChart(bars, fixedSuggestionResolver(prof(0.02, 0.02)))
    expect(markers.filter((m) => m.side === 'sell')).toHaveLength(1)   // 10.05>=10.02
    expect(markers.filter((m) => m.side === 'buy')).toHaveLength(1)    // 9.95<=9.98
  })

  it('建议卖档很大时该腿不触发（对齐表格 19% 成交那种远档）', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 }]
    const { markers } = buildLegChart(bars, fixedSuggestionResolver(prof(0.11, 0.02)))   // 卖11分→10.11 未达
    expect(markers.filter((m) => m.side === 'sell')).toHaveLength(0)
    expect(markers.filter((m) => m.side === 'buy')).toHaveLength(1)
  })
})

describe('gapSuggestionResolver（条件策略按当日场景用该场景建议档）', () => {
  const seg: T0SegmentedProfile = {
    symbol: 'X', thresh: 0.003, segments: [
      { regime: 'high', label: '高开', n_days: 40, profile: prof(0.50, 0.01) },  // 高开卖档巨大→不触卖
      { regime: 'low', label: '低开', n_days: 30, profile: prof(0.01, 0.50) },
      { regime: 'flat', label: '平开', n_days: 100, profile: prof(0.08, 0.02) },
    ],
  }

  it('平开日用平开场景建议档（卖0.08/买0.02），与表格一致', () => {
    // 首日 gap=0 → 平开
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10.0, high: 10.05, low: 9.95, close: 10.0 }]
    const { markers, details } = buildLegChart(bars, gapSuggestionResolver(seg))
    expect(markers.filter((m) => m.side === 'sell')).toHaveLength(0)   // 卖0.08→10.08 未达(高10.05)
    expect(markers.filter((m) => m.side === 'buy')).toHaveLength(1)    // 买0.02→9.98，低9.95<=9.98 触发
    expect(details.get('2025-01-02')).toContain('平开')
    expect(details.get('2025-01-02')).toContain('开+8分')               // 平开建议卖8分
  })

  it('高开日用高开场景建议档', () => {
    const bars: T0DailyBar[] = [
      { d: '2025-01-02', open: 10.0, high: 10.05, low: 9.95, close: 10.0 },   // 首日平开
      { d: '2025-01-03', open: 10.10, high: 10.30, low: 10.05, close: 10.10 }, // gap≈+1%→高开
    ]
    const { markers, details } = buildLegChart(bars, gapSuggestionResolver(seg))
    expect(details.get('2025-01-03')).toContain('高开')
    // 高开卖0.50→10.60 未达；买0.01→10.09，低10.05<=10.09 触发
    expect(markers.filter((m) => m.time === '2025-01-03' && m.side === 'sell')).toHaveLength(0)
    expect(markers.filter((m) => m.time === '2025-01-03' && m.side === 'buy')).toHaveLength(1)
  })
})

describe('noMarkerResolver（动态档）', () => {
  it('不标腿、supported=false', () => {
    const bars: T0DailyBar[] = [{ d: '2025-01-02', open: 10, high: 10.5, low: 9.5, close: 10.0 }]
    const out = buildLegChart(bars, noMarkerResolver)
    expect(out.supported).toBe(false)
    expect(out.markers).toHaveLength(0)
    expect(out.details.get('2025-01-02')).toContain('动态')
  })
})

describe('明细', () => {
  it('每日都有明细，含 OHLC 与跳空', () => {
    const bars: T0DailyBar[] = [
      { d: '2025-01-02', open: 10, high: 10.05, low: 9.95, close: 10.0 },
      { d: '2025-01-03', open: 10.1, high: 10.2, low: 10.0, close: 10.15 },
    ]
    const { details } = buildLegChart(bars, fixedSuggestionResolver(prof(0.02, 0.02)))
    expect(details.size).toBe(2)
    expect(details.get('2025-01-03')).toContain('跳空')
    expect(details.get('2025-01-03')).toContain('开 10.1')
  })
})

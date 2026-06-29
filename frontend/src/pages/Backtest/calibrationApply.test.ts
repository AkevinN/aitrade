// calibrationApply 纯函数测试：建议→策略回填映射、小样本跳过、非对象原样返回。
import { describe, it, expect } from 'vitest'

import { applyFixedSuggestion, applyGapSegments } from './calibrationApply'
import type { TickPolicyCfg, T0Profile, T0SegmentedProfile } from '../../types/t0'

const prof = (sell: number, buy: number): T0Profile => ({
  symbol: 'X', window: ['2023-01-01', '2023-06-30'], rows: [],
  suggested_sell_tick: sell, suggested_buy_tick: buy, note: '',
})

const segmented = (
  high: { n: number; s: number; b: number },
  low: { n: number; s: number; b: number },
  flat: { n: number; s: number; b: number },
): T0SegmentedProfile => ({
  symbol: 'X', thresh: 0.003, segments: [
    { regime: 'high', label: '高开', n_days: high.n, profile: prof(high.s, high.b) },
    { regime: 'low', label: '低开', n_days: low.n, profile: prof(low.s, low.b) },
    { regime: 'flat', label: '平开', n_days: flat.n, profile: prof(flat.s, flat.b) },
  ],
})

const COND: TickPolicyCfg = {
  kind: 'conditional', label: 'c', rules: [
    { name: '高开', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.02, buy_tick: 0.02 },
    { name: '低开', lhs: 'gap', op: 'lt', threshold: -0.003, sell_tick: 0.02, buy_tick: 0.02 },
  ], default_sell_tick: 0.02, default_buy_tick: 0.02, pricetick: 0.01,
}

describe('applyFixedSuggestion', () => {
  it('把全窗建议填进固定档', () => {
    const p: TickPolicyCfg = { kind: 'fixed', label: 'f', sell_tick: 0.02, buy_tick: 0.02 }
    const out = applyFixedSuggestion(p, prof(0.07, 0.03))
    expect(out).toMatchObject({ kind: 'fixed', sell_tick: 0.07, buy_tick: 0.03 })
  })
  it('非固定档原样返回', () => {
    expect(applyFixedSuggestion(COND, prof(0.07, 0.03))).toBe(COND)
  })
})

describe('applyGapSegments', () => {
  it('高/低/平开建议分别填到对应规则与默认档', () => {
    const segs = segmented({ n: 40, s: 0.07, b: 0.01 }, { n: 30, s: 0.09, b: 0.01 }, { n: 100, s: 0.03, b: 0.03 })
    const out = applyGapSegments(COND, segs) as Extract<TickPolicyCfg, { kind: 'conditional' }>
    expect(out.rules[0]).toMatchObject({ op: 'gt', sell_tick: 0.07, buy_tick: 0.01 })  // 高开
    expect(out.rules[1]).toMatchObject({ op: 'lt', sell_tick: 0.09, buy_tick: 0.01 })  // 低开
    expect(out.default_sell_tick).toBe(0.03)
    expect(out.default_buy_tick).toBe(0.03)
  })

  it('样本不足的场景不回填（保持原值）', () => {
    // 高开仅 2 天 (<5) → 高开规则保持 0.02；低开/平开充足 → 回填
    const segs = segmented({ n: 2, s: 0.07, b: 0.01 }, { n: 30, s: 0.09, b: 0.01 }, { n: 100, s: 0.03, b: 0.03 })
    const out = applyGapSegments(COND, segs) as Extract<TickPolicyCfg, { kind: 'conditional' }>
    expect(out.rules[0].sell_tick).toBe(0.02)   // 高开不回填
    expect(out.rules[1].sell_tick).toBe(0.09)   // 低开回填
    expect(out.default_sell_tick).toBe(0.03)    // 平开回填
  })

  it('策略无对应跳空规则时，仅平开落到默认档', () => {
    const noGap: TickPolicyCfg = {
      kind: 'conditional', label: 'c', rules: [
        { lhs: 'signal', op: 'gt', threshold: 0.5, signal_name: 's', sell_tick: 0.05, buy_tick: 0.01 },
      ], default_sell_tick: 0.02, default_buy_tick: 0.02, pricetick: 0.01,
    }
    const segs = segmented({ n: 40, s: 0.07, b: 0.01 }, { n: 30, s: 0.09, b: 0.01 }, { n: 100, s: 0.03, b: 0.03 })
    const out = applyGapSegments(noGap, segs) as Extract<TickPolicyCfg, { kind: 'conditional' }>
    expect(out.rules[0].sell_tick).toBe(0.05)   // signal 规则不受影响
    expect(out.default_sell_tick).toBe(0.03)    // 平开仍填默认档
  })

  it('非条件策略原样返回', () => {
    const f: TickPolicyCfg = { kind: 'fixed', label: 'f', sell_tick: 0.02, buy_tick: 0.02 }
    const segs = segmented({ n: 40, s: 0.07, b: 0.01 }, { n: 30, s: 0.09, b: 0.01 }, { n: 100, s: 0.03, b: 0.03 })
    expect(applyGapSegments(f, segs)).toBe(f)
  })
})

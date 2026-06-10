// 单元测试：聚合数据驱动配置的纯函数。
//
// 测试运行器为仓库已配置的 Vitest。本文件覆盖示例与边界用例，
// 与 aggregation.property.test.ts（基于属性的测试）互补。

import { describe, it, expect } from 'vitest'

import {
  intervalToMinutes,
  INTERVAL_MINUTES,
  computeTargetIntervalOptions,
  computeCommonRange,
  validateAggregation,
  type AggregationConfig,
  type AvailabilityMap,
  type SymbolAvailability,
} from './aggregation'

// ---------------------------------------------------------------------------
// intervalToMinutes（Requirements: 3.1, 3.2）
// ---------------------------------------------------------------------------

describe('intervalToMinutes', () => {
  // Requirement 3.2：日线等非分钟级周期必须被排除（返回 null）。
  it('returns null for daily interval "d"', () => {
    expect(intervalToMinutes('d')).toBeNull()
  })

  // Requirement 3.2：tick 不是分钟级周期，返回 null。
  it('returns null for "tick"', () => {
    expect(intervalToMinutes('tick')).toBeNull()
  })

  // Requirement 3.1：未映射/未知字符串不是有效的分钟级来源周期，返回 null。
  it.each(['xyz', '', '2h', '1d', '1', 'm', '5M', ' 5m', '5m ', 'w'])(
    'returns null for unknown/unmapped interval %j',
    (interval) => {
      expect(intervalToMinutes(interval)).toBeNull()
    },
  )

  // Requirement 3.1：分钟级周期返回对应的正整数分钟数。
  it.each([
    ['1m', 1],
    ['5m', 5],
    ['10m', 10],
    ['15m', 15],
    ['30m', 30],
    ['60m', 60],
  ])('maps minute-level interval %s to %i minutes', (interval, expected) => {
    expect(intervalToMinutes(interval as string)).toBe(expected)
  })

  // Requirement 3.1：所有映射项均返回正整数，且与常量表一致。
  it('returns a positive integer for every mapped interval and matches INTERVAL_MINUTES', () => {
    for (const [interval, minutes] of Object.entries(INTERVAL_MINUTES)) {
      const result = intervalToMinutes(interval)
      expect(result).toBe(minutes)
      expect(result).toBeGreaterThan(0)
      expect(Number.isInteger(result as number)).toBe(true)
    }
  })
})

// ---------------------------------------------------------------------------
// computeTargetIntervalOptions（Requirements: 4.2, 4.3）
// ---------------------------------------------------------------------------

describe('computeTargetIntervalOptions', () => {
  // Requirement 4.2：1m 来源可聚合到所有更粗且整除的候选周期。
  it('returns {5m,10m,15m,30m,60m} for bar source 1m', () => {
    expect(computeTargetIntervalOptions('bar', '1m')).toEqual([
      '5m',
      '10m',
      '15m',
      '30m',
      '60m',
    ])
  })

  // Requirement 4.2：5m 来源保留 10/15/30/60（15%5==0 故含 15m），排除 5m 自身。
  it('returns {10m,15m,30m,60m} for bar source 5m (15m included since 15%5==0)', () => {
    expect(computeTargetIntervalOptions('bar', '5m')).toEqual(['10m', '15m', '30m', '60m'])
  })

  // Requirement 4.2：30m 来源只有 60m 满足整除且更粗。
  it('returns {60m} for bar source 30m', () => {
    expect(computeTargetIntervalOptions('bar', '30m')).toEqual(['60m'])
  })

  // Requirement 4.2：60m 是最粗候选，无可聚合目标。
  it('returns {} for bar source 60m', () => {
    expect(computeTargetIntervalOptions('bar', '60m')).toEqual([])
  })

  // Requirement 4.2：bar 来源但来源周期为空时无可聚合目标。
  it('returns {} for bar source with null sourceInterval', () => {
    expect(computeTargetIntervalOptions('bar', null)).toEqual([])
  })

  // Requirement 4.3：tick 来源返回全部候选周期。
  it('returns all candidates for tick source', () => {
    expect(computeTargetIntervalOptions('tick', null)).toEqual([
      '5m',
      '10m',
      '15m',
      '30m',
      '60m',
    ])
  })
})

// ---------------------------------------------------------------------------
// computeCommonRange（Requirements: 5.1, 5.2）
// ---------------------------------------------------------------------------

/** 构造仅含单一周期区间的合约可用性，便于组装测试用映射。 */
function makeAvailability(
  interval: string,
  start: string,
  end: string,
): SymbolAvailability {
  return {
    intervals: new Set<string>([interval]),
    start,
    end,
    intervalRanges: { [interval]: { start, end } },
  }
}

describe('computeCommonRange', () => {
  // Requirement 5.1：两合约部分重叠时，公共区间 = [max(start), min(end)]。
  it('returns overlap [max(start), min(end)] for partially overlapping bar ranges', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeAvailability('1m', '2024-01-01T00:00:00Z', '2024-06-01T00:00:00Z')],
      ['BBB', makeAvailability('1m', '2024-03-01T00:00:00Z', '2024-09-01T00:00:00Z')],
    ])
    expect(computeCommonRange(['AAA', 'BBB'], availability, 'bar', '1m')).toEqual({
      start: '2024-03-01T00:00:00Z',
      end: '2024-06-01T00:00:00Z',
    })
  })

  // Requirement 5.2：完全无重叠时返回 null。
  it('returns null for completely non-overlapping ranges', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeAvailability('1m', '2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z')],
      ['BBB', makeAvailability('1m', '2024-05-01T00:00:00Z', '2024-06-01T00:00:00Z')],
    ])
    expect(computeCommonRange(['AAA', 'BBB'], availability, 'bar', '1m')).toBeNull()
  })

  // Requirement 5.1：单合约时公共区间即该合约自身区间。
  it('returns the single symbol range for a single selected symbol', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeAvailability('1m', '2024-01-01T00:00:00Z', '2024-06-01T00:00:00Z')],
    ])
    expect(computeCommonRange(['AAA'], availability, 'bar', '1m')).toEqual({
      start: '2024-01-01T00:00:00Z',
      end: '2024-06-01T00:00:00Z',
    })
  })

  // Requirement 5.2：所选合约缺少对应来源周期的区间时返回 null。
  it('returns null when a symbol is missing the requested interval range', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeAvailability('1m', '2024-01-01T00:00:00Z', '2024-06-01T00:00:00Z')],
      // BBB 仅有 5m 数据，缺失 1m 来源区间。
      ['BBB', makeAvailability('5m', '2024-01-01T00:00:00Z', '2024-06-01T00:00:00Z')],
    ])
    expect(computeCommonRange(['AAA', 'BBB'], availability, 'bar', '1m')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// validateAggregation（Requirements: 7.1, 7.2）
// ---------------------------------------------------------------------------

/**
 * 构造支持多周期、可分别指定区间的合约可用性，便于组装 validateAggregation
 * 的各分支测试用映射。`ranges` 中的键即该合约拥有的周期集合。
 */
function makeMultiAvailability(
  ranges: Record<string, { start: string; end: string }>,
): SymbolAvailability {
  const starts = Object.values(ranges).map((r) => r.start)
  const ends = Object.values(ranges).map((r) => r.end)
  return {
    intervals: new Set<string>(Object.keys(ranges)),
    start: starts.length > 0 ? starts.reduce((a, b) => (b < a ? b : a)) : '',
    end: ends.length > 0 ? ends.reduce((a, b) => (b > a ? b : a)) : '',
    intervalRanges: ranges,
  }
}

const FULL_RANGE = { start: '2024-01-01T00:00:00Z', end: '2024-06-01T00:00:00Z' }

/** 一个完全有效的基线配置 + 可用性，单点修改即可触发各无效分支。 */
function validBaseline(): { config: AggregationConfig; availability: AvailabilityMap } {
  const availability: AvailabilityMap = new Map([
    ['AAA', makeMultiAvailability({ '1m': FULL_RANGE, tick: FULL_RANGE })],
  ])
  const config: AggregationConfig = {
    selectedSymbols: ['AAA'],
    sourceKind: 'bar',
    sourceInterval: '1m',
    targetInterval: '5m',
    range: null,
    sessionProfile: 'cn_equity',
  }
  return { config, availability }
}

describe('validateAggregation', () => {
  // Requirement 7.1：完全有效的配置应通过校验，valid===true 且 reason===null。
  it('returns valid (reason null) for a fully valid combination', () => {
    const { config, availability } = validBaseline()
    expect(validateAggregation(config, availability)).toEqual({
      valid: true,
      reason: null,
    })
  })

  // Requirement 7.1：用户区间与公共区间部分重叠时仍然有效。
  it('returns valid when user range partially overlaps the common range', () => {
    const { config, availability } = validBaseline()
    config.range = ['2024-03-01T00:00:00Z', '2024-09-01T00:00:00Z']
    expect(validateAggregation(config, availability)).toEqual({
      valid: true,
      reason: null,
    })
  })

  // Requirement 7.2 / 维度 no-symbol：未选任何合约。
  it('returns reason "no-symbol" when no symbol is selected', () => {
    const { config, availability } = validBaseline()
    config.selectedSymbols = []
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-symbol',
    })
  })

  // Requirement 7.2 / 维度 no-common-source：所选合约无任何公共可用来源类型。
  it('returns reason "no-common-source" when selected symbols share no source kind', () => {
    const availability: AvailabilityMap = new Map([
      // AAA 仅有 bar（1m），BBB 仅有 tick，交集为空。
      ['AAA', makeMultiAvailability({ '1m': FULL_RANGE })],
      ['BBB', makeMultiAvailability({ tick: FULL_RANGE })],
    ])
    const config: AggregationConfig = {
      selectedSymbols: ['AAA', 'BBB'],
      sourceKind: 'bar',
      sourceInterval: '1m',
      targetInterval: '5m',
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-common-source',
    })
  })

  // Requirement 7.2 / 维度 no-common-source：选定来源类型不在公共可用集合中。
  it('returns reason "no-common-source" when the selected source kind is unavailable', () => {
    const { config, availability } = validBaseline()
    // 基线 AAA 仅有 bar（1m）+ tick；移除 tick 后选定 tick 即越界。
    availability.set('AAA', makeMultiAvailability({ '1m': FULL_RANGE }))
    config.sourceKind = 'tick'
    config.sourceInterval = null
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-common-source',
    })
  })

  // Requirement 7.2 / 维度 no-source-interval：bar 来源下选定来源周期不在可用选项中。
  it('returns reason "no-source-interval" when the bar source interval is unavailable', () => {
    const { config, availability } = validBaseline()
    // AAA 仅有 1m，选定 5m 越界。
    config.sourceInterval = '5m'
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-source-interval',
    })
  })

  // Requirement 7.2 / 维度 no-target：当前来源下无可聚合目标周期。
  it('returns reason "no-target" when no aggregatable target interval exists', () => {
    const availability: AvailabilityMap = new Map([
      // AAA 拥有 60m（最粗候选），bar+60m 无更粗的整除目标。
      ['AAA', makeMultiAvailability({ '60m': FULL_RANGE })],
    ])
    const config: AggregationConfig = {
      selectedSymbols: ['AAA'],
      sourceKind: 'bar',
      sourceInterval: '60m',
      targetInterval: '5m',
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-target',
    })
  })

  // Requirement 7.2 / 维度 no-target：选定目标周期不在可用选项中。
  it('returns reason "no-target" when the selected target interval is out of options', () => {
    const { config, availability } = validBaseline()
    // bar+1m 的目标选项为 {5m,10m,15m,30m,60m}，1m 不在其中。
    config.targetInterval = '1m'
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-target',
    })
  })

  // Requirement 7.2 / 维度 no-range-overlap：用户区间与公共可用区间零重叠。
  it('returns reason "no-range-overlap" when the user range does not overlap', () => {
    const { config, availability } = validBaseline()
    // 公共区间为 2024-01 ~ 2024-06；用户区间落在其后，零重叠。
    config.range = ['2024-08-01T00:00:00Z', '2024-09-01T00:00:00Z']
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-range-overlap',
    })
  })

  // Requirement 7.2 / 优先级：no-symbol 先于其它一切维度命中。
  it('prioritizes "no-symbol" over later dimensions', () => {
    const availability: AvailabilityMap = new Map()
    const config: AggregationConfig = {
      selectedSymbols: [],
      sourceKind: null, // 同时缺来源类型
      sourceInterval: null,
      targetInterval: null,
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability).reason).toBe('no-symbol')
  })

  // Requirement 7.2 / 优先级：no-common-source 先于 no-source-interval 命中。
  it('prioritizes "no-common-source" over "no-source-interval"', () => {
    const availability: AvailabilityMap = new Map([
      // AAA 仅有 tick：bar 来源既无公共来源类型、也无来源周期。
      ['AAA', makeMultiAvailability({ tick: FULL_RANGE })],
    ])
    const config: AggregationConfig = {
      selectedSymbols: ['AAA'],
      sourceKind: 'bar', // 选定 bar，但公共集合仅含 tick
      sourceInterval: '1m',
      targetInterval: '5m',
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability).reason).toBe('no-common-source')
  })

  // Requirement 7.2 / 优先级：no-source-interval 先于 no-target / no-range-overlap 命中。
  it('prioritizes "no-source-interval" over "no-target" and "no-range-overlap"', () => {
    const { config, availability } = validBaseline()
    // 选定来源周期越界（5m 不在 {1m} 中），同时目标/区间也可能无效。
    config.sourceInterval = '5m'
    config.targetInterval = '1m' // 目标也无效
    config.range = ['2024-08-01T00:00:00Z', '2024-09-01T00:00:00Z'] // 区间也无效
    expect(validateAggregation(config, availability).reason).toBe('no-source-interval')
  })

  // Requirement 7.2 / 优先级：no-target 先于 no-range-overlap 命中。
  it('prioritizes "no-target" over "no-range-overlap"', () => {
    const { config, availability } = validBaseline()
    config.targetInterval = '1m' // 目标无效
    config.range = ['2024-08-01T00:00:00Z', '2024-09-01T00:00:00Z'] // 区间也无效
    expect(validateAggregation(config, availability).reason).toBe('no-target')
  })

  // Requirement 7.1 / tick 来源：无需来源周期即可有效。
  it('returns valid for a tick source without source interval', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeMultiAvailability({ tick: FULL_RANGE })],
    ])
    const config: AggregationConfig = {
      selectedSymbols: ['AAA'],
      sourceKind: 'tick',
      sourceInterval: null,
      targetInterval: '5m',
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability)).toEqual({
      valid: true,
      reason: null,
    })
  })

  // Requirement 7.2 / 维度 no-range-overlap：所选合约无公共可用区间。
  it('returns reason "no-range-overlap" when symbols have no common available range', () => {
    const availability: AvailabilityMap = new Map([
      ['AAA', makeMultiAvailability({ '1m': { start: '2024-01-01T00:00:00Z', end: '2024-02-01T00:00:00Z' } })],
      ['BBB', makeMultiAvailability({ '1m': { start: '2024-05-01T00:00:00Z', end: '2024-06-01T00:00:00Z' } })],
    ])
    const config: AggregationConfig = {
      selectedSymbols: ['AAA', 'BBB'],
      sourceKind: 'bar',
      sourceInterval: '1m',
      targetInterval: '5m',
      range: null,
      sessionProfile: 'cn_equity',
    }
    expect(validateAggregation(config, availability)).toEqual({
      valid: false,
      reason: 'no-range-overlap',
    })
  })
})

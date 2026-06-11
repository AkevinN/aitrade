// 基于属性的测试（fast-check）：聚合数据驱动配置的纯函数。
//
// 测试运行器为仓库已配置的 Vitest；属性库为 fast-check。
// 每个属性最少运行 100 次迭代（numRuns: 100），并以注释标注其对应的
// 设计属性编号（见 design.md 的 Correctness Properties）。

import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import dayjs from 'dayjs'

import { buildAvailableSymbols } from '../hooks/useAvailableSymbols'
import type {
  DataResourceKind,
  DataResourceList,
  DataResourceSummary,
} from '../types/alpha'

// ---------------------------------------------------------------------------
// 共享生成器
// ---------------------------------------------------------------------------

/** 与 buildAvailableSymbols 一致：仅去除单个尾点。 */
const stripTrailingDot = (s: string): string => s.replace(/\.$/, '')

/**
 * 随机 vt_symbol：包含普通代码、以及带 0/1/2 个尾点的变体，
 * 还包含会在去尾点后变为空串（应被排除）的退化样本（如 '.'、''）。
 * 不同基底叠加尾点会发生归并（例如 '600000' 与 '600000.' 同键）。
 */
const symbolArb = fc
  .tuple(
    fc.constantFrom('AAA.SZ', '600000', 'INDEX.TEST', 'sz000415', 'X', '', '.'),
    fc.constantFrom('', '.', '..'),
  )
  .map(([base, dots]) => base + dots)

/**
 * 随机 ISO 时间字符串。ISO 8601 的字典序与时间先后一致，
 * 因此可与实现中基于字符串比较的 min/max 逻辑保持一致。
 */
const dateArb = fc
  .date({
    min: new Date('2015-01-01T00:00:00.000Z'),
    max: new Date('2025-12-31T00:00:00.000Z'),
    noInvalidDate: true,
  })
  .map((d) => d.toISOString())

/** 随机周期：混入 'd'、'tick' 与分钟级。 */
const intervalArb = fc.constantFrom('d', 'tick', '1m', '5m', '10m', '15m', '30m', '60m')

interface RawFields {
  vt_symbol: string
  interval: string
  start: string
  end: string
  target_interval?: string
}

/** 用随机字段构造一条完整的 DataResourceSummary（其余字段填充默认值）。 */
function makeSummary(fields: RawFields, kind: DataResourceKind): DataResourceSummary {
  return {
    key: `${kind}:${fields.vt_symbol}:${fields.interval}`,
    kind,
    vt_symbol: fields.vt_symbol,
    interval: fields.interval,
    row_count: 0,
    start: fields.start,
    end: fields.end,
    file_size_kb: 0,
    source_kind: '',
    source_interval: '',
    target_interval: fields.target_interval ?? '',
    created_at: null,
    session_profile: undefined,
  }
}

const rawBarArb = fc
  .record({ vt_symbol: symbolArb, interval: intervalArb, start: dateArb, end: dateArb })
  .map((r) => makeSummary(r, 'raw_bar'))

const rawTickArb = fc
  .record({ vt_symbol: symbolArb, interval: fc.constant('tick'), start: dateArb, end: dateArb })
  .map((r) => makeSummary(r, 'raw_tick'))

const derivedBarArb = fc
  .record({
    vt_symbol: symbolArb,
    interval: intervalArb,
    target_interval: fc.constantFrom('', '5m', '10m', '15m', '30m', '60m'),
    start: dateArb,
    end: dateArb,
  })
  .map((r) => makeSummary(r, 'derived_bar'))

const resourceListArb: fc.Arbitrary<DataResourceList> = fc.record({
  raw_bars: fc.array(rawBarArb, { maxLength: 8 }),
  raw_ticks: fc.array(rawTickArb, { maxLength: 8 }),
  derived_bars: fc.array(derivedBarArb, { maxLength: 8 }),
  raw_bar_intervals: fc.constant<string[]>([]),
  derived_intervals: fc.constant<string[]>([]),
})

// ---------------------------------------------------------------------------
// Property 1
// ---------------------------------------------------------------------------

describe('useAvailableSymbols / buildAvailableSymbols', () => {
  // Feature: aggregation-data-driven-config, Property 1: For any DataResourceList，由 useAvailableSymbols 构建的映射中，每一个在 raw_bars/raw_ticks/derived_bars 中出现过（去尾点后非空）的 vt_symbol 恰好作为一个键出现一次，且其 start 等于该合约所有资源 start 的最小值、end 等于所有资源 end 的最大值。
  // Validates: Requirements 1.1
  it('builds a complete, de-duplicated map with min(start)/max(end) per symbol', () => {
    fc.assert(
      fc.property(resourceListArb, (resources) => {
        const map = buildAvailableSymbols(resources)

        // 收集所有贡献条目（去尾点后非空），仅取键/区间端点。
        const entries = [
          ...resources.raw_bars,
          ...resources.raw_ticks,
          ...resources.derived_bars,
        ]
          .map((r) => ({ key: stripTrailingDot(r.vt_symbol), start: r.start, end: r.end }))
          .filter((e) => e.key !== '')

        const expectedKeys = new Set(entries.map((e) => e.key))

        // 完整且去重：键集合与期望完全一致，无缺失、无多余、每键恰好一次。
        expect(map.size).toBe(expectedKeys.size)
        for (const key of expectedKeys) {
          expect(map.has(key)).toBe(true)
        }
        for (const key of map.keys()) {
          expect(expectedKeys.has(key)).toBe(true)
        }

        // 每个合约：start = 所有资源 start 的最小值、end = 最大值
        // （以与实现一致的字符串比较计算）。
        for (const key of expectedKeys) {
          const group = entries.filter((e) => e.key === key)
          const expectedStart = group.reduce(
            (acc, e) => (e.start < acc ? e.start : acc),
            group[0].start,
          )
          const expectedEnd = group.reduce(
            (acc, e) => (e.end > acc ? e.end : acc),
            group[0].end,
          )
          const availability = map.get(key)!
          expect(availability.start).toBe(expectedStart)
          expect(availability.end).toBe(expectedEnd)
        }
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// 来源类型/来源周期纯函数（Properties 2-5）的共享生成器
// ---------------------------------------------------------------------------

import {
  computeAvailableSourceKinds,
  defaultSourceKind,
  computeSourceIntervalOptions,
  defaultSourceInterval,
  reconcileSelected,
  intervalToMinutes,
  computeTargetIntervalOptions,
  computeCommonRange,
  intersectRanges,
  buildAggregateRequest,
  validateAggregation,
  selectWorkspaceState,
  TARGET_CANDIDATES,
  type AvailabilityMap,
  type SymbolAvailability,
  type SourceKind,
  type TimeRange,
  type AggregationConfig,
  type AggregationValidation,
  type WorkspaceState,
} from './aggregation'

/** 候选合约键池（含尾点变体，但此处作为已归并后的键直接使用）。 */
const symbolKeyArb = fc.constantFrom(
  'AAA.SZ',
  '600000',
  'INDEX.TEST',
  'sz000415',
  'X',
  '600000.', // 与 '600000' 不同的键，覆盖尾点字面量
)

/** 随机周期集合：混入 'd'、'tick' 与分钟级，可能为空。 */
const intervalSetArb: fc.Arbitrary<Set<string>> = fc
  .array(intervalArb, { maxLength: 8 })
  .map((arr) => new Set(arr))

/** 由周期集合构造一个 SymbolAvailability（区间端点对本组属性无关，填占位）。 */
function makeAvailability(intervals: Set<string>): SymbolAvailability {
  return {
    intervals,
    start: '2020-01-01T00:00:00.000Z',
    end: '2021-01-01T00:00:00.000Z',
    intervalRanges: {},
  }
}

/** 随机 AvailabilityMap：键来自 symbolKeyArb，值由随机周期集合构造。 */
const availabilityMapArb: fc.Arbitrary<AvailabilityMap> = fc
  .array(fc.tuple(symbolKeyArb, intervalSetArb), { maxLength: 6 })
  .map((pairs) => {
    const map: AvailabilityMap = new Map<string, SymbolAvailability>()
    for (const [key, intervals] of pairs) {
      map.set(key, makeAvailability(intervals))
    }
    return map
  })

/**
 * 随机已选合约子集：可能为空、可能含 availability 中不存在的合约
 * （覆盖“任一合约缺失即交集为空”的分支），允许重复。
 */
function selectedArb(map: AvailabilityMap): fc.Arbitrary<string[]> {
  const pool = [...map.keys(), 'MISSING.SYM', 'UNKNOWN']
  return fc.array(fc.constantFrom(...pool), { maxLength: 5 })
}

/** 是否存在任一“非 tick”周期（与实现的 bar 判定一致）。 */
const hasNonTick = (intervals: Set<string>): boolean => {
  for (const interval of intervals) {
    if (interval !== 'tick') return true
  }
  return false
}

// ---------------------------------------------------------------------------
// Property 2
// ---------------------------------------------------------------------------

describe('computeAvailableSourceKinds', () => {
  // Feature: aggregation-data-driven-config, Property 2: For any 已选合约集合与可用性映射，computeAvailableSourceKinds 返回的集合中，某个 kind（bar/tick）当且仅当每一个已选合约都拥有该 kind 时才出现（bar = 存在任一非 tick 周期；tick = intervals 含 'tick'）。
  // Validates: Requirements 2.1, 2.2, 2.3
  it('returns a kind iff every selected symbol has it', () => {
    fc.assert(
      fc.property(
        availabilityMapArb.chain((map) =>
          fc.tuple(fc.constant(map), selectedArb(map)),
        ),
        ([map, selected]) => {
          const result = computeAvailableSourceKinds(selected, map)

          // 期望值：所有所选合约存在且分别拥有对应 kind 时该 kind 才出现。
          const allPresent =
            selected.length > 0 && selected.every((s) => map.has(s))
          const expectBar =
            allPresent && selected.every((s) => hasNonTick(map.get(s)!.intervals))
          const expectTick =
            allPresent && selected.every((s) => map.get(s)!.intervals.has('tick'))

          expect(result.has('bar')).toBe(expectBar)
          expect(result.has('tick')).toBe(expectTick)
          // 结果集合不含 'bar'/'tick' 之外的任何元素。
          expect(result.size).toBe((expectBar ? 1 : 0) + (expectTick ? 1 : 0))
        },
      ),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 3
// ---------------------------------------------------------------------------

describe('defaultSourceKind', () => {
  // Feature: aggregation-data-driven-config, Property 3: For any 非空 Source_Kind 集合，defaultSourceKind 返回 'bar'（若含 bar），否则返回 'tick'；空集返回 null。
  // Validates: Requirements 2.5
  it('prefers bar over tick, null for empty set', () => {
    const kindSetArb: fc.Arbitrary<Set<SourceKind>> = fc
      .subarray<SourceKind>(['bar', 'tick'])
      .map((arr) => new Set(arr))

    fc.assert(
      fc.property(kindSetArb, (kinds) => {
        const result = defaultSourceKind(kinds)
        if (kinds.size === 0) {
          expect(result).toBeNull()
        } else if (kinds.has('bar')) {
          expect(result).toBe('bar')
        } else {
          expect(result).toBe('tick')
        }
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 4
// ---------------------------------------------------------------------------

describe('computeSourceIntervalOptions', () => {
  // Feature: aggregation-data-driven-config, Property 4: For any 已选合约集合与可用性映射，computeSourceIntervalOptions 返回的每个周期均满足 intervalToMinutes 为正整数（绝不含 'd'/'tick'）且在每个已选合约 intervals 中存在；返回数组按分钟数严格升序。
  // Validates: Requirements 3.1, 3.2, 3.4
  it('returns minute-level intersection in strictly ascending order', () => {
    fc.assert(
      fc.property(
        availabilityMapArb.chain((map) =>
          fc.tuple(fc.constant(map), selectedArb(map)),
        ),
        ([map, selected]) => {
          const options = computeSourceIntervalOptions(selected, map)

          for (const interval of options) {
            const minutes = intervalToMinutes(interval)
            // (a) 分钟级：正整数，绝不含 'd'/'tick'。
            expect(minutes).not.toBeNull()
            expect(Number.isInteger(minutes)).toBe(true)
            expect(minutes as number).toBeGreaterThan(0)
            expect(interval).not.toBe('d')
            expect(interval).not.toBe('tick')
            // (b) 在每一个已选合约的 intervals 中都存在。
            for (const s of selected) {
              expect(map.get(s)!.intervals.has(interval)).toBe(true)
            }
          }

          // 严格升序（按分钟数）。
          for (let i = 1; i < options.length; i++) {
            const prev = intervalToMinutes(options[i - 1]) as number
            const curr = intervalToMinutes(options[i]) as number
            expect(curr).toBeGreaterThan(prev)
          }
        },
      ),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 5
// ---------------------------------------------------------------------------

describe('defaultSourceInterval / reconcileSelected', () => {
  // Feature: aggregation-data-driven-config, Property 5: For any 选项数组 options：非空时 defaultSourceInterval 等于分钟数最小者，reconcileSelected(current, options) 在 current ∈ options 时返回 current 否则返回最小者；空数组两者均返回 null。
  // Validates: Requirements 3.5, 3.6, 4.5
  it('default = min-minutes element; reconcile keeps valid current else min; empty -> null', () => {
    // 分钟级周期池（保证 intervalToMinutes 非 null），覆盖含重复/乱序。
    const optionsArb = fc.array(
      fc.constantFrom<string>('1m', '5m', '10m', '15m', '30m', '60m'),
      { maxLength: 6 },
    )
    // current 可能在选项内、在选项外、或为 null。
    const currentArb = fc.oneof(
      fc.constant<string | null>(null),
      fc.constantFrom('1m', '5m', '10m', '15m', '30m', '60m', 'd', 'tick', 'XX'),
    )

    fc.assert(
      fc.property(optionsArb, currentArb, (options, current) => {
        const def = defaultSourceInterval(options)
        const reconciled = reconcileSelected(current, options)

        if (options.length === 0) {
          expect(def).toBeNull()
          expect(reconciled).toBeNull()
          return
        }

        // 期望最小分钟数元素（取首次出现者，与 reduce 实现一致）。
        const minElement = options.reduce((min, c) =>
          (intervalToMinutes(c) as number) < (intervalToMinutes(min) as number) ? c : min,
        )
        expect(intervalToMinutes(def!)).toBe(intervalToMinutes(minElement))

        if (current !== null && options.includes(current)) {
          expect(reconciled).toBe(current)
        } else {
          expect(intervalToMinutes(reconciled!)).toBe(intervalToMinutes(minElement))
        }
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// 目标周期/区间/请求构造纯函数（Properties 6-9）的共享生成器
// ---------------------------------------------------------------------------

/** 随机时间区间，保证 start <= end（按 dayjs 毫秒比较）。 */
const rangeArb: fc.Arbitrary<TimeRange> = fc
  .tuple(dateArb, dateArb)
  .map(([a, b]) =>
    dayjs(a).valueOf() <= dayjs(b).valueOf() ? { start: a, end: b } : { start: b, end: a },
  )

/** 候选与非候选周期混合的来源周期生成器（含 null 与非分钟级）。 */
const sourceIntervalArb = fc.oneof(
  fc.constant<string | null>(null),
  fc.constantFrom('1m', '5m', '10m', '15m', '30m', '60m', 'd', 'tick', 'XX'),
)

/** 候选全集按分钟数升序（结果排序与子集基准）。 */
const sortedTargetCandidates = [...TARGET_CANDIDATES].sort(
  (a, b) => (intervalToMinutes(a) as number) - (intervalToMinutes(b) as number),
)

// ---------------------------------------------------------------------------
// Property 6
// ---------------------------------------------------------------------------

describe('computeTargetIntervalOptions', () => {
  // Feature: aggregation-data-driven-config, Property 6: For any sourceKind 与 sourceInterval，computeTargetIntervalOptions 的结果恒为候选集合 {5m,10m,15m,30m,60m} 的子集且升序；tick 来源等于全部候选；bar 来源且 sourceInterval 非空（分钟数 s）时恰为满足 t % s == 0 且 t > s 的候选；sourceInterval 为空时结果为空。
  // Validates: Requirements 4.1, 4.2, 4.3
  it('filters target candidates by divisibility and ordering', () => {
    fc.assert(
      fc.property(
        fc.constantFrom<SourceKind>('bar', 'tick'),
        sourceIntervalArb,
        (sourceKind, sourceInterval) => {
          const options = computeTargetIntervalOptions(sourceKind, sourceInterval)

          // 恒为候选全集的子集。
          for (const option of options) {
            expect(TARGET_CANDIDATES).toContain(option)
          }
          // 按分钟数严格升序。
          for (let i = 1; i < options.length; i++) {
            const prev = intervalToMinutes(options[i - 1]) as number
            const curr = intervalToMinutes(options[i]) as number
            expect(curr).toBeGreaterThan(prev)
          }

          if (sourceKind === 'tick') {
            // tick 来源：等于全部候选（升序）。
            expect(options).toEqual(sortedTargetCandidates)
            return
          }

          // bar 来源：
          const s = sourceInterval === null ? null : intervalToMinutes(sourceInterval)
          if (s === null) {
            // sourceInterval 为空或非分钟级 -> 无可聚合目标。
            expect(options).toEqual([])
          } else {
            const expected = sortedTargetCandidates.filter((c) => {
              const t = intervalToMinutes(c) as number
              return t % s === 0 && t > s
            })
            expect(options).toEqual(expected)
          }
        },
      ),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 7
// ---------------------------------------------------------------------------

describe('computeCommonRange', () => {
  // 场景：唯一合约键 + 随机来源 + 每合约是否拥有对应周期区间。
  const commonRangeScenarioArb = fc
    .uniqueArray(fc.constantFrom('A', 'B', 'C', 'D'), { maxLength: 4 })
    .chain((symbols) =>
      fc.record({
        symbols: fc.constant(symbols),
        sourceKind: fc.constantFrom<SourceKind>('bar', 'tick'),
        sourceInterval: fc.constantFrom('1m', '5m', '10m', '15m', '30m', '60m', null),
        perSymbol: fc.tuple(
          ...symbols.map(() => fc.record({ has: fc.boolean(), range: rangeArb })),
        ),
      }),
    )

  // Feature: aggregation-data-driven-config, Property 7: For any 已选合约集合、可用性映射、sourceKind 与 sourceInterval，若每个已选合约在对应周期（bar 用 sourceInterval、tick 用 'tick'）下都有区间，则 computeCommonRange 返回 start = 各合约 start 最大值、end = 各合约 end 最小值；若任一合约缺该区间或 max(start) > min(end)，返回 null。
  // Validates: Requirements 5.1, 5.2
  it('returns overlap [max(start), min(end)] or null', () => {
    fc.assert(
      fc.property(commonRangeScenarioArb, ({ symbols, sourceKind, sourceInterval, perSymbol }) => {
        const rangeKey = sourceKind === 'bar' ? sourceInterval : 'tick'

        // 构建可用性映射：仅当 has 且 rangeKey 非空时写入该周期区间。
        const map: AvailabilityMap = new Map<string, SymbolAvailability>()
        symbols.forEach((sym, i) => {
          const intervalRanges: Record<string, { start: string; end: string }> = {}
          if (rangeKey && perSymbol[i].has) {
            intervalRanges[rangeKey] = perSymbol[i].range
          }
          map.set(sym, {
            intervals: new Set(Object.keys(intervalRanges)),
            start: '2020-01-01T00:00:00.000Z',
            end: '2021-01-01T00:00:00.000Z',
            intervalRanges,
          })
        })

        const result = computeCommonRange(symbols, map, sourceKind, sourceInterval)

        // 独立计算期望值。
        let expected: TimeRange | null
        if (symbols.length === 0) {
          expected = null
        } else if (sourceKind === 'bar' && !sourceInterval) {
          expected = null
        } else {
          const everyHas = symbols.every((_, i) => perSymbol[i].has)
          if (!everyHas) {
            expected = null
          } else {
            let maxStart: string | null = null
            let minEnd: string | null = null
            symbols.forEach((_, i) => {
              const r = perSymbol[i].range
              if (maxStart === null || dayjs(r.start).valueOf() > dayjs(maxStart).valueOf()) {
                maxStart = r.start
              }
              if (minEnd === null || dayjs(r.end).valueOf() < dayjs(minEnd).valueOf()) {
                minEnd = r.end
              }
            })
            expected =
              dayjs(maxStart!).valueOf() > dayjs(minEnd!).valueOf()
                ? null
                : { start: maxStart!, end: minEnd! }
          }
        }

        if (expected === null) {
          expect(result).toBeNull()
        } else {
          expect(result).not.toBeNull()
          expect(dayjs(result!.start).valueOf()).toBe(dayjs(expected.start).valueOf())
          expect(dayjs(result!.end).valueOf()).toBe(dayjs(expected.end).valueOf())
        }
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 8
// ---------------------------------------------------------------------------

describe('intersectRanges', () => {
  // Feature: aggregation-data-driven-config, Property 8: For any 两个时间区间 a、b，当二者重叠时长大于 0 时 intersectRanges(a,b) 返回 [max(a.start,b.start), min(a.end,b.end)] 且非空；当重叠时长为 0（含完全不相交）时返回 null。该运算满足交换律。
  // Validates: Requirements 5.5, 5.6
  it('computes [max(start), min(end)] or null and is commutative', () => {
    fc.assert(
      fc.property(rangeArb, rangeArb, (a, b) => {
        const result = intersectRanges(a, b)

        // 期望端点（与实现的 dayjs 毫秒比较一致）。
        const expStart = dayjs(a.start).valueOf() > dayjs(b.start).valueOf() ? a.start : b.start
        const expEnd = dayjs(a.end).valueOf() < dayjs(b.end).valueOf() ? a.end : b.end

        if (dayjs(expStart).valueOf() > dayjs(expEnd).valueOf()) {
          expect(result).toBeNull()
        } else {
          expect(result).not.toBeNull()
          expect(dayjs(result!.start).valueOf()).toBe(dayjs(expStart).valueOf())
          expect(dayjs(result!.end).valueOf()).toBe(dayjs(expEnd).valueOf())
        }

        // 交换律：intersectRanges(a,b) 与 intersectRanges(b,a) 等价。
        const swapped = intersectRanges(b, a)
        if (result === null) {
          expect(swapped).toBeNull()
        } else {
          expect(swapped).not.toBeNull()
          expect(dayjs(swapped!.start).valueOf()).toBe(dayjs(result.start).valueOf())
          expect(dayjs(swapped!.end).valueOf()).toBe(dayjs(result.end).valueOf())
        }
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 9
// ---------------------------------------------------------------------------

describe('buildAggregateRequest', () => {
  // 场景：随机来源类型 + 配置 + 保证与公共区间重叠的用户区间（或不选区间）。
  const aggregateScenarioArb = fc.constantFrom<SourceKind>('bar', 'tick').chain((sourceKind) =>
    fc.record({
      sourceKind: fc.constant(sourceKind),
      selectedSymbols: fc.array(fc.string(), { maxLength: 5 }),
      sourceInterval:
        sourceKind === 'bar'
          ? fc.constantFrom('1m', '5m', '10m', '15m', '30m', '60m')
          : fc.constantFrom(null, '5m'),
      targetInterval: fc.constantFrom(...TARGET_CANDIDATES),
      sessionProfile: fc.constantFrom('', 'cn_equity', 'us_equity'),
      four: fc.tuple(dateArb, dateArb, dateArb, dateArb),
      pattern: fc.constantFrom('partial1', 'nested', 'partial2', 'equal'),
      withUser: fc.boolean(),
    }),
  )

  // Feature: aggregation-data-driven-config, Property 9: For any 通过校验的 AggregationConfig 与对应公共区间，buildAggregateRequest 产出的 DataAggregateRequest 满足：vt_symbols 等于 selectedSymbols；bar 时 source_interval 有定义、tick 时为 undefined；session_profile 在配置为空时回退 'cn_equity' 否则等于配置值；start/end 等于用户所选区间与公共区间的交集端点。
  // Validates: Requirements 1.6, 5.5, 6.3, 6.4, 7.3, 7.4, 7.5
  it('builds a DataAggregateRequest with intersected range and correct fields', () => {
    fc.assert(
      fc.property(aggregateScenarioArb, (scn) => {
        const sorted = [...scn.four].sort((a, b) => dayjs(a).valueOf() - dayjs(b).valueOf())
        const [t1, t2, t3, t4] = sorted

        // 由排序点构造保证重叠的 common/user 区间。
        let common: [string, string]
        let user: [string, string]
        switch (scn.pattern) {
          case 'partial1':
            common = [t1, t3]
            user = [t2, t4]
            break
          case 'nested':
            common = [t1, t4]
            user = [t2, t3]
            break
          case 'partial2':
            common = [t2, t4]
            user = [t1, t3]
            break
          default: // 'equal'
            common = [t1, t3]
            user = [t1, t3]
            break
        }

        const commonRange: TimeRange = { start: common[0], end: common[1] }
        const config: AggregationConfig = {
          selectedSymbols: scn.selectedSymbols,
          sourceKind: scn.sourceKind,
          sourceInterval: scn.sourceInterval,
          targetInterval: scn.targetInterval,
          range: scn.withUser ? user : null,
          sessionProfile: scn.sessionProfile,
        }

        const req = buildAggregateRequest(config, commonRange)

        // vt_symbols / source_kind / target_interval
        expect(req.vt_symbols).toEqual(scn.selectedSymbols)
        expect(req.source_kind).toBe(scn.sourceKind)
        expect(req.target_interval).toBe(scn.targetInterval)

        // source_interval：bar 时有定义且等于配置值；tick 时省略。
        if (scn.sourceKind === 'bar') {
          expect(req.source_interval).toBe(scn.sourceInterval)
        } else {
          expect(req.source_interval).toBeUndefined()
        }

        // session_profile：空回退 'cn_equity'，否则等于配置值。
        expect(req.session_profile).toBe(scn.sessionProfile || 'cn_equity')

        // start/end：用户区间与公共区间的交集端点，归一为 YYYY-MM-DD（后端为 date）。
        let expStart: string
        let expEnd: string
        if (!scn.withUser) {
          expStart = commonRange.start
          expEnd = commonRange.end
        } else {
          expStart = dayjs(user[0]).valueOf() > dayjs(common[0]).valueOf() ? user[0] : common[0]
          expEnd = dayjs(user[1]).valueOf() < dayjs(common[1]).valueOf() ? user[1] : common[1]
        }
        expect(req.start).toBe(dayjs(expStart).format('YYYY-MM-DD'))
        expect(req.end).toBe(dayjs(expEnd).format('YYYY-MM-DD'))
      }),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// 校验与工作区状态纯函数（Properties 10-11）的共享生成器
// ---------------------------------------------------------------------------

/**
 * 富可用性条目：为随机周期集合分别赋予随机区间，使 computeCommonRange
 * 可计算出非空交集（覆盖 no-range-overlap 与 valid 两类分支）。
 */
const symbolAvailabilityArb: fc.Arbitrary<SymbolAvailability> = fc
  .array(fc.tuple(intervalArb, rangeArb), { maxLength: 6 })
  .map((entries) => {
    const intervalRanges: Record<string, { start: string; end: string }> = {}
    for (const [interval, range] of entries) {
      intervalRanges[interval] = range
    }
    return {
      intervals: new Set(Object.keys(intervalRanges)),
      start: '2020-01-01T00:00:00.000Z',
      end: '2021-01-01T00:00:00.000Z',
      intervalRanges,
    }
  })

/** 随机富可用性映射：键来自 symbolKeyArb，值含分周期区间。 */
const richAvailabilityMapArb: fc.Arbitrary<AvailabilityMap> = fc
  .array(fc.tuple(symbolKeyArb, symbolAvailabilityArb), { maxLength: 6 })
  .map((pairs) => {
    const map: AvailabilityMap = new Map<string, SymbolAvailability>()
    for (const [key, avail] of pairs) {
      map.set(key, avail)
    }
    return map
  })

/**
 * 随机 AggregationConfig：覆盖空合约、缺失合约、各类 sourceKind/sourceInterval/
 * targetInterval（含 null 与越界值）、以及含/不含用户区间。
 */
function aggregationConfigArb(map: AvailabilityMap): fc.Arbitrary<AggregationConfig> {
  const pool = [...map.keys(), 'MISSING.SYM']
  return fc.record({
    selectedSymbols: fc.array(fc.constantFrom(...pool), { maxLength: 5 }),
    sourceKind: fc.constantFrom<SourceKind | null>('bar', 'tick', null),
    sourceInterval: fc.constantFrom<string | null>(
      '1m', '5m', '10m', '15m', '30m', '60m', 'd', 'tick', null,
    ),
    targetInterval: fc.constantFrom<string | null>(
      '5m', '10m', '15m', '30m', '60m', '1m', null,
    ),
    range: fc.oneof(
      fc.constant<[string, string] | null>(null),
      rangeArb.map((r) => [r.start, r.end] as [string, string]),
    ),
    sessionProfile: fc.constantFrom('', 'cn_equity'),
  })
}

/**
 * 独立按固定优先级再推导期望的校验结果，镜像
 * `no-symbol → no-common-source → no-source-interval → no-target → no-range-overlap`。
 * 复用 Properties 2-9 已分别验证的子函数，仅在此独立组合并施加优先级顺序。
 */
function expectedValidation(
  config: AggregationConfig,
  availability: AvailabilityMap,
): AggregationValidation {
  if (config.selectedSymbols.length === 0) {
    return { valid: false, reason: 'no-symbol' }
  }
  const kinds = computeAvailableSourceKinds(config.selectedSymbols, availability)
  if (kinds.size === 0) {
    return { valid: false, reason: 'no-common-source' }
  }
  if (config.sourceKind === null || !kinds.has(config.sourceKind)) {
    return { valid: false, reason: 'no-common-source' }
  }
  if (config.sourceKind === 'bar') {
    const sourceOptions = computeSourceIntervalOptions(config.selectedSymbols, availability)
    if (sourceOptions.length === 0) {
      return { valid: false, reason: 'no-source-interval' }
    }
    if (config.sourceInterval === null || !sourceOptions.includes(config.sourceInterval)) {
      return { valid: false, reason: 'no-source-interval' }
    }
  }
  const targetOptions = computeTargetIntervalOptions(config.sourceKind, config.sourceInterval)
  if (targetOptions.length === 0) {
    return { valid: false, reason: 'no-target' }
  }
  if (config.targetInterval === null || !targetOptions.includes(config.targetInterval)) {
    return { valid: false, reason: 'no-target' }
  }
  const common = computeCommonRange(
    config.selectedSymbols,
    availability,
    config.sourceKind,
    config.sourceInterval,
  )
  if (common === null) {
    return { valid: false, reason: 'no-range-overlap' }
  }
  if (config.range !== null) {
    const userRange: TimeRange = { start: config.range[0], end: config.range[1] }
    if (intersectRanges(userRange, common) === null) {
      return { valid: false, reason: 'no-range-overlap' }
    }
  }
  return { valid: true, reason: null }
}

// ---------------------------------------------------------------------------
// Property 10
// ---------------------------------------------------------------------------

describe('validateAggregation', () => {
  // Feature: aggregation-data-driven-config, Property 10: For any AggregationConfig 与可用性映射，validateAggregation 的结果满足：当且仅当配置为 Valid_Aggregation_Combination 时 valid===true；当 valid===false 时，reason 指向按固定优先级 no-symbol → no-common-source → no-source-interval → no-target → no-range-overlap 命中的第一个无效维度。
  // Validates: Requirements 1.7, 2.6, 3.7, 4.6, 5.6, 7.1, 7.2
  it('returns valid iff combination is valid, else first invalid dimension by fixed priority', () => {
    fc.assert(
      fc.property(
        richAvailabilityMapArb.chain((map) =>
          fc.tuple(fc.constant(map), aggregationConfigArb(map)),
        ),
        ([map, config]) => {
          const result = validateAggregation(config, map)
          const expected = expectedValidation(config, map)

          // valid 标志与第一个命中的 reason 必须与独立再推导一致。
          expect(result.valid).toBe(expected.valid)
          expect(result.reason).toBe(expected.reason)

          // valid 与 reason 的耦合：有效时 reason 为 null，无效时 reason 非 null。
          if (result.valid) {
            expect(result.reason).toBeNull()
          } else {
            expect(result.reason).not.toBeNull()
          }
        },
      ),
      { numRuns: 100 },
    )
  })
})

// ---------------------------------------------------------------------------
// Property 11
// ---------------------------------------------------------------------------

describe('selectWorkspaceState', () => {
  // Feature: aggregation-data-driven-config, Property 11: For any 输入三元组 (isLoading, hasError, symbolCount)，selectWorkspaceState 恰好返回 loading/error/empty/ready 之一，且遵循优先级 loading > error > empty(symbolCount===0) > ready。
  // Validates: Requirements 8.1, 8.3, 8.4, 8.5
  it('returns exactly one mutually-exclusive state by fixed priority loading > error > empty > ready', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        fc.oneof(fc.constant(0), fc.integer({ min: 1, max: 50 })),
        (isLoading, hasError, symbolCount) => {
          const result = selectWorkspaceState(isLoading, hasError, symbolCount)

          // 独立按优先级再推导期望状态。
          let expected: WorkspaceState
          if (isLoading) {
            expected = 'loading'
          } else if (hasError) {
            expected = 'error'
          } else if (symbolCount === 0) {
            expected = 'empty'
          } else {
            expected = 'ready'
          }

          expect(result).toBe(expected)
          // 恰好为四态之一。
          expect(['loading', 'error', 'empty', 'ready']).toContain(result)
        },
      ),
      { numRuns: 100 },
    )
  })
})

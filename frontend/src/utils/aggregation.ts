// 本地聚合工作区：数据驱动配置的类型与常量定义
//
// 本文件承载“本地聚合工作区”联动逻辑所需的类型与常量。所有联动计算
// （交集、整除过滤、区间重叠、有效性校验等）将以纯函数形式在后续任务中
// 实现于此，与 React 组件解耦，便于单元测试与基于属性的测试（fast-check）。
//
// 复用现有的 `DataAggregateRequest`、`DataResourceList`（来自 ../types/alpha，
// 保持不变），不引入新的后端契约。

import dayjs from 'dayjs'

import type { DataAggregateRequest, DataResourceList } from '../types/alpha'

// 重新导出，便于本模块消费者从单一入口引用聚合相关契约类型。
export type { DataAggregateRequest, DataResourceList }

// ---------------------------------------------------------------------------
// 基础类型
// ---------------------------------------------------------------------------

/** 聚合来源类型：`bar`（原始/派生 K 线）或 `tick`（历史 Tick）。 */
export type SourceKind = 'bar' | 'tick'

/** 闭区间 [start, end]；两端均为可被 dayjs 解析的时间字符串（ISO 或 "YYYY-MM-DD HH:mm:ss"）。 */
export interface TimeRange {
  /** 区间起点（含），语义上不晚于 end。 */
  start: string
  /** 区间终点（含），语义上不早于 start。 */
  end: string
}

// ---------------------------------------------------------------------------
// 每合约可用性映射（Hook 与纯函数共享）
// ---------------------------------------------------------------------------

/**
 * 单个合约的本地数据可用性。
 *
 * 由 `useAvailableSymbols` 基于 `raw_bars`、`raw_ticks`（视为 `interval='tick'`）、
 * `derived_bars`（`interval = target_interval || interval`）归并而成，供 Hook
 * 与纯函数共享。
 */
export interface SymbolAvailability {
  /** 该合约本地存在的所有周期（含 'tick'、'd'、分钟级、派生周期）。 */
  intervals: Set<string>
  /** 该合约所有资源中最早的 start。 */
  start: string
  /** 该合约所有资源中最晚的 end。 */
  end: string
  /** 按周期记录的可用区间。 */
  intervalRanges: Record<string, { start: string; end: string }>
}

/** 去除尾点后的 vt_symbol -> 可用性映射。 */
export type AvailabilityMap = Map<string, SymbolAvailability>

// ---------------------------------------------------------------------------
// 工作区配置状态（受控）
// ---------------------------------------------------------------------------

/** 本地聚合工作区的受控配置状态。 */
export interface AggregationConfig {
  /** 已选合约代码列表（写入 Aggregate_Request 的 vt_symbols）。 */
  selectedSymbols: string[]
  /** 来源类型；未确定时为 null。 */
  sourceKind: SourceKind | null
  /** 来源周期；仅 bar 来源时有效，tick 时为 null。 */
  sourceInterval: string | null
  /** 目标周期。 */
  targetInterval: string | null
  /** 用户所选时间范围 [start, end]；未选时为 null。 */
  range: [string, string] | null
  /** 时段规则，默认 'cn_equity'。 */
  sessionProfile: string
}

// ---------------------------------------------------------------------------
// 派生选项集合（供渲染）
// ---------------------------------------------------------------------------

/** 基于 availability 与已选合约计算出的派生选项集合。 */
export interface DerivedOptions {
  /** 可用来源类型集合。 */
  availableSourceKinds: Set<SourceKind>
  /** 来源周期选项，按分钟数升序。 */
  sourceIntervalOptions: string[]
  /** 目标周期选项，按分钟数升序。 */
  targetIntervalOptions: string[]
  /** 公共可用时间区间；无重叠时为 null。 */
  commonRange: TimeRange | null
}

// ---------------------------------------------------------------------------
// 校验
// ---------------------------------------------------------------------------

/** 无效聚合组合的具体维度，用于定向提示。 */
export type InvalidDimension =
  | 'no-symbol' // 未选合约
  | 'no-common-source' // 无公共 Source_Kind
  | 'no-source-interval' // bar 下无公共分钟级来源周期
  | 'no-target' // 无可聚合目标周期
  | 'no-range-overlap' // 用户区间与公共区间零重叠

/** 聚合配置的校验结果。 */
export interface AggregationValidation {
  /** 配置是否为可提交的有效聚合组合；所有维度均通过时为 true。 */
  valid: boolean
  /** 第一个命中的无效维度，用于定向提示；valid 时为 null。 */
  reason: InvalidDimension | null
}

// ---------------------------------------------------------------------------
// 工作区降级状态
// ---------------------------------------------------------------------------

/** 工作区四态：互斥，优先级 loading > error > empty > ready。 */
export type WorkspaceState = 'loading' | 'error' | 'empty' | 'ready'

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------

/** 分钟级周期 -> 分钟数映射；非分钟级（如 'd'、'tick'）不在此表中。 */
export const INTERVAL_MINUTES: Record<string, number> = {
  '1m': 1,
  '5m': 5,
  '10m': 10,
  '15m': 15,
  '30m': 30,
  '60m': 60,
}

/** 目标周期候选全集（Requirement 4.1）。 */
export const TARGET_CANDIDATES: readonly string[] = ['5m', '10m', '15m', '30m', '60m']

/** 受支持的时段规则集合（Requirement 6.2）。 */
export const SUPPORTED_SESSION_PROFILES: readonly string[] = ['cn_equity']

// ---------------------------------------------------------------------------
// 来源类型与来源周期纯函数（Task 2.1）
// ---------------------------------------------------------------------------

/**
 * 将周期字符串换算为分钟数。
 *
 * 分钟级周期（见 {@link INTERVAL_MINUTES}）返回对应的正整数分钟数；
 * 非分钟级周期（如 `'d'`、`'tick'`）或未知值返回 `null`。
 *
 * Requirements: 3.1, 3.2
 */
export function intervalToMinutes(interval: string): number | null {
  const minutes = INTERVAL_MINUTES[interval]
  return minutes ?? null
}

/**
 * 计算所选合约均拥有的 Source_Kind 集合（取交集）。
 *
 * - `bar` 命中：该合约存在任一“非 tick”周期（含 `'d'`、分钟级与派生周期）。
 * - `tick` 命中：该合约 intervals 含 `'tick'`。
 *
 * 某个 kind 仅当**每一个**已选合约都拥有该 kind 时才出现在结果中。
 * 已选合约为空、或任一合约在 availability 中缺失时，返回空集。
 *
 * Requirements: 2.1, 2.2, 2.3
 */
export function computeAvailableSourceKinds(
  selected: string[],
  availability: AvailabilityMap,
): Set<SourceKind> {
  const result = new Set<SourceKind>()
  if (selected.length === 0) {
    return result
  }

  const hasBar = (intervals: Set<string>): boolean => {
    for (const interval of intervals) {
      if (interval !== 'tick') {
        return true
      }
    }
    return false
  }

  let barForAll = true
  let tickForAll = true

  for (const symbol of selected) {
    const avail = availability.get(symbol)
    if (!avail) {
      // 任一所选合约缺失可用性数据，交集为空。
      return new Set<SourceKind>()
    }
    if (!hasBar(avail.intervals)) {
      barForAll = false
    }
    if (!avail.intervals.has('tick')) {
      tickForAll = false
    }
  }

  if (barForAll) {
    result.add('bar')
  }
  if (tickForAll) {
    result.add('tick')
  }
  return result
}

/**
 * 按固定优先级 `['bar', 'tick']` 取集合中最靠前者作为默认 Source_Kind。
 *
 * 含 `bar` 时返回 `'bar'`，否则含 `tick` 时返回 `'tick'`，空集返回 `null`。
 *
 * Requirements: 2.5
 */
export function defaultSourceKind(kinds: Set<SourceKind>): SourceKind | null {
  const priority: readonly SourceKind[] = ['bar', 'tick']
  for (const kind of priority) {
    if (kinds.has(kind)) {
      return kind
    }
  }
  return null
}

/**
 * 计算所选合约在 bar 来源下的公共分钟级 Source_Interval 选项。
 *
 * 取各所选合约“分钟级”周期集合的交集（通过 {@link intervalToMinutes} 过滤，
 * 因此绝不包含 `'d'`、`'tick'` 等非分钟级周期），并按分钟数升序排列。
 *
 * 已选合约为空、或任一合约在 availability 中缺失时，返回空数组。
 *
 * Requirements: 3.1, 3.2, 3.4
 */
export function computeSourceIntervalOptions(
  selected: string[],
  availability: AvailabilityMap,
): string[] {
  if (selected.length === 0) {
    return []
  }

  // 收集每个所选合约的分钟级周期集合，过程中校验合约存在性。
  const perSymbolMinuteIntervals: Set<string>[] = []
  for (const symbol of selected) {
    const avail = availability.get(symbol)
    if (!avail) {
      // 任一所选合约缺失可用性数据，交集为空。
      return []
    }
    const minuteIntervals = new Set<string>()
    for (const interval of avail.intervals) {
      if (intervalToMinutes(interval) !== null) {
        minuteIntervals.add(interval)
      }
    }
    perSymbolMinuteIntervals.push(minuteIntervals)
  }

  // 以第一个合约的分钟级周期为基准，逐一与其余合约求交集。
  const [first, ...rest] = perSymbolMinuteIntervals
  const intersection: string[] = []
  for (const interval of first) {
    if (rest.every((set) => set.has(interval))) {
      intersection.push(interval)
    }
  }

  // 按分钟数升序排列（分钟数恒为正整数，故 non-null）。
  intersection.sort((a, b) => (intervalToMinutes(a) as number) - (intervalToMinutes(b) as number))
  return intersection
}

/**
 * 取选项中分钟数最小（最细）的周期作为默认 Source_Interval。
 *
 * 空数组返回 `null`。
 *
 * Requirements: 3.5, 3.6
 */
export function defaultSourceInterval(options: string[]): string | null {
  if (options.length === 0) {
    return null
  }
  return options.reduce((min, current) =>
    (intervalToMinutes(current) as number) < (intervalToMinutes(min) as number) ? current : min,
  )
}

/**
 * 归一化当前选中的周期：在选项内则保留，否则取分钟数最小者。
 *
 * 空数组返回 `null`。用于在选项集合变化时归一化越界选择
 * （来源周期与目标周期共用此逻辑）。
 *
 * Requirements: 3.6, 4.5
 */
export function reconcileSelected(current: string | null, options: string[]): string | null {
  if (options.length === 0) {
    return null
  }
  if (current !== null && options.includes(current)) {
    return current
  }
  return defaultSourceInterval(options)
}

// ---------------------------------------------------------------------------
// 目标周期与时间区间纯函数（Task 3.1）
// ---------------------------------------------------------------------------

/**
 * 从 {@link TARGET_CANDIDATES} 过滤出可聚合的目标周期选项。
 *
 * - `tick` 来源：返回全部候选。
 * - `bar` 来源且 `sourceInterval` 非空（设其分钟数为 s）：保留候选中所有满足
 *   `t % s == 0 && t > s` 的周期（即目标分钟数为来源分钟数的整数倍且严格更粗）。
 * - `bar` 来源且 `sourceInterval` 为空/null：返回 `[]`。
 *
 * 结果恒为 {@link TARGET_CANDIDATES} 的子集，并按分钟数升序排列。
 *
 * Requirements: 4.1, 4.2, 4.3
 */
export function computeTargetIntervalOptions(
  sourceKind: SourceKind,
  sourceInterval: string | null,
): string[] {
  // 候选全集按分钟数升序，作为结果排序与子集基准。
  const sortedCandidates = [...TARGET_CANDIDATES].sort(
    (a, b) => (intervalToMinutes(a) as number) - (intervalToMinutes(b) as number),
  )

  if (sourceKind === 'tick') {
    return sortedCandidates
  }

  // bar 来源：sourceInterval 为空/null 时无可聚合目标。
  if (!sourceInterval) {
    return []
  }
  const s = intervalToMinutes(sourceInterval)
  if (s === null) {
    return []
  }

  return sortedCandidates.filter((candidate) => {
    const t = intervalToMinutes(candidate) as number
    return t % s === 0 && t > s
  })
}

/**
 * 计算所选合约在指定来源下的公共可用时间区间（重叠区间）。
 *
 * - bar 来源使用 `intervalRanges[sourceInterval]`；tick 来源使用 `intervalRanges['tick']`。
 * - 重叠区间 = `[max(start), min(end)]`（各合约 start 的最大值到 end 的最小值）。
 *
 * 以下情形返回 `null`：
 * - `selected` 为空；
 * - bar 来源但 `sourceInterval` 为 null；
 * - 任一所选合约在 availability 中缺失，或缺少对应周期的区间；
 * - `max(start) > min(end)`（无重叠）。
 *
 * 时间比较内部统一用 `dayjs(x).valueOf()`。
 *
 * Requirements: 5.1, 5.2
 */
export function computeCommonRange(
  selected: string[],
  availability: AvailabilityMap,
  sourceKind: SourceKind,
  sourceInterval: string | null,
): TimeRange | null {
  if (selected.length === 0) {
    return null
  }

  // 确定用于查找区间的周期键：bar 用 sourceInterval、tick 用 'tick'。
  let rangeKey: string
  if (sourceKind === 'bar') {
    if (!sourceInterval) {
      return null
    }
    rangeKey = sourceInterval
  } else {
    rangeKey = 'tick'
  }

  let maxStart: string | null = null
  let minEnd: string | null = null

  for (const symbol of selected) {
    const avail = availability.get(symbol)
    if (!avail) {
      return null
    }
    const range = avail.intervalRanges[rangeKey]
    if (!range) {
      return null
    }
    if (maxStart === null || dayjs(range.start).valueOf() > dayjs(maxStart).valueOf()) {
      maxStart = range.start
    }
    if (minEnd === null || dayjs(range.end).valueOf() < dayjs(minEnd).valueOf()) {
      minEnd = range.end
    }
  }

  if (maxStart === null || minEnd === null) {
    return null
  }
  if (dayjs(maxStart).valueOf() > dayjs(minEnd).valueOf()) {
    return null
  }
  return { start: maxStart, end: minEnd }
}

/**
 * 计算两个时间区间的交集。
 *
 * 返回 `[max(a.start, b.start), min(a.end, b.end)]`；当二者零重叠
 * （即 `startMs > endMs`，重叠时长为零或为负）时返回 `null`。
 *
 * 满足交换律：`intersectRanges(a, b)` 与 `intersectRanges(b, a)` 等价。
 * 时间比较内部统一用 `dayjs(x).valueOf()`。
 *
 * Requirements: 5.5, 5.6
 */
export function intersectRanges(a: TimeRange, b: TimeRange): TimeRange | null {
  const start = dayjs(a.start).valueOf() > dayjs(b.start).valueOf() ? a.start : b.start
  const end = dayjs(a.end).valueOf() < dayjs(b.end).valueOf() ? a.end : b.end
  if (dayjs(start).valueOf() > dayjs(end).valueOf()) {
    return null
  }
  return { start, end }
}

// ---------------------------------------------------------------------------
// 请求组装纯函数（Task 3.2）
// ---------------------------------------------------------------------------

/**
 * 将已通过校验的 {@link AggregationConfig} 与对应公共可用区间组装为
 * {@link DataAggregateRequest}。
 *
 * - `vt_symbols` 等于 `config.selectedSymbols`（Requirement 1.6）。
 * - `source_kind` 取自 `config.sourceKind`（已通过校验时非空）。
 * - `source_interval`：bar 来源时包含 `config.sourceInterval`，tick 来源时省略
 *   （Requirement 7.4, 7.5）。
 * - `target_interval` 取自 `config.targetInterval`。
 * - `session_profile`：`config.sessionProfile` 为空时回退为 `'cn_equity'`，
 *   否则使用配置值（Requirement 6.3, 6.4）。
 * - `start`/`end`：用户所选区间（`config.range`）与公共可用区间 `commonRange`
 *   的交集端点；用户未选区间时退化为 `commonRange`（Requirement 5.5）。
 *
 * 产出结构与既有 {@link DataAggregateRequest} 保持不变。
 *
 * Requirements: 1.6, 5.5, 6.3, 6.4, 7.4, 7.5
 */
export function buildAggregateRequest(
  config: AggregationConfig,
  commonRange: TimeRange,
): DataAggregateRequest {
  // start/end 取用户选区与公共区间的交集端点；未选区间时退化为公共区间。
  const userRange: TimeRange | null = config.range
    ? { start: config.range[0], end: config.range[1] }
    : null
  const effectiveRange = userRange ? intersectRanges(userRange, commonRange) : commonRange
  // 已通过校验的配置保证存在非空交集；防御性退回公共区间。
  const range = effectiveRange ?? commonRange

  // session_profile 为空时回退默认时段规则。
  const sessionProfile = config.sessionProfile || 'cn_equity'

  const request: DataAggregateRequest = {
    vt_symbols: config.selectedSymbols,
    // 后端 DataAggregateRequest.start/end 为 pydantic `date`，而可用区间来源于
    // 资源的 datetime 字符串（如 "2025-01-01 09:30:00"），须归一为 YYYY-MM-DD，
    // 否则后端无法解析 date 而返回 422。
    start: dayjs(range.start).format('YYYY-MM-DD'),
    end: dayjs(range.end).format('YYYY-MM-DD'),
    source_kind: config.sourceKind as SourceKind,
    target_interval: config.targetInterval as string,
    session_profile: sessionProfile,
  }

  // source_interval 仅在 bar 来源时包含，tick 来源时省略。
  if (config.sourceKind === 'bar' && config.sourceInterval) {
    request.source_interval = config.sourceInterval
  }

  return request
}

// ---------------------------------------------------------------------------
// 校验与工作区状态纯函数（Task 4.1）
// ---------------------------------------------------------------------------

/**
 * 综合校验当前配置是否为 Valid_Aggregation_Combination，并在无效时定位
 * 第一个命中的无效维度。
 *
 * 按固定优先级依次检查（命中即返回，后续维度不再检查）：
 * 1. `no-symbol`：未选任何合约（Requirement 1.7）。
 * 2. `no-common-source`：所选合约无公共可用来源类型，或当前选定的来源类型
 *    不在公共可用集合中（Requirement 2.6）。
 * 3. `no-source-interval`：bar 来源下无公共分钟级来源周期，或当前选定的来源
 *    周期不在可用选项中（Requirement 3.7）。
 * 4. `no-target`：当前来源下无可聚合的目标周期，或当前选定的目标周期不在
 *    可用选项中（Requirement 4.6）。
 * 5. `no-range-overlap`：所选合约无公共可用区间，或用户所选区间与公共可用
 *    区间零重叠（Requirement 5.6）。
 *
 * 仅当所有维度均通过时返回 `{ valid: true, reason: null }`。
 *
 * Requirements: 1.7, 2.6, 3.7, 4.6, 5.6, 7.1, 7.2
 */
export function validateAggregation(
  config: AggregationConfig,
  availability: AvailabilityMap,
): AggregationValidation {
  // 1. no-symbol：未选任何合约。
  if (config.selectedSymbols.length === 0) {
    return { valid: false, reason: 'no-symbol' }
  }

  // 2. no-common-source：无公共可用来源类型，或选定类型不在公共集合中。
  const availableKinds = computeAvailableSourceKinds(config.selectedSymbols, availability)
  if (availableKinds.size === 0) {
    return { valid: false, reason: 'no-common-source' }
  }
  if (config.sourceKind === null || !availableKinds.has(config.sourceKind)) {
    return { valid: false, reason: 'no-common-source' }
  }

  // 3. no-source-interval：bar 来源下无公共分钟级来源周期，或选定值越界。
  if (config.sourceKind === 'bar') {
    const sourceOptions = computeSourceIntervalOptions(config.selectedSymbols, availability)
    if (sourceOptions.length === 0) {
      return { valid: false, reason: 'no-source-interval' }
    }
    if (config.sourceInterval === null || !sourceOptions.includes(config.sourceInterval)) {
      return { valid: false, reason: 'no-source-interval' }
    }
  }

  // 4. no-target：当前来源下无可聚合目标周期，或选定值越界。
  const targetOptions = computeTargetIntervalOptions(config.sourceKind, config.sourceInterval)
  if (targetOptions.length === 0) {
    return { valid: false, reason: 'no-target' }
  }
  if (config.targetInterval === null || !targetOptions.includes(config.targetInterval)) {
    return { valid: false, reason: 'no-target' }
  }

  // 5. no-range-overlap：无公共可用区间，或用户区间与公共区间零重叠。
  const commonRange = computeCommonRange(
    config.selectedSymbols,
    availability,
    config.sourceKind,
    config.sourceInterval,
  )
  if (commonRange === null) {
    return { valid: false, reason: 'no-range-overlap' }
  }
  if (config.range !== null) {
    const userRange: TimeRange = { start: config.range[0], end: config.range[1] }
    if (intersectRanges(userRange, commonRange) === null) {
      return { valid: false, reason: 'no-range-overlap' }
    }
  }

  return { valid: true, reason: null }
}

/**
 * 将页面数据资源查询的状态归一为单一互斥的工作区状态。
 *
 * 优先级固定为 `loading > error > empty > ready`：
 * - `isLoading` 为真 → `'loading'`；
 * - 否则 `hasError` 为真 → `'error'`；
 * - 否则 `symbolCount === 0` → `'empty'`；
 * - 否则 → `'ready'`。
 *
 * Requirements: 8.1, 8.3, 8.4, 8.5
 */
export function selectWorkspaceState(
  isLoading: boolean,
  hasError: boolean,
  symbolCount: number,
): WorkspaceState {
  if (isLoading) {
    return 'loading'
  }
  if (hasError) {
    return 'error'
  }
  if (symbolCount === 0) {
    return 'empty'
  }
  return 'ready'
}

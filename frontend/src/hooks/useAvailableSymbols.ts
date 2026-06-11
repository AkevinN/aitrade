// useAvailableSymbols：从本地数据资源归并出“每合约可用性映射”。
//
// 本 Hook 抽取自 `frontend/src/pages/CNNTrain/index.tsx` 中既有的
// `availableSymbols` useMemo 逻辑，供数据准备页“本地聚合工作区”与 CNNTrain
// 共用，消除重复（见 design.md 的 useAvailableSymbols 设计）。
//
// 设计将映射构建逻辑抽为可被直接调用的**纯函数** `buildAvailableSymbols`，
// 以便基于属性的测试（task 6.2），并由 `useAvailableSymbols` 以 useMemo 包装。

import { useMemo } from 'react'

import type { DataResourceList } from '../types/alpha'
import type { AvailabilityMap, SymbolAvailability } from '../utils/aggregation'

/** 单条被归并的资源记录（仅取构建映射所需字段）。 */
interface ResourceLike {
  vt_symbol: string
  interval: string
  start: string
  end: string
}

/**
 * 把 `raw_bars`、`raw_ticks`（视为 `interval='tick'`）、`derived_bars`
 * （`interval = target_interval || interval`）归并为
 * `Map<vt_symbol, SymbolAvailability>`（key 去除尾点）。
 *
 * - `start` 取该合约所有资源中的最小值、`end` 取最大值。
 * - `intervalRanges` 按周期累计各自的可用区间。
 *
 * 与 CNNTrain 现有 `availableSymbols` 的归并逻辑等价；`resources` 为
 * `undefined` 时返回空映射。
 *
 * 该函数为纯函数（不读取/修改外部状态），便于属性测试。
 *
 * Requirements: 1.1
 */
export function buildAvailableSymbols(
  resources: DataResourceList | undefined,
): AvailabilityMap {
  const symbolMap: AvailabilityMap = new Map<string, SymbolAvailability>()
  if (!resources) {
    return symbolMap
  }

  const addResource = (item: ResourceLike): void => {
    const sym = item.vt_symbol.replace(/\.$/, '')
    if (!sym) return
    const existing = symbolMap.get(sym)
    if (existing) {
      existing.intervals.add(item.interval)
      if (item.start < existing.start) existing.start = item.start
      if (item.end > existing.end) existing.end = item.end
      const current = existing.intervalRanges[item.interval]
      if (!current) {
        existing.intervalRanges[item.interval] = { start: item.start, end: item.end }
      } else {
        if (item.start < current.start) current.start = item.start
        if (item.end > current.end) current.end = item.end
      }
    } else {
      symbolMap.set(sym, {
        intervals: new Set([item.interval]),
        start: item.start,
        end: item.end,
        intervalRanges: {
          [item.interval]: { start: item.start, end: item.end },
        },
      })
    }
  }

  resources.raw_bars.forEach((b) => addResource(b))
  resources.raw_ticks.forEach((t) => addResource({ ...t, interval: 'tick' }))
  resources.derived_bars.forEach((d) =>
    addResource({ ...d, interval: d.target_interval || d.interval }),
  )

  return symbolMap
}

/**
 * `buildAvailableSymbols` 的 React Hook 包装：以 `useMemo` 缓存映射结果，
 * 仅在 `resources` 引用变化时重新计算。
 *
 * Requirements: 1.1
 */
export function useAvailableSymbols(
  resources: DataResourceList | undefined,
): AvailabilityMap {
  return useMemo(() => buildAvailableSymbols(resources), [resources])
}

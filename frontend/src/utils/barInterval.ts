/**
 * K 线周期 ↔ bar 根数 / 交易日 的换算（前端单一事实源）。
 *
 * 与后端 `aitrade/cnn/intervals.py` 镜像：A 股每个完整交易日 4 小时连续竞价
 * = 240 分钟，各周期每交易日的 bar 数固定。供「按天配置、自动换算 bar 根数」的
 * 表单（lookback / 预测跨度 / OCO 最大持有等）共用，避免常量在各页重复。
 */

/** 各周期每个交易日的 bar 数。派生/自定义周期不在表内（换算时回退手填）。 */
export const BARS_PER_TRADING_DAY: Record<string, number> = {
  d: 1,
  '60m': 4,
  '30m': 8,
  '15m': 16,
  '10m': 24,
  '5m': 48,
  '1m': 240,
}

/**
 * 返回该周期每个交易日的 bar 数；周期不在换算表内时返回 null（无法自动换算）。
 *
 * @param interval - K 线周期，如 'd' / '30m' / '1m'
 * @returns 每交易日 bar 数；未知周期返回 null
 */
export function barsPerDay(interval: string): number | null {
  return BARS_PER_TRADING_DAY[interval] ?? null
}

/**
 * 把「观测交易日数」换算成 bar 根数（days × 每日 bar 数）。
 *
 * @param days - 交易日数
 * @param interval - K 线周期
 * @returns bar 根数（>= 1）；周期不在换算表内时返回 null
 *
 * @example
 * ```ts
 * daysToBars(30, '30m') // 30 × 8 = 240
 * daysToBars(5, 'd')    // 5
 * ```
 */
export function daysToBars(days: number, interval: string): number | null {
  const bpd = barsPerDay(interval)
  if (bpd == null) return null
  return Math.max(1, Math.round(days * bpd))
}

/**
 * 把 bar 根数向上取整换算成交易日数（与后端 bars_to_days 同口径，ceil）。
 *
 * @param bars - bar 根数
 * @param interval - K 线周期
 * @returns 交易日数（>= 1）；周期不在换算表内时返回 null
 *
 * @example
 * ```ts
 * barsToDays(10, '30m') // ceil(10/8) = 2
 * barsToDays(5, 'd')    // 5
 * ```
 */
export function barsToDays(bars: number, interval: string): number | null {
  const bpd = barsPerDay(interval)
  if (bpd == null) return null
  if (bars <= 0) return 0
  return Math.max(1, Math.ceil(bars / bpd))
}

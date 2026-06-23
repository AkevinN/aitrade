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

/** A 股每个完整交易日的连续竞价分钟数（4 小时）；与 BARS_PER_TRADING_DAY 同一口径。 */
const MINUTES_PER_TRADING_DAY = 240

/**
 * 支持的分钟级周期 → 每根 bar 的分钟数，由 BARS_PER_TRADING_DAY 反推（240 / 每日 bar 数），
 * 不含日线/派生周期。用于从时间戳间隔就近吸附出周期。降序排列便于稳定吸附。
 */
const MINUTE_INTERVALS: ReadonlyArray<readonly [string, number]> = Object.entries(
  BARS_PER_TRADING_DAY,
)
  .filter(([key]) => key !== 'd')
  .map(([key, bpd]) => [key, MINUTES_PER_TRADING_DAY / bpd] as const)
  .sort((a, b) => b[1] - a[1])

/** 解析 ISO 时间戳串为 {当天分钟数, 日期键}；纯日期/无时刻部分按零点处理。非法串返回 null。 */
function parseDayMinute(value: string): { dayKey: string; minuteOfDay: number } | null {
  const date = /^(\d{4})-(\d{2})-(\d{2})/.exec(value)
  if (!date) return null
  const time = /[T ](\d{2}):(\d{2})/.exec(value)
  const minuteOfDay = time ? Number(time[1]) * 60 + Number(time[2]) : 0
  return { dayKey: date[0], minuteOfDay }
}

/**
 * 从一组成交/行情时间戳反推回测实际所用的 K 线周期（"bar 是什么图就用什么图"）。
 *
 * 取**同一交易日内相邻时间戳间隔（分钟）的最大公约数**，就近吸附到支持的分钟周期
 * {1m/5m/10m/15m/30m/60m}。成交都落在 bar 网格上，各间隔皆为真实 bar 的整数倍，故其 GCD
 * 收敛到真实 bar——比"取最小间隔"更稳健（成交稀疏、相邻 bar 无成交时最小间隔会偏大）。
 * 午休、跨日等大缺口同为整数倍，不影响 GCD。所有时间戳都落在零点（纯日期或 00:00）时
 * 判定为日线，返回 ``'d'``。
 *
 * 仅作回测周期的**数据驱动兜底**：当报告未显式回显回测周期时，用成交流水自身的时间粒度
 * 决定 K 线该按什么周期拉行情，从而绝不把分钟级标的硬拉成日线而报错。结果为启发式
 * （样本足够时可靠），调用方应优先采用显式的回测配置周期。
 *
 * @param datetimes - ISO 时间戳串数组（如成交流水的 datetime 列）；null/undefined/非法值被跳过
 * @returns 反推的周期（``'d'`` | ``'1m'`` | ``'5m'`` | ``'10m'`` | ``'15m'`` | ``'30m'`` | ``'60m'``）；
 *   无有效时刻样本或找不到同日相邻对而无法判定时返回 ``null``
 *
 * @example
 * ```ts
 * inferIntervalFromDatetimes(['2024-10-09T10:30:00', '2024-10-09T11:00:00']) // '30m'
 * inferIntervalFromDatetimes(['2024-10-09', '2024-10-10'])                    // 'd'
 * inferIntervalFromDatetimes(['2024-10-09T10:00:00'])                        // null（无相邻对）
 * ```
 */
export function inferIntervalFromDatetimes(
  datetimes: ReadonlyArray<string | null | undefined>,
): string | null {
  // 1) 解析为 {日期键, 当天分钟数}，跳过非法/空值。
  const parsed = datetimes
    .map((v) => (typeof v === 'string' ? parseDayMinute(v) : null))
    .filter((x): x is { dayKey: string; minuteOfDay: number } => x !== null)
  if (parsed.length === 0) return null

  // 2) 全部落在零点 → 日线。
  if (parsed.every((p) => p.minuteOfDay === 0)) return 'd'

  // 3) 按日分组，取各日内相邻时刻间隔的全局 GCD（成交落在 bar 网格上 → 间隔皆为真实 bar 整数倍）。
  const byDay = new Map<string, number[]>()
  for (const { dayKey, minuteOfDay } of parsed) {
    const arr = byDay.get(dayKey)
    if (arr) arr.push(minuteOfDay)
    else byDay.set(dayKey, [minuteOfDay])
  }
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
  let gridGap = 0 // gcd(0, x) = x，逐步并入各间隔
  for (const minutes of byDay.values()) {
    const sorted = [...new Set(minutes)].sort((a, b) => a - b)
    for (let i = 1; i < sorted.length; i++) {
      gridGap = gcd(gridGap, sorted[i] - sorted[i - 1])
    }
  }
  if (gridGap <= 0) return null

  // 4) 就近吸附到支持的分钟周期（绝对差最小）。
  let best = MINUTE_INTERVALS[0]
  for (const cand of MINUTE_INTERVALS) {
    if (Math.abs(cand[1] - gridGap) < Math.abs(best[1] - gridGap)) best = cand
  }
  return best[0]
}

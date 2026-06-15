/**
 * 决策 bar 频率（bar_freq）工具：与后端 `live/decision_instant.py` 的映射保持一致。
 *
 * 间隔锁定：bar_freq 由所选模型的训练间隔（input_interval）派生，前端不提供自由选择。
 */

/** 受支持的日内（盘中监控）bar_freq 取值集合，作为 {@link isIntradayBarFreq} 的判定白名单。 */
export const INTRADAY_BAR_FREQS = ['1m', '5m', '10m', '15m', '30m', '60m'] as const

/**
 * 将模型训练间隔（AlphaLab 口径）映射为决策域 bar_freq。
 *
 * `'d'`（日线）→ `'1d'`；其余分钟周期原样返回（如 `'30m'` → `'30m'`）。
 *
 * @param interval - AlphaLab 口径周期字符串；缺省视为 `'d'`。
 * @returns 决策域 bar_freq 字符串。
 *
 * @example
 * ```ts
 * barFreqOfInterval('d')    // '1d'
 * barFreqOfInterval('30m')  // '30m'
 * ```
 */
export const barFreqOfInterval = (interval?: string): string => {
  const v = interval || 'd'
  return v === 'd' ? '1d' : v
}

/**
 * 判断给定 bar_freq 是否为日内（盘中监控）频率。
 *
 * @param barFreq - 决策域 bar_freq 字符串。
 * @returns 属于日内频率（`1m`/`5m`/`10m`/`15m`/`30m`/`60m`）时返回 `true`。
 */
export const isIntradayBarFreq = (barFreq: string): boolean =>
  (INTRADAY_BAR_FREQS as readonly string[]).includes(barFreq)

/**
 * 生成 bar_freq 的展示文案。
 *
 * @param barFreq - 决策域 bar_freq 字符串。
 * @returns 日频：`"日频（1d）"`；日内：`"盘中监控 · 30m"`。
 */
export const barFreqLabel = (barFreq: string): string =>
  isIntradayBarFreq(barFreq) ? `盘中监控 · ${barFreq}` : `日频（${barFreq}）`

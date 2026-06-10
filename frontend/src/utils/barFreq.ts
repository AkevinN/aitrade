/**
 * 决策 bar 频率（bar_freq）工具：与后端 `live/decision_instant.py` 的映射保持一致。
 *
 * 间隔锁定：bar_freq 由所选模型的训练间隔（input_interval）派生，前端不提供自由选择。
 */

export const INTRADAY_BAR_FREQS = ['1m', '5m', '10m', '15m', '30m', '60m'] as const

/** 模型训练间隔（AlphaLab 口径，"d"/"30m"...）→ 决策域 bar_freq（"1d"/"30m"...）。 */
export const barFreqOfInterval = (interval?: string): string => {
  const v = interval || 'd'
  return v === 'd' ? '1d' : v
}

/** 是否日内频率（盘中监控模式）。 */
export const isIntradayBarFreq = (barFreq: string): boolean =>
  (INTRADAY_BAR_FREQS as readonly string[]).includes(barFreq)

/** 展示文案："1d" → "日频（1d）"；分钟频 → "盘中监控 · 30m"。 */
export const barFreqLabel = (barFreq: string): string =>
  isIntradayBarFreq(barFreq) ? `盘中监控 · ${barFreq}` : `日频（${barFreq}）`

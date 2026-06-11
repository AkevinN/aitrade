import dayjs, { type Dayjs } from 'dayjs'

import type {
  ConfidenceLevel,
  ObservationGroup,
  SchemeSuggestion,
  SuggestionItem,
  SymbolProfileRequest,
} from '../types/alpha'

export const CONFIDENCE_ORDER: Record<ConfidenceLevel, number> = {
  insufficient: 0,
  low: 1,
  medium: 2,
  high: 3,
}

export interface ConfidenceStyle {
  color: string
  text: string
  weak: boolean
  description: string
}

export function confidenceStyle(confidence: ConfidenceLevel): ConfidenceStyle {
  const map: Record<ConfidenceLevel, ConfidenceStyle> = {
    high: { color: 'green', text: '高', weak: false, description: '样本充足，指标可作为主要参考。' },
    medium: { color: 'blue', text: '中', weak: false, description: '样本基本够用，建议结合其它指标确认。' },
    low: { color: 'orange', text: '低', weak: true, description: '样本或稳定性偏弱，只适合作为弱参考。' },
    insufficient: { color: 'default', text: '样本不足', weak: true, description: '样本不足，数值被抑制，不应用于强参数建议。' },
  }
  return map[confidence]
}

export interface MetricHelp {
  label: string
  description: string
}

export const METRIC_HELP: Record<string, MetricHelp> = {
  count_valid_bars: {
    label: '有效 K 线数',
    description: '参与画像计算的有效 K 线数量。样本越多，后续统计和建议越可靠。',
  },
  gap_ratio: {
    label: '缺口比例',
    description: '窗口内估计缺失 K 线比例，越低越好。较高时需优先补齐数据。',
  },
  zero_volume_ratio: {
    label: '零成交比例',
    description: '成交量为 0 或缺失的 bar 占比。较高通常意味着流动性或数据质量较弱。',
  },
  alignment_coverage: {
    label: '对齐覆盖率',
    description: '目标标的与观测标的在公共时间轴上的覆盖程度，越高越适合做组间对比。',
  },
  avg_turnover: {
    label: '日均成交额',
    description: '窗口内按自然日汇总后的平均成交额，用于判断交易容量和滑点风险。',
  },
  intraday_concentration: {
    label: '日内集中度',
    description: '分钟级成交是否集中在开盘和收盘附近；非分钟周期不适用。',
  },
  realized_volatility: {
    label: '已实现波动',
    description: '基于收盘对数收益计算的窗口波动，反映近期真实价格波动强度。',
  },
  atr_ratio: {
    label: 'ATR 比例',
    description: 'ATR 相对价格的比例，衡量常规波动幅度和止盈止损空间。',
  },
  amplitude_quantiles: {
    label: '振幅分位数',
    description: '单根 K 线高低价振幅的分布分位数，用于观察典型和极端波动范围。',
  },
  return_autocorr: {
    label: '收益自相关',
    description: '不同滞后阶的收益相关性，用于观察短期延续或反转迹象。',
  },
  hurst_exponent: {
    label: 'Hurst 指数',
    description: '趋势/均值回复倾向指标。接近 0.5 类似随机游走，高于 0.5 偏趋势，低于 0.5 偏均值回复。',
  },
  variance_ratio: {
    label: '方差比',
    description: '大于 1 偏趋势，小于 1 偏均值回复，接近 1 表示更接近随机游走。',
  },
  adf_pvalue: {
    label: 'ADF p 值',
    description: '平稳性检验的 p 值。越低越倾向于均值回复，越高越接近非平稳/趋势结构。',
  },
  skewness: {
    label: '偏度',
    description: '收益分布的不对称程度。正偏表示右尾更长，负偏表示左尾更长。',
  },
  kurtosis: {
    label: '峰度',
    description: '收益分布尾部厚度。值越高，极端收益出现概率越值得关注。',
  },
}

export function metricHelp(metricKey: string): MetricHelp {
  return METRIC_HELP[metricKey] ?? {
    label: metricKey,
    description: '后端返回的画像指标。当前版本尚未配置更详细说明。',
  }
}

function compactMoney(value: number): string {
  const abs = Math.abs(value)
  if (abs >= 100_000_000) {
    return `${(value / 100_000_000).toFixed(2).replace(/\.?0+$/, '')} 亿`
  }
  if (abs >= 10_000) {
    return `${(value / 10_000).toFixed(2).replace(/\.?0+$/, '')} 万`
  }
  return value.toFixed(0)
}

function compactDecimal(value: number): string {
  if (Number.isInteger(value)) {
    return String(value)
  }
  const abs = Math.abs(value)
  if (abs > 0 && abs < 0.0001) {
    return value.toExponential(2)
  }
  return value.toFixed(4).replace(/\.?0+$/, '')
}

function percent(value: number): string {
  return `${(value * 100).toFixed(2).replace(/\.?0+$/, '')}%`
}

export function formatMetricValue(
  value: number | Record<string, number> | null,
  metricKey?: string,
): string {
  if (value === null) {
    return '-'
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return '-'
    }
    if (metricKey === 'avg_turnover') {
      return compactMoney(value)
    }
    if (
      metricKey === 'gap_ratio' ||
      metricKey === 'zero_volume_ratio' ||
      metricKey === 'alignment_coverage' ||
      metricKey === 'intraday_concentration' ||
      metricKey === 'realized_volatility' ||
      metricKey === 'atr_ratio'
    ) {
      return percent(value)
    }
    return compactDecimal(value)
  }
  return Object.entries(value)
    .map(([key, item]) => `${key}: ${metricKey === 'amplitude_quantiles' ? percent(item) : formatMetricValue(item)}`)
    .join(', ')
}

export function isLowConfidenceItem(item: SuggestionItem): boolean {
  return CONFIDENCE_ORDER[item.based_on_confidence] <= CONFIDENCE_ORDER.low
}

export function shouldDropObservation(
  alignmentCoverage: number,
  correlationAbs: number | null | undefined,
): boolean {
  const corr = Math.abs(correlationAbs ?? 0)
  return alignmentCoverage < 0.6 || corr < 0.05
}

export interface BuildProfilingRequestInput {
  targetSymbol: string
  interval: string
  asOf: string | Dayjs
  lookbackDays: number
  observationGroups?: ObservationGroup[]
  withSuggestion?: boolean
  persist?: boolean
}

export function buildProfilingRequest(input: BuildProfilingRequestInput): SymbolProfileRequest {
  const seen = new Set<string>()
  const targetKey = input.targetSymbol.trim()
  const observations: string[] = []
  for (const group of input.observationGroups ?? []) {
    for (const raw of group.symbols) {
      const symbol = raw.trim()
      if (!symbol || symbol === targetKey || seen.has(symbol)) {
        continue
      }
      seen.add(symbol)
      observations.push(symbol)
    }
  }
  const asOf =
    typeof input.asOf === 'string'
      ? input.asOf
      : dayjs(input.asOf).format('YYYY-MM-DDTHH:mm:ss')
  return {
    vt_symbol: targetKey,
    interval: input.interval,
    as_of: asOf,
    lookback_days: Math.max(1, Math.floor(input.lookbackDays)),
    observation_symbols: observations,
    with_suggestion: input.withSuggestion ?? true,
    persist: input.persist ?? true,
  }
}

export interface SuggestionFormMapping {
  values: Record<string, unknown>
  unmapped: SuggestionItem[]
}

const STRONG_FIELDS = new Set([
  'label_spec.mode',
  'label_spec.take_profit',
  'label_spec.stop_loss',
  'label_spec.max_hold',
  'label_spec.horizon',
  'predictor.params.label_type',
])

function percentValue(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }
  return Number((value * 100).toFixed(4))
}

export function mapSuggestionToFormValues(suggestion: SchemeSuggestion): SuggestionFormMapping {
  const values: Record<string, unknown> = {}
  const unmapped: SuggestionItem[] = []

  for (const item of suggestion.items) {
    if (suggestion.degraded && STRONG_FIELDS.has(item.field)) {
      unmapped.push(item)
      continue
    }

    if (item.field === 'label_spec.mode' && typeof item.value === 'string') {
      values.label_mode = item.value
    } else if (item.field === 'label_spec.take_profit') {
      const pct = percentValue(item.value)
      if (pct === undefined) unmapped.push(item)
      else values.oco_take_profit_pct = pct
    } else if (item.field === 'label_spec.stop_loss') {
      const pct = percentValue(item.value)
      if (pct === undefined) unmapped.push(item)
      else values.oco_stop_loss_pct = pct
    } else if (item.field === 'label_spec.max_hold' && typeof item.value === 'number') {
      values.oco_max_hold = item.value
    } else if (item.field === 'label_spec.horizon' && typeof item.value === 'number') {
      values.label_horizon = item.value
    } else if (item.field === 'predictor.params.label_type' && typeof item.value === 'string') {
      if (item.value === 'reg') {
        values.objective = 'regression'
      } else if (item.value === 'cls') {
        values.objective = 'classification'
      } else {
        unmapped.push(item)
      }
    } else {
      unmapped.push(item)
    }
  }

  return { values, unmapped }
}

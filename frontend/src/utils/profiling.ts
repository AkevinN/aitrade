import dayjs, { type Dayjs } from 'dayjs'

import type {
  ConfidenceLevel,
  ObservationGroup,
  SchemeSuggestion,
  SuggestionItem,
  SymbolProfileRequest,
} from '../types/alpha'

/** 置信度级别的数值排序（用于比较，数值越大越可信）。 */
export const CONFIDENCE_ORDER: Record<ConfidenceLevel, number> = {
  insufficient: 0,
  low: 1,
  medium: 2,
  high: 3,
}

/** 置信度的可视化样式属性（颜色、文案、弱化标记、提示说明）。 */
export interface ConfidenceStyle {
  /** Ant Design Tag 颜色名（如 `green` / `blue` / `orange` / `default`）。 */
  color: string
  /** 中文标签文案（如「高」「中」「低」「样本不足」）。 */
  text: string
  /** 是否为弱置信度（展示时可淡化）。 */
  weak: boolean
  /** Tooltip 提示说明。 */
  description: string
}

/**
 * 根据置信度枚举值返回对应的可视化样式属性。
 *
 * @param confidence - 置信度枚举值（`high` / `medium` / `low` / `insufficient`）。
 * @returns 包含颜色、文案、弱化标记和说明的 {@link ConfidenceStyle} 对象。
 */
export function confidenceStyle(confidence: ConfidenceLevel): ConfidenceStyle {
  const map: Record<ConfidenceLevel, ConfidenceStyle> = {
    high: { color: 'green', text: '高', weak: false, description: '样本充足，指标可作为主要参考。' },
    medium: { color: 'blue', text: '中', weak: false, description: '样本基本够用，建议结合其它指标确认。' },
    low: { color: 'orange', text: '低', weak: true, description: '样本或稳定性偏弱，只适合作为弱参考。' },
    insufficient: { color: 'default', text: '样本不足', weak: true, description: '样本不足，数值被抑制，不应用于强参数建议。' },
  }
  return map[confidence]
}

/** 画像指标的中文标签与说明（用于 Tooltip 展示）。 */
export interface MetricHelp {
  /** 指标的中文名称（如「Hurst 指数」）。 */
  label: string
  /** 指标含义说明。 */
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

/**
 * 按指标键名查找对应的中文标签与说明；未知指标返回以键名为标签的兜底对象。
 *
 * @param metricKey - 后端画像返回的指标键名（如 `hurst_exponent`）。
 * @returns 对应的 {@link MetricHelp}；键名未在 {@link METRIC_HELP} 中注册时返回兜底对象。
 */
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

/**
 * 将画像指标值格式化为可读字符串。
 *
 * - `null` → `'-'`
 * - 百分比型指标（如 `gap_ratio`、`realized_volatility` 等）→ `xx.xx%`
 * - `avg_turnover` → 万/亿紧凑金额
 * - 分布型对象（如 `amplitude_quantiles`）→ `key: value, ...`
 * - 其余数值 → 简洁小数（最多 4 位，去除末位零）
 *
 * @param value - 画像指标值；`null` 表示无效。
 * @param metricKey - 指标键名，用于判断格式化规则（可省略）。
 * @returns 格式化后的字符串。
 */
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

/**
 * 判断建议条目的置信度是否为低（`low`）或更低（`insufficient`）。
 *
 * @param item - 建议条目。
 * @returns 置信度 ≤ `low` 时返回 `true`。
 */
export function isLowConfidenceItem(item: SuggestionItem): boolean {
  return CONFIDENCE_ORDER[item.based_on_confidence] <= CONFIDENCE_ORDER.low
}

/**
 * 判断是否应丢弃某个观测标的（对齐覆盖率或相关性过低）。
 *
 * 覆盖率 < 60% 或绝对相关系数 < 0.05 时建议丢弃，避免噪声观测干扰训练。
 *
 * @param alignmentCoverage - 与目标标的的时间轴对齐覆盖率（0~1）。
 * @param correlationAbs - 收益相关系数绝对值；`null`/`undefined` 视为 0。
 * @returns 建议丢弃时返回 `true`。
 */
export function shouldDropObservation(
  alignmentCoverage: number,
  correlationAbs: number | null | undefined,
): boolean {
  const corr = Math.abs(correlationAbs ?? 0)
  return alignmentCoverage < 0.6 || corr < 0.05
}

/** {@link buildProfilingRequest} 的输入参数（前端表单值形态）。 */
export interface BuildProfilingRequestInput {
  /** 目标合约代码（会去除首尾空格）。 */
  targetSymbol: string
  /** K 线周期（如 `'d'`、`'30m'`）。 */
  interval: string
  /** 画像基准时刻（ISO 字符串或 dayjs 实例）。 */
  asOf: string | Dayjs
  /** 回看天数（取整，最小为 1）。 */
  lookbackDays: number
  /** 观测分组列表；省略时 `observation_symbols` 为 `[]`。 */
  observationGroups?: ObservationGroup[]
  /** 是否请求参数建议（默认 `true`）。 */
  withSuggestion?: boolean
  /** 是否将结果持久化为 Artifact（默认 `true`）。 */
  persist?: boolean
}

/**
 * 将前端表单输入值转换为后端 {@link SymbolProfileRequest}。
 *
 * 自动去重观测标的（过滤空值和目标标的本身），`asOf` 统一格式化为 ISO 字符串。
 *
 * @param input - 前端表单值。
 * @returns 后端请求体。
 */
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

/** {@link mapSuggestionToFormValues} 的输出：已映射的表单字段值 + 无法映射的条目列表。 */
export interface SuggestionFormMapping {
  /** 可直接写入 Ant Design Form 的字段值映射（如 `label_mode`、`oco_take_profit_pct`）。 */
  values: Record<string, unknown>
  /** 无法自动映射到表单字段的建议条目（需用户手动确认）。 */
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

/**
 * 将画像参数建议（{@link SchemeSuggestion}）映射为 Ant Design Form 可用的字段值。
 *
 * 支持的字段：`label_spec.mode`、`label_spec.take_profit/stop_loss`（百分比化）、
 * `label_spec.max_hold/horizon`、`predictor.params.label_type`。
 * 不支持的字段或 `suggestion.degraded && STRONG_FIELDS` 命中时放入 `unmapped`。
 *
 * @param suggestion - 画像返回的参数建议对象。
 * @returns 已映射字段值 + 未映射条目的 {@link SuggestionFormMapping}。
 */
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

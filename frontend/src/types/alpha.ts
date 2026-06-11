// Backend response types — must match Pydantic models in backend/aitrade/models/alpha.py

export type TaskStatusValue = 'pending' | 'running' | 'completed' | 'failed'

export interface Task {
  task_id: string
  type: string
  title: string
  entity_type: string
  entity_name: string
  status: TaskStatusValue
  progress: number
  message: string
  result?: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface TaskStartResponse {
  task_id: string
  message?: string
  name?: string
}

// GET /api/alpha/status
export interface AlphaStatus {
  installed: boolean
  lab_path: string
  lab_exists: boolean
  version?: string
}

// GET /api/alpha/datasets/{name}
export interface DatasetDetail {
  name: string
  feature_count: number
  sample_count: number
  label_expression: string
}

// GET /api/alpha/bar-data
export interface BarDataList {
  daily: BarDataItem[]
  minute: BarDataItem[]
}

export interface BarDataItem {
  vt_symbol: string
  row_count: number
  start: string
  end: string
  file_size_kb: number
}

export interface BarDataItemWithInterval extends BarDataItem {
  intervalType: 'daily' | 'minute'
}

// GET /api/alpha/bar-data/{interval}/{vt_symbol}
export interface BarDataDetail {
  vt_symbol: string
  interval: string
  row_count: number
  start: string
  end: string
  columns: string[]
  preview: Record<string, unknown>[]
  loaded_count: number
  has_more: boolean
  next_before: string | null
}

export interface BarDataDetailQuery {
  limit?: number
  before?: string
}

// GET /api/alpha/models/{name}
export interface ModelDetail {
  name: string
  model_type: string
}

// GET /api/alpha/signals/{name}
export interface SignalDetail {
  name: string
  row_count: number
  columns: string[]
  preview: Record<string, unknown>[]
}

// Request types (match backend Pydantic models)

export interface DataDownloadRequest {
  vt_symbols: string[]
  start: string
  end: string
  data_kind?: 'bar' | 'tick'
  source_interval?: string
  interval?: string
  provider?: string
}

export interface DataAggregateRequest {
  vt_symbols: string[]
  start: string
  end: string
  source_kind: 'bar' | 'tick'
  source_interval?: string
  target_interval: string
  session_profile?: string
}

export interface DatasetCreateRequest {
  name: string
  vt_symbols: string[]
  start: string
  end: string
  train_end: string
  valid_end?: string
  features?: string[]
  label_period?: number
}

export interface ModelTrainRequest {
  name: string
  dataset: string
  model_type: string
  params?: Record<string, unknown>
}

export interface SignalGenerateRequest {
  name: string
  model: string
  vt_symbols: string[]
  start: string
  end: string
}

export interface BacktestRunRequest {
  name: string
  signal: string
  capital?: number
  start: string
  end: string
  benchmark?: string
}

export interface BacktestStatistics {
  start_date?: string
  end_date?: string
  total_days?: number
  profit_days?: number
  loss_days?: number
  end_balance?: number
  max_drawdown?: number
  max_ddpercent?: number
  max_drawdown_duration?: number
  total_net_pnl?: number
  daily_net_pnl?: number
  total_commission?: number
  daily_commission?: number
  total_turnover?: number
  daily_turnover?: number
  total_trade_count?: number
  daily_trade_count?: number
  total_return?: number
  annual_return?: number
  daily_return?: number
  return_std?: number
  sharpe_ratio?: number
  return_drawdown_ratio?: number
  capital?: number
  error?: string
  /** 基准（买入持有标的）合约 */
  benchmark_symbol?: string
  /** 基准期末累计收益（%） */
  benchmark_return?: number
  /** 策略相对基准的超额收益（%），正为额外收益、负为额外亏损 */
  excess_return?: number
  /** 本次回测使用的成本假设（CNN 回测回传） */
  commission_rate?: number
  stamp_duty?: number
  slippage?: number
  price_add?: number
  /** 本次回测的出场配置与 label↔策略一致性信息（CNN 回测回传） */
  exit_mode?: string
  hold_days?: number
  t_plus1?: boolean
  label_spec?: Record<string, unknown>
  consistency_warnings?: string[]
}

export interface CNNTrainRequest {
  name: string
  start: string
  end: string
  vt_symbols?: string[]
  target_symbol?: string
  input_data_kind?: 'bar' | 'tick'
  input_interval?: string
  label_spec?: LabelSpec
  observation_groups?: ObservationGroup[]
  epochs?: number
  batch_size?: number
  learning_rate?: number
  lookback?: number
  dropout?: number
  train_ratio?: number
  /** 损失加权：none=普通BCE；magnitude=按|未来收益|加权（仅分类） */
  loss_weighting?: 'none' | 'magnitude'
  /** 预测目标：classification=方向二分类；regression=直接预测涨跌幅 */
  objective?: 'classification' | 'regression'
}

// CSV Import types

export interface CsvPreviewResult {
  data_kind?: 'bar' | 'tick'
  columns: string[]
  sample_rows: Record<string, unknown>[]
  matched_fields: Record<string, string>
  unmapped_columns: string[]
  missing_required: string[]
  total_rows: number
  date_range: [string, string]
  symbols: string[]
}

export interface CsvImportResponse {
  success: boolean
  message: string
  imported_count: number
  skipped_count: number
  errors: string[]
  batches?: DataResourceSummary[]
}

export type CsvInterval = 'd' | '1m' | '5m' | '15m' | '30m' | '60m'
export type CsvImportMode = 'merge' | 'replace'
export type CsvSaveMode = 'official' | 'batch'

export type DataResourceKind = 'raw_bar' | 'raw_tick' | 'derived_bar' | 'raw_bar_batch' | 'raw_tick_batch'

export interface DataResourceSummary {
  key: string
  kind: DataResourceKind
  vt_symbol: string
  interval: string
  row_count: number
  start: string
  end: string
  file_size_kb: number
  source_kind: string
  source_interval: string
  target_interval: string
  created_at?: string | null
  session_profile?: string
  status?: 'pending' | 'merged' | string | null
  batch_id?: string | null
  file_name?: string | null
  batch_resource_kind?: 'raw_bar' | 'raw_tick'
}

export interface DataResourceList {
  raw_bars: DataResourceSummary[]
  raw_ticks: DataResourceSummary[]
  raw_bar_batches?: DataResourceSummary[]
  raw_tick_batches?: DataResourceSummary[]
  derived_bars: DataResourceSummary[]
  raw_bar_intervals: string[]
  derived_intervals: string[]
}

export interface DataResourceDetail extends DataResourceSummary {
  columns: string[]
  preview: Record<string, unknown>[]
  loaded_count: number
  has_more: boolean
  next_before: string | null
}

export interface RelocateBarIntervalResponse {
  success: boolean
  message: string
  key: string
  interval: string
  vt_symbol: string
}

export interface DataResourceMergeRequest {
  kind: 'raw_bar' | 'raw_tick'
  keys: string[]
}

export interface DataResourceMergePreview {
  can_merge: boolean
  reason: string
  errors: string[]
  kind: 'raw_bar' | 'raw_tick'
  keys: string[]
  vt_symbol?: string
  interval?: string
  intersection_start?: string
  intersection_end?: string
  conflict_count?: number
  estimated_rows?: number
  batch_count?: number
  has_official?: boolean
  adjust_type?: string
}

export interface DataResourceMergeResponse extends DataResourceMergePreview {
  success: boolean
  message: string
  row_count?: number
  start?: string
  end?: string
}

export interface ObservationGroup {
  role: 'target' | 'market' | 'sector' | 'leaders' | 'custom'
  name: string
  symbols: string[]
}

export interface LabelSpec {
  mode: 'next_bar' | 'horizon_bars' | 'session_close' | 'next_session_close' | 'oco'
  horizon?: number
  /** 最小有效波动阈值（收益率，0.005=0.5%）；0 关闭去噪 */
  threshold?: number
  /** 中性样本处理：drop=丢弃，negative=并入下跌类 */
  neutral_policy?: 'drop' | 'negative'
  /** 标签计价口径：close=收盘到收盘；next_open=次开盘；next_close=次收盘；next_vwap=次日均价 */
  price_ref?: 'close' | 'next_open' | 'next_close' | 'next_vwap'
  /** oco 模式止盈幅度（收益率，0.03=+3%）；mode=oco 时必填且 > 0 */
  take_profit?: number
  /** oco 模式止损幅度（收益率，0.02=-2%）；mode=oco 时必填且 > 0 */
  stop_loss?: number
  /** oco 模式最大持有 bar 数（到期未触发按时间止损）；缺省回退到 horizon 或 10 */
  max_hold?: number
  /** oco 模式同根 bar 双触发时是否保守假设止损先到（默认 true，与回测一致） */
  stop_first?: boolean
}

// 回测明细数据类型 — 用于图表可视化（成交明细、资金曲线）

/** 单笔成交记录 */
export interface BacktestTrade {
  datetime: string
  vt_symbol: string
  direction: string
  offset: string
  price: number
  volume: number
}

/** 资金曲线单日记录 */
export interface BacktestEquityRow {
  date: string
  balance: number
  drawdown: number
  ddpercent: number
  net_pnl: number
  /** 策略累计收益（%） */
  strategy_return?: number | null
  /** 基准（买入持有标的）累计收益（%） */
  benchmark_return?: number | null
  /** 超额收益（%）= 策略累计收益 - 基准累计收益 */
  excess_return?: number | null
}

/** 回测结果完整载荷（含统计指标、成交明细、资金曲线） */
export interface BacktestResultPayload {
  name?: string
  model?: string
  target_symbol?: string
  statistics?: BacktestStatistics
  trades?: BacktestTrade[]
  equity_curve?: BacktestEquityRow[]
}

// Symbol Profiling types — match backend/aitrade/profiling/types.py

export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'insufficient'

export interface SymbolProfileRequest {
  vt_symbol: string
  interval: string
  as_of: string
  lookback_days: number
  observation_symbols?: string[]
  with_suggestion?: boolean
  persist?: boolean
}

export interface MetricValue {
  key: string
  value: number | Record<string, number> | null
  effective_sample: number
  confidence: ConfidenceLevel
  note?: string | null
}

export interface MetricBlock {
  block: 'data_quality' | 'liquidity' | 'volatility' | 'predictability'
  metrics: MetricValue[]
  level?: string | null
}

export interface SuggestionItem {
  field: string
  value: unknown
  reason: string
  based_on_confidence: ConfidenceLevel
}

export interface SchemeSuggestion {
  status: 'draft'
  interval: string
  vt_symbols: string[]
  items: SuggestionItem[]
  degraded: boolean
  note?: string | null
}

export interface GroupProfile {
  target: string
  members: string[]
  alignment_coverage: number
  correlation_summary: Record<string, number>
}

export interface ProfileInput {
  vt_symbol: string
  interval: string
  as_of: string
  lookback_days: number
  effective_right_bound?: string | null
  effective_bar_count: number
  rules_id: string
}

export interface SymbolProfileResponse {
  input: ProfileInput
  available: boolean
  unavailable_reason?: string | null
  blocks: MetricBlock[]
  group_profile?: GroupProfile | null
  suggestion?: SchemeSuggestion | null
  overall_confidence: ConfidenceLevel
  created_at: string
  artifact_id?: string | null
}

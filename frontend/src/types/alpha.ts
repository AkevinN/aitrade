// Backend response types — must match Pydantic models in backend/aitrade/models/alpha.py

/** 异步任务的四种状态枚举。 */
export type TaskStatusValue = 'pending' | 'running' | 'completed' | 'failed'

/**
 * 异步任务完整记录（`GET /api/alpha/tasks/{task_id}` 响应体）。
 *
 * 轮询此对象直到 status 达到终态（completed / failed）。
 * R1 新增字段（started_at / finished_at / duration_ms / error_traceback / params）
 * 均为可选，旧任务无此字段时值为 undefined。
 */
export interface Task {
  /** 任务唯一 ID */
  task_id: string
  /** 任务类型标识，如 "backtest" / "train" */
  type: string
  /** 任务标题（可读名称，展示于 UI） */
  title: string
  /** 关联实体类型，如 "model" / "signal" */
  entity_type: string
  /** 关联实体名称 */
  entity_name: string
  /** 当前状态 */
  status: TaskStatusValue
  /** 进度 0–100（整数百分比） */
  progress: number
  /** 当前进度消息或失败原因 */
  message: string
  /** 终态结果载荷；运行中为 null */
  result?: Record<string, unknown> | null
  /** 创建时刻 ISO */
  created_at: string
  /** 最近更新时刻 ISO */
  updated_at: string
  // ---- task-scheduler-observability R1 新增字段（全带默认值，旧任务无此字段为 undefined）----
  /** 任务开始执行时刻 ISO；未开始为 null。 */
  started_at?: string | null
  /** 任务到达终态时刻 ISO；未终态为 null。 */
  finished_at?: string | null
  /** 执行耗时（毫秒）；未完成或未计算为 null。 */
  duration_ms?: number | null
  /** 失败时的错误堆栈（截断 8000 字符）；无错误为空字符串。 */
  error_traceback?: string
  /** 创建参数摘要（疑似凭证键值已替换为 "***"）。 */
  params?: Record<string, unknown>
}

/** 异步任务启动响应（`POST` 启动类接口的通用返回体）。 */
export interface TaskStartResponse {
  /** 新建任务 ID，用于轮询或 WS 订阅。 */
  task_id: string
  /** 可选的可读提示信息。 */
  message?: string
  /** 新建资源名称（如模型名、信号名），部分接口返回。 */
  name?: string
}

/** Alpha 模块状态（`GET /api/alpha/status` 响应体）。 */
export interface AlphaStatus {
  /** Alpha 模块是否已安装。 */
  installed: boolean
  /** Lab 数据目录绝对路径。 */
  lab_path: string
  /** Lab 数据目录是否实际存在于磁盘。 */
  lab_exists: boolean
  /** 模块版本号，旧版本后端可能不返回。 */
  version?: string
}

/** Alpha 数据集详情（`GET /api/alpha/datasets/{name}` 响应体）。 */
export interface DatasetDetail {
  /** 数据集名称。 */
  name: string
  /** 特征列数量。 */
  feature_count: number
  /** 样本行数量。 */
  sample_count: number
  /** 标签定义表达式。 */
  label_expression: string
}

/** K 线数据列表（`GET /api/alpha/bar-data` 响应体）。 */
export interface BarDataList {
  /** 日线数据条目列表。 */
  daily: BarDataItem[]
  /** 分钟线数据条目列表。 */
  minute: BarDataItem[]
}

/** 单个 K 线文件的元数据摘要（合约级别）。 */
export interface BarDataItem {
  /** 合约代码（如 `000001.SZSE`）。 */
  vt_symbol: string
  /** 数据行数。 */
  row_count: number
  /** 数据起始日期（ISO 格式）。 */
  start: string
  /** 数据结束日期（ISO 格式）。 */
  end: string
  /** 文件大小（KB）。 */
  file_size_kb: number
}

/** 带周期类型标记的 K 线条目（前端展示层扩展）。 */
export interface BarDataItemWithInterval extends BarDataItem {
  /** 周期分类：`daily` 日线；`minute` 分钟线。 */
  intervalType: 'daily' | 'minute'
}

/** K 线详情（`GET /api/alpha/bar-data/{interval}/{vt_symbol}` 响应体），含分页预览。 */
export interface BarDataDetail {
  /** 合约代码。 */
  vt_symbol: string
  /** K 线周期（如 `d`、`1m`）。 */
  interval: string
  /** 总行数（全量）。 */
  row_count: number
  /** 数据起始时间。 */
  start: string
  /** 数据结束时间。 */
  end: string
  /** 列名列表。 */
  columns: string[]
  /** 当前分页的预览行数据。 */
  preview: Record<string, unknown>[]
  /** 本次返回的行数。 */
  loaded_count: number
  /** 是否还有更早的数据（cursor 分页）。 */
  has_more: boolean
  /** 下一页游标（datetime 字符串）；无更多数据时为 `null`。 */
  next_before: string | null
}

/** K 线详情分页查询参数。 */
export interface BarDataDetailQuery {
  /** 单次返回行数上限。 */
  limit?: number
  /** 向前翻页游标（datetime 字符串）。 */
  before?: string
}

/** Alpha 模型详情（`GET /api/alpha/models/{name}` 响应体）。 */
export interface ModelDetail {
  /** 模型名称。 */
  name: string
  /** 模型类型（如 `lgb` / `mlp` / `lasso`）。 */
  model_type: string
}

/** Alpha 信号详情（`GET /api/alpha/signals/{name}` 响应体），含预览数据。 */
export interface SignalDetail {
  /** 信号名称。 */
  name: string
  /** 信号行数（总量）。 */
  row_count: number
  /** 列名列表。 */
  columns: string[]
  /** 预览行数据（前 N 行）。 */
  preview: Record<string, unknown>[]
}

// Request types (match backend Pydantic models)

/** 行情/Tick 数据下载请求（`POST /api/alpha/data/download`）。 */
export interface DataDownloadRequest {
  vt_symbols: string[]
  start: string
  end: string
  data_kind?: 'bar' | 'tick'
  source_interval?: string
  interval?: string
  provider?: string
  /** 品种类型：stock=A股股票；etf=交易所交易基金；cbond=可转债（与后端 Pydantic 对齐） */
  asset_class?: 'stock' | 'etf' | 'cbond'
}

/** K 线聚合请求（`POST /api/alpha/data/aggregate`）。 */
export interface DataAggregateRequest {
  /** 待聚合的合约代码列表。 */
  vt_symbols: string[]
  /** 聚合起始日期（ISO 格式）。 */
  start: string
  /** 聚合结束日期（ISO 格式）。 */
  end: string
  /** 来源数据类型：`bar` 原始K线；`tick` 历史Tick。 */
  source_kind: 'bar' | 'tick'
  /** 来源K线周期（`source_kind='bar'` 时必填）。 */
  source_interval?: string
  /** 目标聚合周期（如 `5m`、`30m`）。 */
  target_interval: string
  /** 时段规则（默认 `cn_equity`）。 */
  session_profile?: string
}

/** Alpha 数据集创建请求（`POST /api/alpha/datasets/create`）。 */
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

/** Alpha 模型训练请求（`POST /api/alpha/models/train`）。 */
export interface ModelTrainRequest {
  name: string
  dataset: string
  model_type: string
  params?: Record<string, unknown>
}

/** 信号生成请求（`POST /api/alpha/signals/generate`）。 */
export interface SignalGenerateRequest {
  name: string
  model: string
  vt_symbols: string[]
  start: string
  end: string
}

/** Alpha 回测运行请求（`POST /api/alpha/backtest/run`）。 */
export interface BacktestRunRequest {
  name: string
  signal: string
  capital?: number
  start: string
  end: string
  benchmark?: string
}

/** 回测统计指标汇总（`task.result.statistics`）。 */
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

/** CSV 导入预览结果（`POST /api/alpha/bar-data/import/preview` 响应体）。 */
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

/** CSV 导入结果响应（`POST /api/alpha/bar-data/import` 响应体）。 */
export interface CsvImportResponse {
  success: boolean
  message: string
  imported_count: number
  skipped_count: number
  errors: string[]
  batches?: DataResourceSummary[]
}

/** CSV 导入支持的 K 线周期（与后端 `CsvInterval` 对齐）。 */
export type CsvInterval = 'd' | '1m' | '5m' | '15m' | '30m' | '60m'
/** CSV 导入模式：`merge` 合并保留旧数据；`replace` 全量覆盖。 */
export type CsvImportMode = 'merge' | 'replace'
/** CSV 存储模式：`official` 直接写入官方库；`batch` 暂存为待合并批次。 */
export type CsvSaveMode = 'official' | 'batch'

/** 数据资源类型标识（与后端 `DataResourceKind` 枚举对齐）。 */
export type DataResourceKind = 'raw_bar' | 'raw_tick' | 'derived_bar' | 'raw_bar_batch' | 'raw_tick_batch'

/** 数据资源摘要（资源列表元素）。 */
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

/** 本地数据资源总览（`GET /api/alpha/data/resources` 响应体）。 */
export interface DataResourceList {
  raw_bars: DataResourceSummary[]
  raw_ticks: DataResourceSummary[]
  raw_bar_batches?: DataResourceSummary[]
  raw_tick_batches?: DataResourceSummary[]
  derived_bars: DataResourceSummary[]
  raw_bar_intervals: string[]
  derived_intervals: string[]
}

/** 数据资源详情（`GET /api/alpha/data/resources/{kind}/{key}` 响应体），含分页预览。 */
export interface DataResourceDetail extends DataResourceSummary {
  columns: string[]
  preview: Record<string, unknown>[]
  loaded_count: number
  has_more: boolean
  next_before: string | null
}

/** K 线周期重定义响应（`PATCH /api/alpha/data/resources/raw_bar/{key}/interval`）。 */
export interface RelocateBarIntervalResponse {
  success: boolean
  message: string
  key: string
  interval: string
  vt_symbol: string
}

/** 数据资源批次合并请求（`POST /api/alpha/data/resources/merge`）。 */
export interface DataResourceMergeRequest {
  kind: 'raw_bar' | 'raw_tick'
  keys: string[]
}

/** 数据资源合并预览结果（`POST /api/alpha/data/resources/merge/preview`）。 */
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

/** 数据资源合并执行结果（`POST /api/alpha/data/resources/merge` 执行响应）。 */
export interface DataResourceMergeResponse extends DataResourceMergePreview {
  success: boolean
  message: string
  row_count?: number
  start?: string
  end?: string
}

/**
 * CNN 训练的观测分组（`target` 目标股、`market` 大盘、`sector` 板块等）。
 *
 * 每组映射到模型的一个输入通道，提供多标的语义关联信息。
 */
export interface ObservationGroup {
  role: 'target' | 'market' | 'sector' | 'leaders' | 'custom'
  name: string
  symbols: string[]
}

/**
 * CNN 训练标签规格：定义如何从历史行情计算监督学习标签。
 *
 * 支持五种模式：`next_bar`（下一根）、`horizon_bars`（N根后）、
 * `session_close`（当日收盘）、`next_session_close`（次日收盘）、
 * `oco`（止盈止损路径依赖）。
 */
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

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
  /** 待下载的合约代码列表（如 `["000001.SZSE"]`）。 */
  vt_symbols: string[]
  /** 下载起始日期（ISO 格式，含）。 */
  start: string
  /** 下载结束日期（ISO 格式，含）。 */
  end: string
  /** 数据类型：`bar` K线（默认）；`tick` 逐笔。 */
  data_kind?: 'bar' | 'tick'
  /** 来源数据周期（`data_kind='bar'` 时指定原始周期）。 */
  source_interval?: string
  /** 目标存储周期（如 `d`、`1m`）。 */
  interval?: string
  /** 数据源标识；缺省走后端默认数据源。 */
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
  /** 新建数据集名称（唯一标识）。 */
  name: string
  /** 纳入数据集的合约代码列表。 */
  vt_symbols: string[]
  /** 样本区间起始日期（ISO 格式，含）。 */
  start: string
  /** 样本区间结束日期（ISO 格式，含）。 */
  end: string
  /** 训练集结束日期（含）；此后至 valid_end 为验证集。 */
  train_end: string
  /** 验证集结束日期（含）；缺省则无独立验证集。 */
  valid_end?: string
  /** 选用的特征列；缺省使用全部可用特征。 */
  features?: string[]
  /** 标签前瞻周期（未来 N 根 K 线收益）；缺省走后端默认值。 */
  label_period?: number
}

/** Alpha 模型训练请求（`POST /api/alpha/models/train`）。 */
export interface ModelTrainRequest {
  /** 新建模型名称（唯一标识）。 */
  name: string
  /** 训练所用数据集名称。 */
  dataset: string
  /** 模型类型（如 `lgb` / `mlp` / `lasso`）。 */
  model_type: string
  /** 模型超参数键值对；缺省使用该模型类型的默认超参。 */
  params?: Record<string, unknown>
}

/** 信号生成请求（`POST /api/alpha/signals/generate`）。 */
export interface SignalGenerateRequest {
  /** 新建信号名称（唯一标识）。 */
  name: string
  /** 用于生成信号的已训练模型名称。 */
  model: string
  /** 生成信号覆盖的合约代码列表。 */
  vt_symbols: string[]
  /** 信号区间起始日期（ISO 格式，含）。 */
  start: string
  /** 信号区间结束日期（ISO 格式，含）。 */
  end: string
}

/** Alpha 回测运行请求（`POST /api/alpha/backtest/run`）。 */
export interface BacktestRunRequest {
  /** 新建回测名称（唯一标识）。 */
  name: string
  /** 回测所用信号名称。 */
  signal: string
  /** 初始资金（元）；缺省走后端默认本金。 */
  capital?: number
  /** 回测起始日期（ISO 格式，含）。 */
  start: string
  /** 回测结束日期（ISO 格式，含）。 */
  end: string
  /** 基准合约代码（买入持有对照）；缺省则不计算超额收益。 */
  benchmark?: string
}

/**
 * 回测统计指标汇总（`task.result.statistics`）。
 *
 * 与 vnpy 回测引擎口径一致；回测失败时仅返回 `error` 字段。
 */
export interface BacktestStatistics {
  /** 回测起始日期（ISO 格式）。 */
  start_date?: string
  /** 回测结束日期（ISO 格式）。 */
  end_date?: string
  /** 交易日总数。 */
  total_days?: number
  /** 盈利交易日数。 */
  profit_days?: number
  /** 亏损交易日数。 */
  loss_days?: number
  /** 期末账户权益（元）。 */
  end_balance?: number
  /** 最大回撤（绝对金额，元）。 */
  max_drawdown?: number
  /** 最大回撤百分比（%，负值）。 */
  max_ddpercent?: number
  /** 最大回撤持续天数。 */
  max_drawdown_duration?: number
  /** 累计净盈亏（元，已扣费）。 */
  total_net_pnl?: number
  /** 日均净盈亏（元）。 */
  daily_net_pnl?: number
  /** 累计手续费（元）。 */
  total_commission?: number
  /** 日均手续费（元）。 */
  daily_commission?: number
  /** 累计成交额（元）。 */
  total_turnover?: number
  /** 日均成交额（元）。 */
  daily_turnover?: number
  /** 累计成交笔数。 */
  total_trade_count?: number
  /** 日均成交笔数。 */
  daily_trade_count?: number
  /** 累计收益率（%）。 */
  total_return?: number
  /** 年化收益率（%）。 */
  annual_return?: number
  /** 日均收益率（%）。 */
  daily_return?: number
  /** 日收益率标准差（%）。 */
  return_std?: number
  /** 夏普比率（年化）。 */
  sharpe_ratio?: number
  /** 收益回撤比（年化收益 / 最大回撤百分比）。 */
  return_drawdown_ratio?: number
  /** 初始本金（元）。 */
  capital?: number
  /** 回测失败时的错误信息；成功时为 undefined。 */
  error?: string
  /** 基准（买入持有标的）合约 */
  benchmark_symbol?: string
  /** 基准期末累计收益（%） */
  benchmark_return?: number
  /** 策略相对基准的超额收益（%），正为额外收益、负为额外亏损 */
  excess_return?: number
  /** 本次回测使用的成本假设——佣金费率（小数，0.0003=万三）（CNN 回测回传） */
  commission_rate?: number
  /** 印花税率（小数，仅卖出计提）。 */
  stamp_duty?: number
  /** 滑点（每股价格偏移，元）。 */
  slippage?: number
  /** 下单超价档位（元），追价成交用。 */
  price_add?: number
  /** 本次回测的出场配置与 label↔策略一致性信息——出场模式（如 `hold` / `oco`）（CNN 回测回传） */
  exit_mode?: string
  /** 固定持有天数（`exit_mode` 为定时出场时生效）。 */
  hold_days?: number
  /** 是否遵循 T+1 交易规则（当日买入次日才可卖）。 */
  t_plus1?: boolean
  /** 训练时的标签规格快照，用于核对 label 与回测策略是否一致。 */
  label_spec?: Record<string, unknown>
  /** label 与回测策略不一致时的告警文案列表；一致时为空。 */
  consistency_warnings?: string[]
  /**
   * 否决买入次数（path_class 模型回测回传）：本次回测中因 prob_sl ≥ veto_threshold
   * 而被放弃的买入信号数量；非 path_class 模型或关闭否决时为 undefined。
   */
  veto_count?: number
}

/** CNN 模型训练请求（`POST /api/alpha/cnn/train`）。 */
export interface CNNTrainRequest {
  /** 新建 CNN 模型名称（唯一标识）。 */
  name: string
  /** 训练样本起始日期（ISO 格式，含）。 */
  start: string
  /** 训练样本结束日期（ISO 格式，含）。 */
  end: string
  /** 参与训练的合约代码列表；与 observation_groups 二选一组织输入。 */
  vt_symbols?: string[]
  /** 预测目标合约（标签基于该标的计算）。 */
  target_symbol?: string
  /** 输入数据类型：`bar` K线（默认）；`tick` 逐笔。 */
  input_data_kind?: 'bar' | 'tick'
  /** 输入 K 线周期（如 `d`、`1m`）。 */
  input_interval?: string
  /** 标签规格；定义如何从行情计算监督标签。 */
  label_spec?: LabelSpec
  /** 观测分组（多通道输入）；提供大盘/板块等语义关联。 */
  observation_groups?: ObservationGroup[]
  /** 训练轮数；缺省走后端默认值。 */
  epochs?: number
  /** 批大小；缺省走后端默认值。 */
  batch_size?: number
  /** 学习率；缺省走后端默认值。 */
  learning_rate?: number
  /** 回看窗口长度（输入序列的 bar 数）；缺省走后端默认值。 */
  lookback?: number
  /** Dropout 比例（0–1）；缺省走后端默认值。 */
  dropout?: number
  /** 训练集占比（0–1），其余作验证集；缺省走后端默认值。 */
  train_ratio?: number
  /** 损失加权：none=普通BCE；magnitude=按|未来收益|加权（仅分类） */
  loss_weighting?: 'none' | 'magnitude'
  /**
   * 预测目标：
   * - `classification`：方向二分类，输出上涨概率；
   * - `regression`：直接预测涨跌幅；
   * - `path_class`：路径形态四分类（先触止盈/先触止损/到期小涨/到期小跌），需搭配 OCO 标签。
   */
  objective?: 'classification' | 'regression' | 'path_class'
}

// CSV Import types

/** CSV 导入预览结果（`POST /api/alpha/bar-data/import/preview` 响应体）。 */
export interface CsvPreviewResult {
  /** 识别出的数据类型：`bar` K线；`tick` 逐笔。 */
  data_kind?: 'bar' | 'tick'
  /** CSV 原始列名列表。 */
  columns: string[]
  /** 抽样行数据（用于展示预览）。 */
  sample_rows: Record<string, unknown>[]
  /** 自动匹配的字段映射：标准字段名 → CSV 列名。 */
  matched_fields: Record<string, string>
  /** 未能映射到标准字段的 CSV 列名列表。 */
  unmapped_columns: string[]
  /** 缺失的必填标准字段名列表；非空则无法导入。 */
  missing_required: string[]
  /** CSV 总行数（不含表头）。 */
  total_rows: number
  /** 数据日期范围 `[起始, 结束]`（ISO 格式）。 */
  date_range: [string, string]
  /** 识别出的合约代码列表。 */
  symbols: string[]
}

/** CSV 导入结果响应（`POST /api/alpha/bar-data/import` 响应体）。 */
export interface CsvImportResponse {
  /** 导入是否整体成功。 */
  success: boolean
  /** 结果提示信息（成功摘要或失败原因）。 */
  message: string
  /** 成功导入的行数。 */
  imported_count: number
  /** 跳过的行数（重复或非法）。 */
  skipped_count: number
  /** 逐行错误信息列表；无错误时为空。 */
  errors: string[]
  /** batch 存储模式下新建的批次资源摘要；official 模式为 undefined。 */
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
  /** 资源唯一键（后端寻址用）。 */
  key: string
  /** 资源种类（原始/派生/批次）。 */
  kind: DataResourceKind
  /** 合约代码。 */
  vt_symbol: string
  /** K 线周期（如 `d`、`1m`）；tick 资源为空。 */
  interval: string
  /** 数据行数。 */
  row_count: number
  /** 数据起始时间（ISO 格式）。 */
  start: string
  /** 数据结束时间（ISO 格式）。 */
  end: string
  /** 文件大小（KB）。 */
  file_size_kb: number
  /** 派生数据的来源类型（如 `bar` / `tick`）；原始资源为空。 */
  source_kind: string
  /** 派生数据的来源周期；原始资源为空。 */
  source_interval: string
  /** 派生数据的目标周期；原始资源为空。 */
  target_interval: string
  /** 资源创建时刻（ISO 格式）；未知为 null。 */
  created_at?: string | null
  /** 时段规则标识（如 `cn_equity`）。 */
  session_profile?: string
  /** 批次状态：`pending` 待合并；`merged` 已合并；其它字符串为后端扩展。 */
  status?: 'pending' | 'merged' | string | null
  /** 所属批次 ID；非批次资源为 null。 */
  batch_id?: string | null
  /** 原始上传文件名；无则为 null。 */
  file_name?: string | null
  /** 批次资源的底层数据类型；仅批次资源有意义。 */
  batch_resource_kind?: 'raw_bar' | 'raw_tick'
}

/** 本地数据资源总览（`GET /api/alpha/data/resources` 响应体）。 */
export interface DataResourceList {
  /** 原始 K 线资源列表。 */
  raw_bars: DataResourceSummary[]
  /** 原始 Tick 资源列表。 */
  raw_ticks: DataResourceSummary[]
  /** 待合并的原始 K 线批次列表。 */
  raw_bar_batches?: DataResourceSummary[]
  /** 待合并的原始 Tick 批次列表。 */
  raw_tick_batches?: DataResourceSummary[]
  /** 派生（聚合）K 线资源列表。 */
  derived_bars: DataResourceSummary[]
  /** 现有原始 K 线的全部周期取值。 */
  raw_bar_intervals: string[]
  /** 现有派生 K 线的全部周期取值。 */
  derived_intervals: string[]
}

/** 数据资源详情（`GET /api/alpha/data/resources/{kind}/{key}` 响应体），含分页预览。 */
export interface DataResourceDetail extends DataResourceSummary {
  /** 列名列表。 */
  columns: string[]
  /** 当前分页的预览行数据。 */
  preview: Record<string, unknown>[]
  /** 本次返回的行数。 */
  loaded_count: number
  /** 是否还有更早的数据（cursor 分页）。 */
  has_more: boolean
  /** 下一页游标（datetime 字符串）；无更多数据时为 null。 */
  next_before: string | null
}

/** K 线周期重定义响应（`PATCH /api/alpha/data/resources/raw_bar/{key}/interval`）。 */
export interface RelocateBarIntervalResponse {
  /** 重定义是否成功。 */
  success: boolean
  /** 结果提示信息。 */
  message: string
  /** 被操作资源的新键。 */
  key: string
  /** 重定义后的 K 线周期。 */
  interval: string
  /** 合约代码。 */
  vt_symbol: string
}

/** 数据资源批次合并请求（`POST /api/alpha/data/resources/merge`）。 */
export interface DataResourceMergeRequest {
  /** 待合并资源的底层数据类型。 */
  kind: 'raw_bar' | 'raw_tick'
  /** 参与合并的资源键列表。 */
  keys: string[]
}

/** 数据资源合并预览结果（`POST /api/alpha/data/resources/merge/preview`）。 */
export interface DataResourceMergePreview {
  /** 是否允许合并；false 时见 reason / errors。 */
  can_merge: boolean
  /** 不可合并的主要原因；可合并时为空。 */
  reason: string
  /** 详细的阻断/校验错误列表。 */
  errors: string[]
  /** 合并资源的底层数据类型。 */
  kind: 'raw_bar' | 'raw_tick'
  /** 参与合并的资源键列表。 */
  keys: string[]
  /** 合并后合约代码（各资源一致时给出）。 */
  vt_symbol?: string
  /** 合并后 K 线周期。 */
  interval?: string
  /** 各资源时间区间的交集起点（ISO 格式）。 */
  intersection_start?: string
  /** 各资源时间区间的交集终点（ISO 格式）。 */
  intersection_end?: string
  /** 重叠区间内数据值冲突的条数。 */
  conflict_count?: number
  /** 合并后预计的总行数。 */
  estimated_rows?: number
  /** 参与合并的批次数量。 */
  batch_count?: number
  /** 是否已存在官方库数据（影响覆盖策略）。 */
  has_official?: boolean
  /** 复权类型（如 `none` / `qfq` / `hfq`）。 */
  adjust_type?: string
}

/** 数据资源合并执行结果（`POST /api/alpha/data/resources/merge` 执行响应）。 */
export interface DataResourceMergeResponse extends DataResourceMergePreview {
  /** 合并是否执行成功。 */
  success: boolean
  /** 结果提示信息。 */
  message: string
  /** 合并落库后的实际行数。 */
  row_count?: number
  /** 合并后数据的起始时间（ISO 格式）。 */
  start?: string
  /** 合并后数据的结束时间（ISO 格式）。 */
  end?: string
}

/**
 * CNN 训练的观测分组（`target` 目标股、`market` 大盘、`sector` 板块等）。
 *
 * 每组映射到模型的一个输入通道，提供多标的语义关联信息。
 */
export interface ObservationGroup {
  /** 分组角色：target=目标股；market=大盘；sector=板块；leaders=龙头；custom=自定义。 */
  role: 'target' | 'market' | 'sector' | 'leaders' | 'custom'
  /** 分组可读名称（展示用）。 */
  name: string
  /** 该组包含的合约代码列表。 */
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
  /** 标签计算模式：next_bar=下一根；horizon_bars=N根后；session_close=当日收盘；next_session_close=次日收盘；oco=止盈止损路径依赖。 */
  mode: 'next_bar' | 'horizon_bars' | 'session_close' | 'next_session_close' | 'oco'
  /** 前瞻 bar 数（`mode='horizon_bars'` 时生效）。 */
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
  /** 成交时刻（ISO 格式）。 */
  datetime: string
  /** 合约代码。 */
  vt_symbol: string
  /** 买卖方向（如 `LONG` / `SHORT`）。 */
  direction: string
  /** 开平标志（如 `OPEN` / `CLOSE`）。 */
  offset: string
  /** 成交价（元）。 */
  price: number
  /** 成交量（股/手）。 */
  volume: number
}

/** 资金曲线单日记录 */
export interface BacktestEquityRow {
  /** 日期（ISO 格式）。 */
  date: string
  /** 当日收盘账户权益（元）。 */
  balance: number
  /** 当日回撤金额（元，距前高的回落）。 */
  drawdown: number
  /** 当日回撤百分比（%，负值）。 */
  ddpercent: number
  /** 当日净盈亏（元，已扣费）。 */
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
  /** 回测名称。 */
  name?: string
  /** 所用模型名称（CNN/Alpha 回测回传）。 */
  model?: string
  /** 预测/交易的目标合约。 */
  target_symbol?: string
  /** 统计指标汇总。 */
  statistics?: BacktestStatistics
  /** 成交明细列表。 */
  trades?: BacktestTrade[]
  /** 资金曲线逐日记录。 */
  equity_curve?: BacktestEquityRow[]
}

// Symbol Profiling types — match backend/aitrade/profiling/types.py

/** 指标置信度等级：high/medium/low 按有效样本量递减，insufficient 表示样本不足无法判定。 */
export type ConfidenceLevel = 'high' | 'medium' | 'low' | 'insufficient'

/** 标的画像请求（`POST /api/alpha/profiling/profile`）。 */
export interface SymbolProfileRequest {
  /** 目标合约代码。 */
  vt_symbol: string
  /** K 线周期（如 `d`、`1m`）。 */
  interval: string
  /** 画像基准日期（ISO 格式），以此为右边界回看。 */
  as_of: string
  /** 回看天数（统计窗口长度）。 */
  lookback_days: number
  /** 协同观测的关联合约列表；提供后才计算 group_profile。 */
  observation_symbols?: string[]
  /** 是否同时生成配置方案建议，默认 false。 */
  with_suggestion?: boolean
  /** 是否将本次画像结果持久化为 artifact，默认 false。 */
  persist?: boolean
}

/** 单项画像指标的取值与可信度。 */
export interface MetricValue {
  /** 指标键名（如 `amihud`、`realized_vol`）。 */
  key: string
  /** 指标值；标量、子指标映射或不可计算时为 null。 */
  value: number | Record<string, number> | null
  /** 参与计算的有效样本量。 */
  effective_sample: number
  /** 该指标的置信度等级。 */
  confidence: ConfidenceLevel
  /** 补充说明（如降级原因）；无则为 null。 */
  note?: string | null
}

/** 一类画像指标的集合（按维度分块）。 */
export interface MetricBlock {
  /** 维度：data_quality 数据质量；liquidity 流动性；volatility 波动性；predictability 可预测性。 */
  block: 'data_quality' | 'liquidity' | 'volatility' | 'predictability'
  /** 该维度下的指标列表。 */
  metrics: MetricValue[]
  /** 该维度的整体评级文案；无则为 null。 */
  level?: string | null
}

/** 配置方案建议中的单条字段建议。 */
export interface SuggestionItem {
  /** 建议作用的配置字段名。 */
  field: string
  /** 建议取值（类型随字段而定）。 */
  value: unknown
  /** 给出该建议的理由。 */
  reason: string
  /** 该建议所依赖指标的置信度。 */
  based_on_confidence: ConfidenceLevel
}

/** 基于画像生成的配置方案建议（草案）。 */
export interface SchemeSuggestion {
  /** 方案状态，恒为 `draft`（仅供参考、未落地）。 */
  status: 'draft'
  /** 建议适用的 K 线周期。 */
  interval: string
  /** 建议覆盖的合约代码列表。 */
  vt_symbols: string[]
  /** 逐字段建议列表。 */
  items: SuggestionItem[]
  /** 是否因样本不足等原因降级生成（建议可信度打折）。 */
  degraded: boolean
  /** 补充说明；无则为 null。 */
  note?: string | null
}

/** 目标标的与关联标的的协同画像。 */
export interface GroupProfile {
  /** 目标合约代码。 */
  target: string
  /** 参与协同分析的关联合约列表。 */
  members: string[]
  /** 时间轴对齐覆盖率（0–1），越高表示共同交易日越充分。 */
  alignment_coverage: number
  /** 各关联标的与目标的相关系数：合约代码 → 相关系数。 */
  correlation_summary: Record<string, number>
}

/** 画像实际生效的输入参数（请求经规范化后的快照）。 */
export interface ProfileInput {
  /** 目标合约代码。 */
  vt_symbol: string
  /** K 线周期。 */
  interval: string
  /** 画像基准日期（ISO 格式）。 */
  as_of: string
  /** 请求的回看天数。 */
  lookback_days: number
  /** 实际生效的数据右边界（ISO 格式）；无数据为 null。 */
  effective_right_bound?: string | null
  /** 实际参与计算的 bar 数。 */
  effective_bar_count: number
  /** 所用规则集版本标识。 */
  rules_id: string
}

/** 标的画像响应（`POST /api/alpha/profiling/profile` 响应体）。 */
export interface SymbolProfileResponse {
  /** 本次画像实际生效的输入快照。 */
  input: ProfileInput
  /** 画像是否成功产出；false 时见 unavailable_reason。 */
  available: boolean
  /** 画像不可用的原因；可用时为 null。 */
  unavailable_reason?: string | null
  /** 各维度指标分块列表。 */
  blocks: MetricBlock[]
  /** 协同画像；未请求关联标的或不可用时为 null。 */
  group_profile?: GroupProfile | null
  /** 配置方案建议；with_suggestion=false 时为 null。 */
  suggestion?: SchemeSuggestion | null
  /** 综合置信度（跨维度汇总）。 */
  overall_confidence: ConfidenceLevel
  /** 画像生成时刻（ISO 格式）。 */
  created_at: string
  /** 持久化后的 artifact ID；persist=false 时为 null。 */
  artifact_id?: string | null
}

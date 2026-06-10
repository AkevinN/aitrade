// 交易操作台（Trading Console）前端类型定义。
// 与后端 backend/aitrade/models/live.py、live/decision.py、live/orchestrator.py 对齐。

/** 组合快照（映射后端 PortfolioSnapshotRequest）。 */
export interface PortfolioSnapshotRequest {
  /** 组合总市值（现金+持仓） */
  portfolio_value: number
  /** 当前总持仓市值 */
  total_position_value?: number
  /** 目标标的当前持仓股数 */
  current_position?: number
  /** 目标标的当前持仓市值 */
  current_symbol_value?: number
}

/** 风控配置（映射后端 RiskConfigRequest）。 */
export interface RiskConfigRequest {
  /** 禁止买入的标的列表 */
  blacklist?: string[]
  /** 总持仓市值 / 组合市值 上限 */
  max_total_position_ratio?: number
  /** 单票市值 / 组合市值 上限 */
  max_single_position_ratio?: number
  /** 停牌/涨跌停封死时是否允许交易 */
  allow_when_halted?: boolean
}

/** 触发一次今日决策的请求体（映射后端 LiveDecisionRequest）。 */
export interface LiveDecisionRequest {
  /** CNN 模型名（必填） */
  model: string
  /** 目标标的（必填） */
  vt_symbol: string
  /** 方案名（必填） */
  scheme: string
  /** 决策时刻 ISO（缺省=当前）；仅 close_time<=as_of 的 bar 可见（无前视） */
  as_of?: string
  /** 决策 bar 频率；v1 仅 1d（日内属 Phase 2） */
  bar_freq?: string
  /** 数据源 */
  data_source?: 'upload' | 'pull'
  portfolio: PortfolioSnapshotRequest
  risk?: RiskConfigRequest
  /** 买入阈值 */
  buy_threshold?: number
  /** 目标仓位比例 */
  position_ratio?: number
  /** 最小成交手数 */
  min_volume?: number
  /** 模型版本，参与 signal_id */
  model_version?: string
  /** 目标标的当日是否停牌/封死 */
  halted?: boolean
  /** 是否触发出场 */
  should_exit?: boolean
}

// ============================================================
// 交易计划自动化（Trading Plan Automation）类型。
// 与后端 backend/aitrade/models/trading_plan.py、live/trading_plan.py 对齐。
// ============================================================

/** 通知通道名（仅名称，凭证由后端环境变量管理，前端永不接触）。 */
export type NotifyChannel = 'dingtalk' | 'wecom' | 'serverchan' | 'webhook'

/** 创建/更新交易计划的请求体（映射后端 TradingPlanRequest）。 */
export interface TradingPlanRequest {
  /** 计划名称（必填） */
  name: string
  model: string
  vt_symbol: string
  scheme: string
  buy_threshold?: number
  position_ratio?: number
  min_volume?: number
  model_version?: string
  data_source?: 'upload' | 'pull'
  should_exit?: boolean
  halted?: boolean
  portfolio: PortfolioSnapshotRequest
  risk?: RiskConfigRequest
  /** 是否启用自动调度 */
  enabled?: boolean
  /** 决策 bar 频率；v1 仅 1d（日内属 Phase 2） */
  bar_freq?: string
  /** 每交易日的调度唤醒时刻 HH:MM 列表；每个时刻触发一次（决策 bar 由 as_of 截断决定） */
  trigger_times?: string[]
  /** 通知通道名（仅名称，无凭证） */
  notify_channels?: NotifyChannel[]
}

/** 计划完整内容（映射后端 TradingPlan dataclass）。 */
export interface TradingPlan extends Required<Omit<TradingPlanRequest, 'risk' | 'notify_channels'>> {
  plan_id: string
  risk: RiskConfigRequest
  notify_channels: NotifyChannel[]
  created_at: string
  updated_at: string
}

/** 计划列表项摘要（映射后端 TradingPlanSummary）。 */
export interface TradingPlanSummary {
  plan_id: string
  name: string
  vt_symbol: string
  scheme: string
  /** 决策 bar 频率 */
  bar_freq: string
  /** 生效唤醒时刻集合（去重升序；多时刻展示用） */
  trigger_times?: string[]
  enabled: boolean
  /** 最近触发日 YYYY-MM-DD（来自 Last_Triggered_Map，取 date） */
  last_triggered?: string | null
}

/** 调度器运行状态（映射后端 SchedulerStatus）。 */
export interface SchedulerStatus {
  running: boolean
  tick_seconds: number
  enabled_plan_count: number
  /** {plan_id: "YYYY-MM-DD"} */
  last_triggered: Record<string, string>
}

/** 单条决策记录（映射后端 Decision dataclass）。 */
export interface Decision {
  /** 幂等键，如 "2026-06-08:eod_buy_v1:model@v3" */
  signal_id: string
  /** 决策 bar 时刻 ISO（取代 trade_date） */
  decision_bar_dt: string
  /** 决策时刻 ISO */
  as_of: string
  /** 决策 bar 频率（1d 即日频） */
  bar_freq: string
  scheme: string
  /** buy / sell / hold */
  action: string
  vt_symbol?: string | null
  volume: number
  price?: number | null
  signal?: number | null
  reason: string
  /** ISO 时间戳 */
  created_at: string
}

/** 风控逐项明细（映射后端 RiskInspector.records 元素）。 */
export interface RiskDetailItem {
  /** 风控检查项名称 */
  check: string
  /** 是否通过 */
  passed: boolean
  /** 明细说明 */
  detail: string
}

/** 今日决策编排结果（映射 run_live_decision 返回值）。 */
export interface LiveDecisionResult {
  /** 决策对象 */
  decision: Decision
  /** 风控逐项明细；幂等命中时为空数组 */
  risk_detail: RiskDetailItem[]
  /** 是否幂等命中（第二次触发，未重新走风控） */
  idempotent_hit: boolean
}

// ============================================================
// 决策过程可观测性（Requirement 8）：Decision_Trace 六段结构。
// 与后端 backend/aitrade/live/decision_trace.py、live/orchestrator.py 对齐。
// ============================================================

/** ① 运行头段（脱敏摘要：无凭证；风控仅摘要；数据源仅类型）。 */
export interface TraceRunHeaderSection {
  /** 本次运行短码（uuid4 前 8 位） */
  run_id: string
  /** CNN 模型名 */
  model_name: string
  /** 模型版本 */
  model_version?: string | null
  /** 目标标的 */
  vt_symbol: string
  /** 方案名 */
  scheme: string
  /** 决策时刻 ISO（取代 trade_date） */
  as_of: string
  /** 决策 bar 频率 */
  bar_freq: string
  /** 数据源类型，仅类型不含 token */
  data_source_type: string
  /** 买入阈值 */
  buy_threshold: number
  /** 组合快照摘要 */
  portfolio: {
    portfolio_value: number
    total_position_value?: number | null
    current_position?: number | null
    current_symbol_value?: number | null
  }
  /** 风控配置摘要（仅比率与黑名单长度，不展开敏感细节） */
  risk_config_summary: {
    max_total_position_ratio?: number | null
    max_single_position_ratio?: number | null
    allow_when_halted?: boolean | null
    /** 黑名单条目数，仅长度 */
    blacklist_size: number
  }
}

/** 信号序列统计（编排器对 signal_df 的聚合）。 */
export interface TraceSignalSeqStats {
  count: number
  mean: number
  min: number
  max: number
}

/** ② 推理段（on_meta 元信息 + signal_df 序列统计）。 */
export interface TraceInferenceSection {
  /** 目标标的 */
  target_symbol: string
  /** 回看窗口 */
  lookback: number
  /** 输入周期 */
  input_interval: string
  /** 训练目标，如 classification */
  objective: string
  /** 观测标的列表 */
  observation_symbols: string[]
  /** 观测分组数 */
  observation_group_count: number
  /** 预热起点（extended_start）YYYY-MM-DD */
  warmup_start: string
  /** 对齐总步数 */
  total_steps: number
  /** 有效推理点数 */
  valid_points: number
  /** 各标的 bar 数量 */
  per_symbol_bars: Record<string, number>
  /** 信号序列统计 */
  signal_seq_stats: TraceSignalSeqStats
  /** Decision_Bar 的 signal 取值 */
  decision_signal: number
  /** Decision_Bar 时刻 ISO */
  decision_bar_dt?: string
}

/** ③ 取价段。 */
export interface TracePricingSection {
  /** 取价周期 */
  interval_used: string
  /** 决策日收盘价 */
  close_price: number
}

/** ④ 决策逻辑段（信号 vs 阈值 + 仓位规模）。 */
export interface TraceDecisionLogicSection {
  /** 决策日信号 */
  signal: number
  /** 买入阈值 */
  buy_threshold: number
  /** 信号是否达标 */
  signal_passed: boolean
  /** 目标仓位市值 */
  target_value: number
  /** 计划成交手数 */
  volume: number
  /** 计划成交市值 */
  intended_value: number
  /** 是否触发出场 */
  should_exit: boolean
  /** 是否停牌/封死 */
  halted: boolean
}

/** ⑤ 风控段（复用 RiskInspector.records）。 */
export interface TraceRiskSection {
  /** 风控逐项明细，复用 RiskDetailItem */
  records: RiskDetailItem[]
  /** 与 RiskManager.check_buy 权威结论是否一致 */
  authoritative_ok: boolean
}

/** ⑥ 结果段（含可观测标记）。中止运行时 action/volume/price/reason 为 null。 */
export interface TraceResultSection {
  /** buy / sell / hold；中止时为 null */
  action: string | null
  /** 成交手数；中止时为 null */
  volume: number | null
  /** 成交价；中止时为 null */
  price: number | null
  /** 决策理由；中止时为 null */
  reason: string | null
  /** 是否幂等命中（8.10） */
  idempotent_hit: boolean
  /** 是否触发提醒 */
  notified: boolean
  /** 关联的 signal_id */
  signal_id: string
  /** best-effort 持久化结果（8.12） */
  trace_persisted: boolean
  /** 持久化失败原因；成功为 null */
  trace_persist_error: string | null
  /** 中止原因（8.11）；正常完成为 null */
  abort_reason: string | null
}

/** 六段 Trace_Section 的分组容器（按段名索引，缺段可能不存在）。 */
export interface TraceSection {
  run_header?: TraceRunHeaderSection
  inference?: TraceInferenceSection
  pricing?: TracePricingSection
  decision_logic?: TraceDecisionLogicSection
  risk?: TraceRiskSection
  result?: TraceResultSection
}

/** 已完成段名（顺序枚举）。 */
export type TraceSectionName =
  | 'run_header'
  | 'inference'
  | 'pricing'
  | 'decision_logic'
  | 'risk'
  | 'result'

/** 完整决策过程档案（映射后端 TraceBuilder.to_trace 产物 / {signal_id}.trace.json）。 */
export interface DecisionTrace {
  /** trace schema 版本 */
  schema_version: number
  /** 本次运行短码 */
  run_id: string
  /** 与对应 Decision 完全一致的关联键 */
  signal_id: string
  /** 已完成段（顺序）；正常运行为全六段，中止时为失败点之前的前缀 */
  completed_sections: TraceSectionName[]
  /** 六段分组明细 */
  sections: TraceSection
}

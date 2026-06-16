import type { LabelSpec, ObservationGroup, TaskStartResponse } from './alpha'

/**
 * 治理回演基准类型：与当前候选模型比较的基准策略。
 *
 * - `fixed_initial_model`：固定初始模型不更新；
 * - `always_retrain`：每期强制重训；
 * - `governed_promotion`：治理晋升策略；
 * - `buy_and_hold`：买入持有基准。
 */
export type GovernanceBaseline =
  | 'fixed_initial_model'
  | 'always_retrain'
  | 'governed_promotion'
  | 'buy_and_hold'

/** CNN 治理全局配置（`GET /PUT /api/cnn/governance/config`），控制评估周期与默认窗口。 */
export interface CNNGovernanceConfig {
  /** 是否启用治理模块；关闭后不再触发自动评估/晋级 */
  enabled: boolean
  /** 默认评估周期，单位天 */
  evaluation_period_days: number
  /** 默认训练窗口，单位天 */
  train_days: number
  /** 默认 OOS（样本外）测试窗口，单位天 */
  test_days: number
  /** 默认每折重复训练的随机种子数；1=单种子，>1 时折内对多种子取均值并衡量波动 */
  n_seeds: number
  /** 系统建议的下一次评估日期（YYYY-MM-DD）；无建议时为 null */
  next_suggested_eval_date?: string | null
  /** 是否自动晋级；false 时需人工确认，第一版保守运营默认 false */
  auto_promote: boolean
}

/** CNN 训练超参数集合（用于治理/候选训练请求），与单次训练接口的参数结构对齐。 */
export interface CNNTrainingParams {
  /** 训练轮数 */
  epochs: number
  /** 批大小 */
  batch_size: number
  /** 学习率 */
  learning_rate: number
  /** 输入序列回看长度（时间步数） */
  lookback: number
  /** Dropout 比例，取值 [0, 1) */
  dropout: number
  /** 训练集占比，取值 (0, 1)，其余为验证集 */
  train_ratio: number
  /** 损失加权：none=不加权；magnitude=按涨跌幅度加权 */
  loss_weighting: 'none' | 'magnitude'
}

/** CNN 回测参数集合（用于治理/候选评估请求）：阈值、A 股交易成本与出场规则。 */
export interface CNNBacktestParams {
  /** 买入信号概率阈值 */
  buy_threshold: number
  /** 卖出信号概率阈值 */
  sell_threshold: number
  /** 单边佣金率（如 0.0003=万3） */
  commission_rate: number
  /** 卖出印花税率（如 0.0005=千0.5，A 股） */
  stamp_duty: number
  /** 每笔成交不利滑点率 */
  slippage: number
  /** 限价单价格缓冲/市价化挂单比例 */
  price_add: number
  /** 出场模式：threshold=概率阈值；fixed_hold=固定持有；oco=止盈止损；auto=按 label 自动对齐 */
  exit_mode: 'threshold' | 'fixed_hold' | 'oco' | 'auto'
  /** fixed_hold/oco 的固定/最大持有交易日数 */
  hold_days: number
  /** oco 止盈幅度（0.02=+2%），0=不启用 */
  take_profit: number
  /** oco 止损幅度（0.03=-3%），0=不启用 */
  stop_loss: number
  /** 是否启用 T+1 卖出限制（A 股现实约束） */
  t_plus1: boolean
}

/**
 * CNN 候选模型晋升门槛配置（Walk-Forward 折内胜出条件）。
 *
 * 候选需同时满足：胜出折数比例 >= min_win_rate、核心分数提升 >= min_core_score_delta、
 * 最大回撤劣化 <= max_drawdown_worsen_pct，方可晋级（或提示人工确认）。
 */
export interface CNNPromotionGate {
  /** 候选胜出折数比例下限，取值 [0, 1] */
  min_win_rate: number
  /** 核心分数相对生产模型的平均提升下限（绝对值，可为负以放宽） */
  min_core_score_delta: number
  /** 最大回撤允许劣化的百分比上限 */
  max_drawdown_worsen_pct: number
  /** 是否要求 OOS（样本外）核心指标为正 */
  require_positive_oos: boolean
}

/**
 * CNN Walk-Forward 评估请求（`POST /api/cnn/governance/evaluate`）。
 *
 * 以 train_days 训练窗 + test_days OOS 测试窗逐步前推，多折评估候选相对生产模型的胜出情况。
 */
export interface CNNWalkForwardRequest {
  /** 评估任务名称 */
  name: string
  /** 目标合约代码 */
  target_symbol: string
  /** 输入数据类型：bar=K 线；tick=逐笔 */
  input_data_kind: 'bar' | 'tick'
  /** 输入数据周期，如 "d"（日线）、"1m"/"30m"（分钟线） */
  input_interval: string
  /** 评估区间起始日期（YYYY-MM-DD，含） */
  start: string
  /** 评估区间结束日期（YYYY-MM-DD，含） */
  end: string
  /** 单折训练窗口长度（天） */
  train_days: number
  /** 单折 OOS 测试窗口长度（天） */
  test_days: number
  /** 相邻折前推步长（天）；为空时默认等于 test_days（无重叠滚动） */
  step_days?: number
  /** 预测目标：classification=方向二分类；regression=涨跌幅回归 */
  objective: 'classification' | 'regression'
  /** 标签生成规格（涨跌阈值/持有窗等） */
  label_spec: LabelSpec
  /** 观测特征分组配置 */
  observation_groups: ObservationGroup[]
  /** 训练超参数 */
  training_params: CNNTrainingParams
  /** 回测参数（阈值/成本/出场） */
  backtest_params: CNNBacktestParams
  /** 候选晋级门槛 */
  promotion_gate: CNNPromotionGate
  /** 用于相对胜出比较的生产模型；为空则读取当前生产模型 */
  production_model?: string
}

/** CNN 候选模型完整训练请求（`POST /api/cnn/governance/candidates/train`），在 Walk-Forward 基础上增加最终训练区间。 */
export interface CNNCandidateTrainRequest extends CNNWalkForwardRequest {
  /** 最终候选训练起始日期（YYYY-MM-DD）；为空时沿用请求 start */
  final_train_start?: string
  /** 最终候选训练结束日期（YYYY-MM-DD）；为空时沿用最后一个训练窗口结束 */
  final_train_end?: string
}

/**
 * CNN 治理历史重演请求（`POST /api/cnn/governance/replay/run`）。
 *
 * 模拟历史上若启用治理流程的收益/风险：以 initial_train_days 启动，每 evaluation_period_days
 * 评估一次，并与 baselines 中的各基准策略横向对比。
 */
export interface CNNGovernanceReplayRequest {
  /** 重演任务名称 */
  name: string
  /** 目标合约代码 */
  target_symbol: string
  /** 输入数据类型：bar=K 线；tick=逐笔 */
  input_data_kind: 'bar' | 'tick'
  /** 输入数据周期，如 "d"、"30m" */
  input_interval: string
  /** 重演区间起始日期（YYYY-MM-DD，含） */
  start: string
  /** 重演区间结束日期（YYYY-MM-DD，含） */
  end: string
  /** 启动时的初始训练窗口长度（天） */
  initial_train_days: number
  /** 每隔多少天评估并考虑晋级一次（天） */
  evaluation_period_days: number
  /** 每次评估的 OOS 测试窗口长度（天） */
  test_period_days: number
  /** 初始资金，单位元 */
  capital: number
  /** 预测目标：classification=方向二分类；regression=涨跌幅回归 */
  objective: 'classification' | 'regression'
  /** 标签生成规格 */
  label_spec: LabelSpec
  /** 观测特征分组配置 */
  observation_groups: ObservationGroup[]
  /** 训练超参数 */
  training_params: CNNTrainingParams
  /** 回测参数 */
  backtest_params: CNNBacktestParams
  /** 候选晋级门槛 */
  promotion_gate: CNNPromotionGate
  /** 参与横向对比的基准策略集合 */
  baselines: GovernanceBaseline[]
}

/**
 * 当前生产模型状态（`GET /api/cnn/governance/production` 响应体），即持久化的晋级记录快照。
 *
 * previous_* 字段保留上一版生产模型信息，供回滚使用。
 */
export interface CNNProductionModel {
  /** 当前生产模型名称 */
  model_name: string
  /** 当前生产模型版本号 */
  model_version: string
  /** 目标合约代码 */
  target_symbol: string
  /** 输入数据周期，如 "d" */
  input_interval: string
  /** 预测目标（classification / regression / path_class） */
  objective: string
  /** 最近一次晋级时刻（ISO 时间串）；从未晋级时为空 */
  promoted_at?: string | null
  /** 最近一次晋级的操作者 */
  promoted_by: string
  /** 晋级所依据的治理评估报告 ID */
  report_id: string
  /** 上一版生产模型名称（回滚目标） */
  previous_model_name: string
  /** 上一版生产模型版本号 */
  previous_model_version: string
}

/** CNN 候选模型记录（`GET /api/cnn/governance/candidates` 列表元素）。 */
export interface CNNCandidate {
  /** 候选记录唯一 ID */
  candidate_id: string
  /** 候选创建时刻（ISO 时间串） */
  created_at: string
  /** 候选状态：pending=待评估；passed=通过门槛；failed=未达门槛；promoted=已晋级；rejected=人工驳回 */
  status: 'pending' | 'passed' | 'failed' | 'promoted' | 'rejected'
  /** 候选模型名称 */
  model_name: string
  /** 关联的治理评估报告 ID */
  report_id: string
  /** 目标合约代码 */
  target_symbol: string
  /** 输入数据周期，如 "d" */
  input_interval: string
  /** 预测目标（classification / regression / path_class） */
  objective: string
  /** 评估时对比的基准生产模型名称；无对比时缺省 */
  baseline_model?: string
  /** 评估结果摘要（指标键值对），结构由后端决定 */
  summary?: Record<string, unknown>
  /** 触发本候选的原始训练请求快照，结构由后端决定 */
  request?: Record<string, unknown>
}

/** CNN 治理评估报告（`GET /api/cnn/governance/reports/{reportId}` 响应体）。 */
export interface CNNGovernanceReport {
  /** 报告唯一 ID */
  report_id: string
  /** 报告类型（如 walk_forward / candidate） */
  type: string
  /** 报告名称（取自评估请求的 name） */
  name: string
  /** 报告生成时刻（ISO 时间串） */
  created_at: string
  /** 评估时对比的生产模型名称；首次评估无对比时缺省 */
  production_model?: string
  /** 逐折评估明细，每个元素为一折的指标键值对，结构由后端决定 */
  folds: Array<Record<string, unknown>>
  /** 跨折汇总指标与晋级判定结果，结构由后端决定 */
  summary: Record<string, unknown>
}

/** CNN 治理历史重演结果报告（`GET /api/cnn/governance/replay/{replayId}` 响应体）。 */
export interface CNNGovernanceReplayReport {
  /** 重演任务唯一 ID */
  replay_id: string
  /** 重演任务名称 */
  name: string
  /** 目标合约代码 */
  target_symbol: string
  /** 重演区间起始日期（YYYY-MM-DD） */
  start: string
  /** 重演区间结束日期（YYYY-MM-DD） */
  end: string
  /** 报告生成时刻（ISO 时间串） */
  created_at: string
  /** 各基准策略的回测结果，键为基准名（见 GovernanceBaseline），值为指标键值对 */
  baselines: Record<string, Record<string, unknown>>
  /** 治理流程逐期执行轨迹，结构由后端决定 */
  process: Record<string, unknown>
  /** 重演期间发生的晋级事件列表，结构由后端决定 */
  promotion_events: Array<Record<string, unknown>>
  /** 重演期间被驳回（未达门槛）的候选事件列表，结构由后端决定 */
  rejected_events: Array<Record<string, unknown>>
  /** 诊断信息（数据缺口、异常折等），结构由后端决定 */
  diagnostics: Record<string, unknown>
  /** 总体结论（治理 vs 各基准的优劣判定），结构由后端决定 */
  conclusion: Record<string, unknown>
}

/** 治理历史事件条目（`GET /api/cnn/governance/history` 列表元素），即审计日志记录。 */
export interface GovernanceHistoryEvent {
  /** 事件发生时刻（ISO 时间串） */
  ts: string
  /** 事件类型，如 promote（晋级）/ rollback（回滚）/ evaluate（评估） */
  event_type: string
  /** 事件载荷，含操作参数与判定结果，结构随 event_type 而定 */
  payload: Record<string, unknown>
}

export type { TaskStartResponse }

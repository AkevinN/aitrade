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

/** CNN 治理全局配置（`GET /PUT /api/cnn/governance/config`）。 */
export interface CNNGovernanceConfig {
  enabled: boolean
  evaluation_period_days: number
  train_days: number
  test_days: number
  n_seeds: number
  next_suggested_eval_date?: string | null
  auto_promote: boolean
}

/** CNN 训练超参数集合（用于治理/候选训练请求）。 */
export interface CNNTrainingParams {
  epochs: number
  batch_size: number
  learning_rate: number
  lookback: number
  dropout: number
  train_ratio: number
  loss_weighting: 'none' | 'magnitude'
}

/** CNN 回测参数集合（用于治理/候选评估请求）。 */
export interface CNNBacktestParams {
  buy_threshold: number
  sell_threshold: number
  commission_rate: number
  stamp_duty: number
  slippage: number
  price_add: number
  exit_mode: 'threshold' | 'fixed_hold' | 'oco' | 'auto'
  hold_days: number
  take_profit: number
  stop_loss: number
  t_plus1: boolean
}

/** CNN 候选模型晋升门槛配置。 */
export interface CNNPromotionGate {
  min_win_rate: number
  min_core_score_delta: number
  max_drawdown_worsen_pct: number
  require_positive_oos: boolean
}

/** CNN Walk-Forward 评估请求（`POST /api/cnn/governance/evaluate`）。 */
export interface CNNWalkForwardRequest {
  name: string
  target_symbol: string
  input_data_kind: 'bar' | 'tick'
  input_interval: string
  start: string
  end: string
  train_days: number
  test_days: number
  step_days?: number
  objective: 'classification' | 'regression'
  label_spec: LabelSpec
  observation_groups: ObservationGroup[]
  training_params: CNNTrainingParams
  backtest_params: CNNBacktestParams
  promotion_gate: CNNPromotionGate
  production_model?: string
}

/** CNN 候选模型完整训练请求（`POST /api/cnn/governance/candidates/train`），在 Walk-Forward 基础上增加最终训练区间。 */
export interface CNNCandidateTrainRequest extends CNNWalkForwardRequest {
  final_train_start?: string
  final_train_end?: string
}

/** CNN 治理历史重演请求（`POST /api/cnn/governance/replay/run`）。 */
export interface CNNGovernanceReplayRequest {
  name: string
  target_symbol: string
  input_data_kind: 'bar' | 'tick'
  input_interval: string
  start: string
  end: string
  initial_train_days: number
  evaluation_period_days: number
  test_period_days: number
  capital: number
  objective: 'classification' | 'regression'
  label_spec: LabelSpec
  observation_groups: ObservationGroup[]
  training_params: CNNTrainingParams
  backtest_params: CNNBacktestParams
  promotion_gate: CNNPromotionGate
  baselines: GovernanceBaseline[]
}

/** 当前生产模型状态（`GET /api/cnn/governance/production` 响应体）。 */
export interface CNNProductionModel {
  model_name: string
  model_version: string
  target_symbol: string
  input_interval: string
  objective: string
  promoted_at?: string | null
  promoted_by: string
  report_id: string
  previous_model_name: string
  previous_model_version: string
}

/** CNN 候选模型记录（`GET /api/cnn/governance/candidates` 列表元素）。 */
export interface CNNCandidate {
  candidate_id: string
  created_at: string
  status: 'pending' | 'passed' | 'failed' | 'promoted' | 'rejected'
  model_name: string
  report_id: string
  target_symbol: string
  input_interval: string
  objective: string
  baseline_model?: string
  summary?: Record<string, unknown>
  request?: Record<string, unknown>
}

/** CNN 治理评估报告（`GET /api/cnn/governance/reports/{reportId}` 响应体）。 */
export interface CNNGovernanceReport {
  report_id: string
  type: string
  name: string
  created_at: string
  production_model?: string
  folds: Array<Record<string, unknown>>
  summary: Record<string, unknown>
}

/** CNN 治理历史重演结果报告（`GET /api/cnn/governance/replay/{replayId}` 响应体）。 */
export interface CNNGovernanceReplayReport {
  replay_id: string
  name: string
  target_symbol: string
  start: string
  end: string
  created_at: string
  baselines: Record<string, Record<string, unknown>>
  process: Record<string, unknown>
  promotion_events: Array<Record<string, unknown>>
  rejected_events: Array<Record<string, unknown>>
  diagnostics: Record<string, unknown>
  conclusion: Record<string, unknown>
}

/** 治理历史事件条目（`GET /api/cnn/governance/history` 列表元素）。 */
export interface GovernanceHistoryEvent {
  ts: string
  event_type: string
  payload: Record<string, unknown>
}

export type { TaskStartResponse }

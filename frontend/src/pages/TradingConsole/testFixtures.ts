// 交易操作台测试夹具：构造确定性的 Task / Decision / 风控明细对象，
// 供示例/快照测试使用，避免各测试文件重复样板。
import type { Task } from '../../types/alpha'
import type { Decision, LiveDecisionResult, RiskDetailItem } from '../../types/live'
import type {
  DecisionTrace,
  SchedulerStatus,
  TradingPlan,
  TradingPlanSummary,
} from '../../types/live'

/** 构造一个最小可用的 Task，可按需覆盖字段。 */
export function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    task_id: 'task-1',
    type: 'live_decision',
    title: '今日决策',
    entity_type: 'live',
    entity_name: '000001.SZSE',
    status: 'running',
    progress: 0,
    message: '',
    result: null,
    created_at: '2026-06-08T14:30:00',
    updated_at: '2026-06-08T14:31:00',
    ...overrides,
  }
}

/** 构造一个 Decision，可按需覆盖字段。 */
export function makeDecision(overrides: Partial<Decision> = {}): Decision {
  return {
    signal_id: '2026-06-08:eod_buy_v1:demo@v1',
    decision_bar_dt: '2026-06-08T15:00:00',
    as_of: '2026-06-08T15:05:00',
    bar_freq: '1d',
    scheme: 'eod_buy_v1',
    action: 'buy',
    vt_symbol: '000001.SZSE',
    volume: 1000,
    price: 12.34,
    signal: 0.82,
    reason: '信号超过买入阈值',
    created_at: '2026-06-08T14:31:00',
    ...overrides,
  }
}

/** 风控明细（默认 5 项全通过），可按需覆盖。 */
export function makeRiskDetail(overrides: RiskDetailItem[] = []): RiskDetailItem[] {
  if (overrides.length > 0) return overrides
  return [
    { check: 'kill_switch_or_circuit', passed: true, detail: '未触发 kill-switch' },
    { check: 'blacklist', passed: true, detail: '不在黑名单' },
    { check: 'halted', passed: true, detail: '正常交易' },
    { check: 'max_total_position', passed: true, detail: '总仓位在上限内' },
    { check: 'max_single_position', passed: true, detail: '单票仓位在上限内' },
  ]
}

/** 构造一个完成态决策结果（写入 task.result）。返回宽松 Record 以匹配 Task.result 类型。 */
export function makeResult(overrides: Partial<LiveDecisionResult> = {}): Record<string, unknown> {
  const result: LiveDecisionResult = {
    decision: makeDecision(),
    risk_detail: makeRiskDetail(),
    idempotent_hit: false,
    ...overrides,
  }
  return result as unknown as Record<string, unknown>
}

/** 构造交易计划摘要，可按需覆盖。 */
export function makePlanSummary(overrides: Partial<TradingPlanSummary> = {}): TradingPlanSummary {
  return {
    plan_id: 'plan-1',
    name: '尾盘买入计划',
    vt_symbol: '000001.SZSE',
    scheme: 'eod_buy_v1',
    bar_freq: '1d',
    trigger_times: ['15:05'],
    enabled: false,
    last_triggered: null,
    strategy_type: 'cnn',
    portfolio_id: '',
    signal_source: '',
    ...overrides,
  }
}

/** 构造交易计划完整对象，可按需覆盖。 */
export function makePlan(overrides: Partial<TradingPlan> = {}): TradingPlan {
  return {
    plan_id: 'plan-1',
    name: '尾盘买入计划',
    model: 'demo',
    vt_symbol: '000001.SZSE',
    scheme: 'eod_buy_v1',
    buy_threshold: 0.6,
    position_ratio: 0.95,
    min_volume: 100,
    model_version: 'v1',
    data_source: 'pull',
    should_exit: false,
    halted: false,
    portfolio: {
      portfolio_value: 1000000,
      total_position_value: 0,
      current_position: 0,
      current_symbol_value: 0,
    },
    risk: {},
    enabled: false,
    bar_freq: '1d',
    trigger_times: ['15:05'],
    notify_channels: ['dingtalk'],
    created_at: '2026-06-08T14:30:00',
    updated_at: '2026-06-08T14:30:00',
    ...overrides,
  }
}

/** 构造调度器状态，可按需覆盖。 */
export function makeSchedulerStatus(overrides: Partial<SchedulerStatus> = {}): SchedulerStatus {
  return {
    running: true,
    tick_seconds: 30,
    enabled_plan_count: 1,
    last_triggered: {},
    ...overrides,
  }
}

/** 构造一份完整六段 Decision_Trace，可按需覆盖顶层字段。 */
export function makeTrace(overrides: Partial<DecisionTrace> = {}): DecisionTrace {
  return {
    schema_version: 1,
    run_id: 'abcd1234',
    signal_id: '2026-06-08:eod_buy_v1:demo@v1',
    completed_sections: [
      'run_header',
      'inference',
      'pricing',
      'decision_logic',
      'risk',
      'result',
    ],
    sections: {
      run_header: {
        run_id: 'abcd1234',
        model_name: 'demo',
        model_version: 'v1',
        vt_symbol: '000001.SZSE',
        scheme: 'eod_buy_v1',
        as_of: '2026-06-08T15:05:00',
        bar_freq: '1d',
        data_source_type: 'upload',
        buy_threshold: 0.6,
        portfolio: {
          portfolio_value: 1000000,
          total_position_value: 200000,
          current_position: 0,
          current_symbol_value: 0,
        },
        risk_config_summary: {
          max_total_position_ratio: 0.95,
          max_single_position_ratio: 0.3,
          allow_when_halted: false,
          blacklist_size: 2,
        },
      },
      inference: {
        target_symbol: '000001.SZSE',
        lookback: 240,
        input_interval: '30m',
        objective: 'classification',
        observation_symbols: ['000001.SZSE', '399001.SZSE'],
        observation_group_count: 1,
        warmup_start: '2026-01-01',
        total_steps: 480,
        valid_points: 460,
        per_symbol_bars: { '000001.SZSE': 480, '399001.SZSE': 480 },
        signal_seq_stats: { count: 460, mean: 0.45, min: 0.01, max: 0.92 },
        decision_signal: 0.82,
        decision_bar_dt: '2026-06-08T15:00:00',
      },
      pricing: {
        interval_used: '30m',
        close_price: 12.34,
      },
      decision_logic: {
        signal: 0.82,
        buy_threshold: 0.6,
        signal_passed: true,
        target_value: 300000,
        volume: 1000,
        intended_value: 12340,
        should_exit: false,
        halted: false,
      },
      risk: {
        records: makeRiskDetail(),
        authoritative_ok: true,
      },
      result: {
        action: 'buy',
        volume: 1000,
        price: 12.34,
        reason: '信号超过买入阈值',
        idempotent_hit: false,
        notified: true,
        signal_id: '2026-06-08:eod_buy_v1:demo@v1',
        trace_persisted: true,
        trace_persist_error: null,
        abort_reason: null,
      },
    },
    ...overrides,
  }
}

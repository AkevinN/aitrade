// 与 backend/aitrade/models/strategy.py 对齐
// Rule-based strategy types for frontend ↔ backend communication

/**
 * 可用信号源的元信息（`GET /api/strategy/sources` 列表元素）。
 *
 * 前端用 `param_spec` 动态渲染信号参数表单，不需要硬编码各信号源的字段。
 */
export interface SignalSourceInfo {
  /** 信号源唯一名称，如 "etf_momentum" */
  name: string
  /** 信号源简短说明，展示在 Select 选项中 */
  description: string
  /**
   * 可配置参数规格；null 表示无额外参数。
   * 键为参数名，值含类型（int/float/str/list[str]）、默认值与显示标签。
   */
  param_spec: Record<string, { type?: string; default?: unknown; label?: string }> | null
}

/**
 * 规则策略回测的成本假设（映射后端 StrategyCostRequest）。
 *
 * 各字段以小数表示比率（如 0.0003 = 万3）。
 */
export interface StrategyCost {
  /** 单边佣金率，如 0.0003 */
  commission_rate: number
  /** 卖出印花税率，如 0.0005 */
  stamp_duty: number
  /** 成交滑点比率，如 0.0005 */
  slippage: number
  /** 是否启用 T+1 限制（当日买入不可当日卖出） */
  t_plus1: boolean
}

/**
 * 启动单次规则策略回测的请求体（`POST /api/strategy/backtest/run`）。
 *
 * 与后端 `StrategyBacktestRequest` Pydantic 模型对齐。
 */
export interface StrategyBacktestRequest {
  /** 信号源名称，需与 `GET /api/strategy/sources` 返回的 name 一致 */
  signal_source: string
  /** 信号源参数，必须包含 universe（标的池列表） */
  signal_params: Record<string, unknown>
  /** 策略名称（可选，留空则由后端自动生成） */
  strategy_name?: string
  /** 策略参数，如 top_k / n_drop / rebalance_freq */
  strategy_params: Record<string, unknown>
  /** 行情周期，如 "d"（日线）；缺省为 "d" */
  interval?: string
  /** 回测起始日期 YYYY-MM-DD（含） */
  start: string
  /** 回测结束日期 YYYY-MM-DD（含） */
  end: string
  /** 初始资金（元）；缺省 1000000 */
  capital?: number
  /** 成本假设；缺省使用后端 A 股默认值 */
  cost?: StrategyCost
}

/**
 * 启动参数网格扫描的请求体（`POST /api/strategy/sweep/run`）。
 *
 * 在 `StrategyBacktestRequest` 基础上追加 `grid` 网格列表，每行为一个参数覆盖组合。
 *
 * @example
 * ```json
 * { "strategy_params": { "top_k": 3 } }
 * { "strategy_params": { "top_k": 5 } }
 * ```
 */
export interface StrategySweepRequest extends StrategyBacktestRequest {
  /**
   * 参数网格列表，每个元素可覆盖 strategy_params 或 signal_params。
   * 后端对每个组合独立跑一次回测，返回 `rows: SweepRow[]`。
   */
  grid: Array<{
    strategy_params?: Record<string, unknown>
    signal_params?: Record<string, unknown>
  }>
}

/**
 * 启动 Walk-Forward 验证的请求体（`POST /api/strategy/walkforward/run`）。
 *
 * 以滚动窗口在时间轴上依次训练→测试，评估策略参数的时序稳定性。
 */
export interface StrategyWalkForwardRequest extends StrategyBacktestRequest {
  /** 每折训练窗口长度（自然日数），建议 ≥ 90 */
  train_days: number
  /** 每折测试窗口长度（自然日数），建议 ≥ 30 */
  test_days: number
}

/**
 * 参数扫描 / Walk-Forward 单行结果（`result.rows` 列表元素）。
 *
 * 用于在扫描结果表格中对比各参数组合的回测表现。
 */
export interface SweepRow {
  /** 当前行覆盖的参数字典 */
  params: Record<string, unknown>
  /** 总收益率（小数，如 0.25 = 25%） */
  total_return: number
  /** 夏普比率 */
  sharpe_ratio: number
  /** 最大回撤百分比（小数，如 0.15 = 15%） */
  max_ddpercent: number
  /** 总净收益（元） */
  total_net_pnl: number
  /** 总成交笔数 */
  trade_count: number
}

// Re-export backtest result types from alpha (reuse existing types)
export type { BacktestStatistics, BacktestResultPayload } from './alpha'

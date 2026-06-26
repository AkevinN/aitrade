// 半仓做 T 回测类型 — 对应后端 /api/t0/backtest（同步返回 T0Report）

/** 单个成交假设：穿越阈值（元）+ 单根触价成交比例。 */
export interface T0FillCfg {
  /** 穿越阈值 ε，单位元；0 = 触价即成交 */
  penetration: number
  /** 单根触价 bar 成交比例，1.0 = 全额 */
  ratio: number
}

/** 半仓做 T 回测请求体。 */
export interface T0BacktestRequest {
  /** 标的 vt_symbol，如 "000415.SZSE" */
  symbol: string
  /** 评估窗起（YYYY-MM-DD，含） */
  start: string
  /** 评估窗止（YYYY-MM-DD，含） */
  end: string
  /** 卖单挂高价差（元） */
  sell_tick: number
  /** 买单挂低价差（元） */
  buy_tick: number
  /** 做 T 摆动占半仓比例，1.0 = 全半仓摆动 */
  swing_frac: number
  /** 半仓锚权重 */
  base_weight: number
  /** 初始资金（元） */
  capital: number
  /** 单边佣金率 */
  commission_rate: number
  /** 卖出印花税率 */
  stamp_duty: number
  /** 成交假设网格 */
  fill_grid: T0FillCfg[]
}

/** 逐年（或逐月）收益与超额行。 */
export interface T0PeriodRow {
  year?: number
  ym?: string
  strat: number
  bh: number
  half_bh: number
  excess_vs_bh: number
  excess_vs_half_bh: number
}

/** 命中分布：开盘±档位是否被触及的占比。 */
export interface T0HitDist {
  both: number
  onlyS: number
  onlyB: number
  none: number
}

/** 单个 (档位 × 成交假设) 组合的结果。 */
export interface T0RunResult {
  tick_label: string
  fill: T0FillCfg
  total_return: number
  cagr: number
  sharpe: number
  max_drawdown: number
  turnover_annual: number
  yearly: T0PeriodRow[]
  monthly_excess: { ym: string; excess_vs_bh: number }[]
  hit_dist: T0HitDist
}

/** 成交敏感性区间的一行（按成交假设汇总）。 */
export interface T0FillSensitivityRow {
  tick_label: string
  fill: T0FillCfg
  total_return: number
  sharpe: number
  max_drawdown: number
}

/** 做 T 回测总报告。 */
export interface T0Report {
  symbol: string
  eval_window: [string, string]
  fill_sensitivity: T0FillSensitivityRow[]
  results: T0RunResult[]
}

/** 标的做 T 画像请求。 */
export interface T0ProfileRequest {
  symbol: string
  /** 标定窗起（YYYY-MM-DD） */
  start: string
  /** 标定窗止（YYYY-MM-DD） */
  end: string
  /** 档位网格上限（分） */
  x_max_fen: number
  commission_rate: number
  stamp_duty: number
}

/** 单档位逐买卖腿的回归边际收益行。 */
export interface T0BandEdgeRow {
  /** 偏离开盘价的档位（分） */
  x_fen: number
  /** 卖腿成交率 P(high>=O+x) */
  sell_fill: number
  /** 卖腿净于成本的每笔边际收益（分） */
  sell_edge_fen: number
  /** 买腿成交率 P(low<=O-x) */
  buy_fill: number
  /** 买腿净于成本的每笔边际收益（分） */
  buy_edge_fen: number
  /** 全日期望盈亏（分） */
  day_pnl_fen: number
}

/** 标的做 T 画像（偏离-回归边际曲线 + 建议档位）。 */
export interface T0Profile {
  symbol: string
  window: [string, string]
  rows: T0BandEdgeRow[]
  suggested_sell_tick: number
  suggested_buy_tick: number
  note: string
  calib_mean_range?: number | null
}

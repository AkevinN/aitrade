/**
 * CNN 选股（CNN Stock Screening）前端类型定义。
 *
 * 与后端 `backend/aitrade/screening/types.py` 及 `backend/aitrade/models/screening.py`
 * 的 Pydantic 模型一一对应（snake_case 字段名、类型同构）。
 * 前端仅消费，不做写入，因此所有字段均为只读语义。
 */

import type { LabelSpec } from './alpha'

/** CNN 适配度评分维度的置信度等级（由后端 ConfidenceLevel 字面量对齐）。 */
export type ConfidenceLevel = 'insufficient' | 'low' | 'medium' | 'high'

/** CNN 训练目标，与 `CNNScreeningRequest.objective` 对齐。 */
export type ObjectiveType = 'classification' | 'regression' | 'path_class'

/**
 * 单维度对 CNN_Fitness_Score 的贡献明细（后端 `ScoreContribution`）。
 *
 * 前端可据此渲染"为什么这只分高/分低"的解释，通常在展开行或 Tooltip 中呈现。
 */
export interface ScoreContribution {
  /** 维度名，如 "volatility" / "nonlinearity" / "data_quality" */
  dimension: string
  /** 该维度的原始指标值；数据不足（insufficient）时为 null */
  raw_value: number | null
  /** 该维度的等级判定，如 "high" / "medium" / "low"；无法判定时为 null */
  level: string | null
  /** 在 ScreeningRules.weights 中配置的权重；恒 >= 0 */
  weight: number
  /** 本维度对总分的贡献量 = weight × normalized_value（归一后） */
  contribution: number
  /** 该维度的置信度等级 */
  confidence: string
}

/**
 * 单标的 Tier-1 廉价预筛打分结果（后端 `Tier1Score`）。
 *
 * `available=false` 时 `fitness_score` 为 null，该行排在榜单末尾且不入围 Tier-2。
 */
export interface Tier1Score {
  /** 标的代码，如 "600030.SSE" */
  vt_symbol: string
  /** CNN 适配度综合分 [0,1]；available=false 或所有维度均 insufficient 时为 null */
  fitness_score: number | null
  /** 逐维贡献明细，用于前端详情展示 */
  contributions: ScoreContribution[]
  /** 综合置信度，不高于所有参与维度的最低置信度 */
  overall_confidence: string
  /** false 表示本地数据不可用，该行无可用分数 */
  available: boolean
  /** 不可用原因 / 置信度降级说明；无时为 null */
  note: string | null
}

/**
 * 单标的 Tier-2 WF/OOS 实证结论（后端 `Tier2Verdict`）。
 *
 * `evaluable=false` 时（无折或抛异常）`edge_ok`/`avg_score` 等字段无意义，
 * 前端应渲染为不可用态。
 */
export interface Tier2Verdict {
  /** 标的代码 */
  vt_symbol: string
  /** false 表示 Tier-2 失败或折数为 0，不可派生 edge 结论 */
  evaluable: boolean
  /** 绝对 edge 门禁结论：跨折平均 candidate_score > 0 且正折占比 >= 门槛；evaluable=false 时恒 false */
  edge_ok: boolean
  /** 跨折跨种子平均 candidate_score；evaluable=false 时为 null */
  avg_score: number | null
  /** candidate_score > 0 的折数占比；evaluable=false 时为 null */
  pos_fold_ratio: number | null
  /** 跨种子 candidate_score 标准差均值；无多种子时为 null */
  avg_cross_seed_std: number | null
  /** WF 报告 ID，可按此在 screening governance store 回读详情；null 表示无报告 */
  report_id: string | null
  /** 失败原因 / 不可评估说明；null 表示正常 */
  note: string | null
}

/**
 * 选股榜单中的一行记录（后端 `LeaderboardRow`）。
 *
 * 每行对应一只标的；`rank` 按 CNN_Fitness_Score 降序编号（从 1 开始）；
 * 未入围 Tier-2 的行 `tier2` 为 null。
 */
export interface LeaderboardRow {
  /** 榜单排名，从 1 开始（按 fitness_score 降序；available=false 的行排末尾） */
  rank: number
  /** Tier-1 打分结果 */
  tier1: Tier1Score
  /** 是否入围了 Tier-2 评估 */
  promoted_to_tier2: boolean
  /** Tier-2 实证结论；未入围或 run_tier2=false 时为 null */
  tier2: Tier2Verdict | null
}

/**
 * 一次 CNN 选股批量运行的完整产物（后端 `ScreeningResult`）。
 *
 * `status` 恒为 `"draft"`，前端应以视觉标注提示该结论为草稿，
 * 不自动开训、不下真实单。
 */
export interface ScreeningResult {
  /** 唯一运行 ID（UUID4） */
  run_id: string
  /** 恒为 "draft"；前端需以 Tag/Alert 明确提示草稿状态 */
  status: 'draft'
  /** 运行完成时间戳（ISO 串） */
  created_at: string
  /**
   * 输入回显（键值对），供复现与审计，常见键：
   * name / interval / as_of / lookback_days / exchange / top_k / run_tier2 / objective
   */
  input: Record<string, unknown>
  /** 本次所用 ScreeningRules 的标识/版本 */
  rules_id: string
  /** 过滤后进入 Tier-1 的候选标的数 */
  universe_size: number
  /** 被排除标的及原因列表（每项含 vt_symbol 与 reason 等） */
  excluded: Array<Record<string, unknown>>
  /** 选股榜单（按 fitness_score 降序） */
  leaderboard: LeaderboardRow[]
  /** Tier-1 实际数据右边界（ISO 串），用于审计无前视；无数据时为 null */
  effective_right_bound: string | null
  /**
   * Tier-2 评估区间；未跑 Tier-2 时为 null，含 start / end / objective 键
   */
  eval_window: Record<string, unknown> | null
}

/**
 * 启动 CNN 批量选股任务的请求体（后端 `CNNScreeningRequest`）。
 *
 * 字段含义与后端 Pydantic 模型一一对应，以下仅标注前端侧的注意事项。
 * `as_of` 采用 ISO 日期字符串（YYYY-MM-DD），后端按当天 00:00:00 UTC 处理。
 */
export interface CNNScreeningRequest {
  /** 本次选股任务名称，用于任务列表展示与产物归档 */
  name: string
  /** K 线周期，如 "d"（日线）、"30m"（30 分钟线） */
  interval: string
  /** 评估截止日期（YYYY-MM-DD），Tier-1/Tier-2 数据均不超过此时点（必填） */
  as_of: string
  /** Tier-1 画像回看天数；必须 > 0 */
  lookback_days: number
  /** 交易所过滤："SSE"/"SZSE"/"BSE" 取其一，null 或空串时不过滤 */
  exchange?: string | null
  /** 最小历史 bar 数；低于此值的标的被排除出候选池 */
  min_bar_count: number
  /** 显式候选池：非空时以此清单为 universe（仍做本地数据校验） */
  include_symbols?: string[]
  /** 强制排除清单：从最终 universe 中剔除（在 include_symbols 后生效） */
  exclude_symbols?: string[]
  /** Tier-1 后入围 Tier-2 的最大标的数 */
  top_k: number
  /** 是否执行 Tier-2 WF/OOS 实证；false 时只产出 Tier-1 榜单 */
  run_tier2: boolean
  /** CNN 训练目标，供 Tier-2 构造 CNNWalkForwardRequest 使用 */
  objective: ObjectiveType
  /**
   * Tier-2 标签配置；不传时后端用 ScreeningRules 默认（next_bar，与改造前等价）。
   * `objective='path_class'` 时必须传 `{ mode: 'oco', take_profit, stop_loss, max_hold }`
   * （止盈/止损为收益率小数，如 0.03=+3%），否则后端入口返回 400——路径四分类标签
   * 依赖 OCO 三重障碍判定。
   */
  label_spec?: LabelSpec
  /** 是否将 ScreeningResult 写入磁盘；false 时仅内存返回 */
  persist?: boolean

  // ── Tier-2 高级覆盖参数（全部可选；不填则后端使用 ScreeningRules 默认值）──
  /**
   * Tier-2 评估窗口总长（天数）；覆盖 ScreeningRules.eval_window_days（默认 900）。
   * 须满足 eval_window_days >= train_days + fold_test_days，否则无法生成折。
   */
  eval_window_days?: number
  /**
   * 每折训练窗口（天数）；覆盖 ScreeningRules.train_days（默认 480）。
   * 历史不足 train_days + fold_test_days 的标的会自动跳过 Tier-2。
   */
  train_days?: number
  /**
   * 单折测试集长度（天数）；覆盖 ScreeningRules.fold_test_days（默认 90）。
   * 同时影响步进粒度（step_days ≈ fold_test_days）。
   */
  fold_test_days?: number
  /**
   * 每折训练随机种子数；覆盖 ScreeningRules.n_seeds（默认 1）。
   * 取多个种子并求均值可降低偶然性，但线性增加算力消耗。
   */
  n_seeds?: number
}

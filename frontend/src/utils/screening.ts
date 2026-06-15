/**
 * CNN 选股页面指标元数据注册表：中英结合标签 + 悬浮解释。
 *
 * 作为单一权威来源，供榜单列头和贡献明细展开行统一使用。
 * 键名与后端 `dimension` 字段及 `Tier1Score`/`Tier2Verdict` 字段名一一对应。
 */
export interface MetricMeta {
  /** 中文主 + 英文辅助标签，形如 "数据质量 (Data Quality)"。 */
  label: string
  /** Tooltip 解释文案；直接面向用户，讲清楚指标的含义与使用场景。 */
  tooltip: string
}

/**
 * 指标键名到 {@link MetricMeta} 的映射表。
 *
 * 包含七个选股打分维度（出现在 `ScoreContribution.dimension` 中）以及
 * 榜单列字段（`fitness_score`、`overall_confidence` 等）。
 * 未注册的键由 {@link metricMeta} 函数提供兜底。
 *
 * @example
 * ```ts
 * METRIC_META['fitness_score'].label   // "CNN 适配度 (Fitness Score)"
 * METRIC_META['data_quality'].tooltip  // "窗口内有效 K 线数量…"
 * ```
 */
export const METRIC_META: Record<string, MetricMeta> = {
  // ── 七大选股打分维度（contributions breakdown + 可能作为列） ────────────────
  data_quality: {
    label: '数据质量 (Data Quality)',
    tooltip:
      '窗口内有效 K 线数量相对该周期最低训练样本量是否充足。数据越足越适合训练 CNN。',
  },
  liquidity: {
    label: '流动性 (Liquidity)',
    tooltip:
      '日均成交额等级。流动性越高，预测越可交易、滑点与成本越低。',
  },
  volatility: {
    label: '波动性 (Volatility)',
    tooltip:
      '已实现波动率等级。适度波动才有可捕捉的行情空间，止盈止损才有意义。',
  },
  predictability: {
    label: '可预测性 (Predictability)',
    tooltip:
      '趋势/均值回复结构判定。有明确结构（非随机游走）才值得建模。',
  },
  nonlinearity: {
    label: '非线性 (Nonlinearity)',
    tooltip:
      '线性模型残差中残留的可学习结构。越高表示存在线性模型抓不住、正是 CNN 能发挥的非线性信号。',
  },
  pattern_recurrence: {
    label: '形态复现 (Pattern Recurrence)',
    tooltip:
      '价格序列中平移不变的局部形态的重复程度。越高表示 CNN 卷积核能反复识别的图形越多。',
  },
  temporal_stability: {
    label: '时间稳定 (Temporal Stability)',
    tooltip:
      '前后子窗统计画像的稳定度。越高表示样本内学到的形态在样本外越可能仍成立。',
  },

  // ── 榜单列字段 ────────────────────────────────────────────────────────────
  fitness_score: {
    label: 'CNN 适配度 (Fitness Score)',
    tooltip:
      'Tier-1 综合分 ∈ [0,1]，由画像四块 + CNN 代理指标按权重合成。仅为高召回预筛排名信号，不代表一定适合 CNN——最终以 Tier-2 实证为准。',
  },
  overall_confidence: {
    label: '综合置信度 (Confidence)',
    tooltip:
      '不高于参与维度的最低置信度。样本不足时降级，提示该分数的可靠性。',
  },
  promoted_to_tier2: {
    label: '入围 Tier-2 (Promoted)',
    tooltip:
      '是否按综合分排名进入 WF/OOS 实证验证。',
  },
  edge_ok: {
    label: '实证胜出 (Edge OK)',
    tooltip:
      'Tier-2 绝对 edge 门禁结论：跨折平均核心分 > 0 且正分折占比 ≥ 阈值。这才是 CNN 在该股样本外真能跑动的判据。',
  },
  avg_score: {
    label: '跨折均分 (Avg Fold Score)',
    tooltip: 'WF 各折候选核心分（跨种子均值）的平均。',
  },
  pos_fold_ratio: {
    label: '正分折占比 (Positive Fold Ratio)',
    tooltip: 'candidate_score > 0 的折数占比。',
  },
  avg_cross_seed_std: {
    label: '跨种子波动 (Cross-seed Std)',
    tooltip:
      '多种子得分标准差均值，反映结果对随机种子的敏感度。越小越稳。',
  },
  rank: {
    label: '排名 (Rank)',
    tooltip: '按 CNN 适配度（Fitness Score）降序编号，从 1 开始。',
  },
  vt_symbol: {
    label: '标的 (Symbol)',
    tooltip: '标的合约代码，如 "600030.SSE"。',
  },

  // ── 贡献明细子列 ─────────────────────────────────────────────────────────
  dimension: {
    label: '维度 (Dimension)',
    tooltip: '参与 CNN 适配度合成的评分维度名称；各维度含义可在对应行悬停查看。',
  },
  raw_value: {
    label: '原始值 (Raw Value)',
    tooltip: '该维度的原始指标计算值；数据不足时为空。',
  },
  level: {
    label: '等级 (Level)',
    tooltip: '指标经区间映射后的等级判定，如 high / medium / low。',
  },
  weight: {
    label: '权重 (Weight)',
    tooltip: '该维度在 ScreeningRules 中配置的合成权重，恒 ≥ 0。',
  },
  contribution: {
    label: '贡献量 (Contribution)',
    tooltip: '本维度对 Fitness Score 总分的贡献量 = weight × 归一化值。',
  },
  confidence: {
    label: '置信度 (Confidence)',
    tooltip: '该维度独立的置信度等级，样本不足时降级为 insufficient。',
  },
}

/**
 * 按指标键名查找 {@link MetricMeta}；未注册时返回以键名为标签的兜底对象。
 *
 * @param key - 指标键名（后端 `dimension` 字段值或列字段名）。
 * @returns 对应的 {@link MetricMeta}；未注册时返回兜底对象。
 *
 * @example
 * ```ts
 * metricMeta('fitness_score').label  // "CNN 适配度 (Fitness Score)"
 * metricMeta('unknown_key').label    // "unknown_key"
 * ```
 */
export function metricMeta(key: string): MetricMeta {
  return METRIC_META[key] ?? {
    label: key,
    tooltip: '后端返回的选股指标，当前版本尚未配置更详细说明。',
  }
}

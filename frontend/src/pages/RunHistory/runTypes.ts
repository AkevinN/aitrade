/**
 * 「运行历史」的任务类型分类与映射。
 *
 * 回测/选股的历史结果都持久化在任务历史里（见后端 TaskHistoryStore），本特性据
 * task.type 把运行归类为"回测"或"选股"，用于列表的类别列与详情的渲染路由。
 * 取值与后端 `models/alpha.py:TaskType` 一一对应。
 */

/** 归类为「回测」的任务类型（CNN/Alpha/规则/方案回测 + 治理回放）。 */
export const BACKTEST_RUN_TYPES = [
  'cnn_backtest',
  'backtest_run',
  'strategy_backtest',
  'scheme_backtest',
  'cnn_governance_replay',
] as const

/** 归类为「选股」的任务类型。 */
export const SCREENING_RUN_TYPES = ['cnn_screening'] as const

/** 回测 ∪ 选股的全部运行类型，供"全部"过滤一次性拉取。 */
export const ALL_RUN_TYPES: string[] = [...BACKTEST_RUN_TYPES, ...SCREENING_RUN_TYPES]

/** 运行类别。 */
export type RunCategory = 'backtest' | 'screening' | 'other'

/**
 * 把任务类型映射为运行类别。
 *
 * @param type - 任务类型字符串（task.type）
 * @returns 'screening' / 'backtest' / 'other'
 */
export function runCategory(type: string): RunCategory {
  if ((SCREENING_RUN_TYPES as readonly string[]).includes(type)) return 'screening'
  if ((BACKTEST_RUN_TYPES as readonly string[]).includes(type)) return 'backtest'
  return 'other'
}

/** 类别中文标签。 */
export function runCategoryLabel(category: RunCategory): string {
  return category === 'screening' ? '选股' : category === 'backtest' ? '回测' : '其他'
}

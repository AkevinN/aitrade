import api from './client'
import type {
  LiveDecisionRequest,
  Decision,
  DecisionTrace,
  TradingPlan,
  TradingPlanRequest,
  TradingPlanSummary,
  SchedulerStatus,
  SchedulerRunEvent,
  RebalanceDecision,
  PortfolioState,
  PortfolioRiskState,
} from '../types/live'
import type { TaskStartResponse } from '../types/alpha'

/**
 * 实盘操作台 API 服务（对应后端 `/api/live/*` 路由组）。
 *
 * 涵盖 CNN 决策、交易计划自动化、调度器、规则调仓与组合账本五大功能域。
 * 所有写操作均返回异步任务或直接结果，读操作直接返回数据。
 */
export const liveService = {
  /**
   * 触发一次今日 CNN 决策（异步任务）。
   *
   * @param req - 决策请求体，包含模型/标的/方案/组合快照/风控配置等
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   */
  startDecision: (req: LiveDecisionRequest) =>
    api.post<TaskStartResponse>('/api/live/decision', req).then((r) => r.data),

  /**
   * 列出已持久化决策的 signal_id 集合。
   *
   * @returns signal_id 字符串数组，可进一步用 `getDecision` 拉取详情
   */
  listDecisions: () =>
    api
      .get<{ signal_ids: string[] }>('/api/live/decisions')
      .then((r) => r.data.signal_ids),

  /**
   * 按 signal_id 返回单条决策详情。
   *
   * @param signalId - 幂等键，如 "2026-06-08:eod_buy_v1:model@v3"
   * @returns 决策对象；不存在时后端返回 404
   */
  getDecision: (signalId: string) =>
    api
      .get<Decision>(`/api/live/decisions/${signalId}`)
      .then((r) => r.data),

  /**
   * 按 signal_id 返回完整决策过程档案（六段 Decision_Trace）。
   *
   * 不存在时后端返回 404，本函数不吞掉 404，由调用方（DecisionTracePanel 的 useQuery）
   * 感知并显示「暂无过程档案」。
   *
   * @param signalId - 与 Decision 一一对应的幂等键
   * @returns 六段决策过程档案
   */
  getDecisionTrace: (signalId: string) =>
    api
      .get<DecisionTrace>(`/api/live/decisions/${signalId}/trace`)
      .then((r) => r.data),

  /**
   * 归档式删除单条决策及其过程档案。
   *
   * 文件移入 `archive/`，解除幂等占位；同一 Decision_Bar 可重新产出决策与提醒。
   *
   * @param signalId - 要归档删除的决策幂等键
   * @returns 操作结果（含 deleted / trace_archived 标记）
   */
  deleteDecision: (signalId: string) =>
    api
      .delete<{ signal_id: string; deleted: boolean; trace_archived: boolean }>(
        `/api/live/decisions/${signalId}`,
      )
      .then((r) => r.data),

  /**
   * 批量归档式删除（部分成功语义）。
   *
   * 存在的决策归档，缺失的归入 `missing` 数组返回，不整体报错。
   *
   * @param signalIds - 要归档删除的 signal_id 列表
   * @returns `{ deleted: string[]; missing: string[] }`
   */
  batchDeleteDecisions: (signalIds: string[]) =>
    api
      .post<{ deleted: string[]; missing: string[] }>(
        '/api/live/decisions/batch-delete',
        { signal_ids: signalIds },
      )
      .then((r) => r.data),

  // ===== 交易计划自动化（Trading Plan Automation）=====

  /**
   * 列出所有交易计划摘要（含 strategy_type / portfolio_id / signal_source 等 v2 字段）。
   *
   * @returns 计划摘要列表，用于 PlanList 渲染与 Portfolio 页面的组合选择器
   */
  listPlans: () =>
    api.get<TradingPlanSummary[]>('/api/live/plans').then((r) => r.data),

  /**
   * 按 plan_id 返回计划完整内容（含所有字段，用于编辑回填）。
   *
   * @param planId - 计划唯一 ID
   * @returns 计划完整对象
   */
  getPlan: (planId: string) =>
    api.get<TradingPlan>(`/api/live/plans/${planId}`).then((r) => r.data),

  /**
   * 创建交易计划。
   *
   * @param req - 计划请求体（name / model / vt_symbol 等必填，rule 模式另需 signal_source 等）
   * @returns 创建后的计划完整对象（含 plan_id / created_at）
   */
  createPlan: (req: TradingPlanRequest) =>
    api.post<TradingPlan>('/api/live/plans', req).then((r) => r.data),

  /**
   * 更新交易计划（全量替换）。
   *
   * @param planId - 要更新的计划 ID
   * @param req - 更新后的完整请求体
   * @returns 更新后的计划完整对象
   */
  updatePlan: (planId: string, req: TradingPlanRequest) =>
    api.put<TradingPlan>(`/api/live/plans/${planId}`, req).then((r) => r.data),

  /**
   * 删除交易计划。
   *
   * @param planId - 要删除的计划 ID
   * @returns 操作结果（含 deleted 标记）
   */
  deletePlan: (planId: string) =>
    api
      .delete<{ plan_id: string; deleted: boolean }>(`/api/live/plans/${planId}`)
      .then((r) => r.data),

  /**
   * 启用或停用交易计划自动调度。
   *
   * @param planId - 计划 ID
   * @param enabled - true=启用；false=停用
   * @returns 更新后的启用状态
   */
  togglePlan: (planId: string, enabled: boolean) =>
    api
      .patch<{ plan_id: string; enabled: boolean }>(
        `/api/live/plans/${planId}/enabled`,
        { enabled },
      )
      .then((r) => r.data),

  /**
   * 按计划立即触发一次今日决策（异步任务）。
   *
   * 对于 strategy_type=rule 计划，触发规则调仓；对于 cnn 计划，触发 CNN 决策。
   *
   * @param planId - 要触发的计划 ID
   * @returns 异步任务启动凭据（含 task_id）
   */
  runPlan: (planId: string) =>
    api
      .post<TaskStartResponse>(`/api/live/plans/${planId}/run`)
      .then((r) => r.data),

  /**
   * 查询调度器当前运行状态（含启用计划数与上次触发日期映射）。
   *
   * @returns 调度器状态快照；调用方通常以 5 秒 refetchInterval 轮询
   */
  getSchedulerStatus: () =>
    api.get<SchedulerStatus>('/api/live/scheduler/status').then((r) => r.data),

  /**
   * 查询调度运行日志（只读，默认当日，倒序）。
   *
   * @param params - 可选过滤参数：plan_id（按计划过滤）、date（YYYY-MM-DD）、limit（条数上限）
   * @returns 当日调度事件列表（trigger / skip / error）
   */
  getSchedulerRuns: (params?: { plan_id?: string; date?: string; limit?: number }) =>
    api
      .get<SchedulerRunEvent[]>('/api/live/scheduler/runs', { params })
      .then((r) => r.data),

  // ===== 规则调仓（Rebalance）=====

  /**
   * 触发一次规则调仓决策（引用 rule 类型计划），返回异步任务。
   *
   * 终态 result 为 `RebalanceResult`（含 decision / risk / idempotent_hit）。
   *
   * @param planId - rule 类型计划 ID
   * @returns 异步任务启动凭据（含 task_id）
   */
  startRebalance: (planId: string) =>
    api
      .post<TaskStartResponse>('/api/live/rebalance', { plan_id: planId })
      .then((r) => r.data),

  /**
   * 列出所有调仓决策摘要（仅 signal_id / status / portfolio_id / created_at）。
   *
   * @returns 摘要列表，Portfolio 页面用此过滤当前组合的历史调仓
   */
  listRebalances: () =>
    api
      .get<{ items: Array<{ signal_id: string; status: string; portfolio_id: string; created_at: string }> }>(
        '/api/live/rebalances',
      )
      .then((r) => r.data.items),

  /**
   * 按 signal_id 获取调仓决策完整内容（含调仓明细、风控摘要、目标组合）。
   *
   * @param signalId - 调仓决策的幂等键
   * @returns 调仓决策完整对象
   */
  getRebalance: (signalId: string) =>
    api
      .get<RebalanceDecision>(`/api/live/rebalances/${signalId}`)
      .then((r) => r.data),

  /**
   * 确认调仓执行完毕，并回填账本持仓记录。
   *
   * 调用方需在人工按清单执行完后再确认；幂等约束：已确认或卖超时后端返回 409。
   *
   * @param signalId - 要确认的调仓决策幂等键
   * @returns 更新后的决策对象与账本持仓快照
   */
  confirmRebalance: (signalId: string) =>
    api
      .post<{ decision: RebalanceDecision; portfolio: PortfolioState }>(
        `/api/live/rebalances/${signalId}/confirm`,
      )
      .then((r) => r.data),

  /**
   * 获取指定组合的持仓账本快照。
   *
   * @param portfolioId - 组合唯一 ID（如 "portfolio-001"）
   * @returns 持仓账本，含 positions（{vt_symbol: 股数}）与 cash 余额
   */
  getPortfolio: (portfolioId: string) =>
    api
      .get<PortfolioState>(`/api/live/portfolios/${portfolioId}`)
      .then((r) => r.data),

  /**
   * 获取指定组合的风险状态（含熔断信息）。
   *
   * @param portfolioId - 组合唯一 ID
   * @returns 风险状态，broken=true 表示已触发熔断
   */
  getPortfolioRisk: (portfolioId: string) =>
    api
      .get<PortfolioRiskState>(`/api/live/portfolio-risk/${portfolioId}`)
      .then((r) => r.data),

  /**
   * 复位组合熔断状态（人工处置完毕后调用）。
   *
   * 复位后 broken=false、broken_date=null、reason=null，峰值净值保留。
   *
   * @param portfolioId - 要复位的组合 ID
   * @returns 复位后的风险状态
   */
  resetPortfolioRisk: (portfolioId: string) =>
    api
      .post<PortfolioRiskState>(`/api/live/portfolio-risk/${portfolioId}/reset`)
      .then((r) => r.data),
}

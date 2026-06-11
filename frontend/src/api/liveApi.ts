import api from './client'
import type {
  LiveDecisionRequest,
  Decision,
  DecisionTrace,
  TradingPlan,
  TradingPlanRequest,
  TradingPlanSummary,
  SchedulerStatus,
} from '../types/live'
import type { TaskStartResponse } from '../types/alpha'

export const liveService = {
  // 触发一次今日决策（异步任务，返回 task_id）
  startDecision: (req: LiveDecisionRequest) =>
    api.post<TaskStartResponse>('/api/live/decision', req).then((r) => r.data),

  // 列出已持久化决策的 signal_id 集合
  listDecisions: () =>
    api
      .get<{ signal_ids: string[] }>('/api/live/decisions')
      .then((r) => r.data.signal_ids),

  // 按 signal_id 返回单条决策详情
  getDecision: (signalId: string) =>
    api
      .get<Decision>(`/api/live/decisions/${signalId}`)
      .then((r) => r.data),

  // 按 signal_id 返回完整决策过程档案（六段 Decision_Trace）。
  // 注意：不存在时后端返回 404，本函数不吞掉 404，由调用方（DecisionTracePanel 的 useQuery）
  // 感知并显示「暂无过程档案」。
  getDecisionTrace: (signalId: string) =>
    api
      .get<DecisionTrace>(`/api/live/decisions/${signalId}/trace`)
      .then((r) => r.data),

  // 归档式删除单条决策及其过程档案（文件移入 archive/，解除幂等占位，
  // 同一 Decision_Bar 可重新产出决策与提醒）。
  deleteDecision: (signalId: string) =>
    api
      .delete<{ signal_id: string; deleted: boolean; trace_archived: boolean }>(
        `/api/live/decisions/${signalId}`,
      )
      .then((r) => r.data),

  // ===== 交易计划自动化（Trading Plan Automation）=====

  // 列出所有交易计划摘要
  listPlans: () =>
    api.get<TradingPlanSummary[]>('/api/live/plans').then((r) => r.data),

  // 按 plan_id 返回计划完整内容
  getPlan: (planId: string) =>
    api.get<TradingPlan>(`/api/live/plans/${planId}`).then((r) => r.data),

  // 创建计划
  createPlan: (req: TradingPlanRequest) =>
    api.post<TradingPlan>('/api/live/plans', req).then((r) => r.data),

  // 更新计划
  updatePlan: (planId: string, req: TradingPlanRequest) =>
    api.put<TradingPlan>(`/api/live/plans/${planId}`, req).then((r) => r.data),

  // 删除计划
  deletePlan: (planId: string) =>
    api
      .delete<{ plan_id: string; deleted: boolean }>(`/api/live/plans/${planId}`)
      .then((r) => r.data),

  // 启用/停用计划
  togglePlan: (planId: string, enabled: boolean) =>
    api
      .patch<{ plan_id: string; enabled: boolean }>(
        `/api/live/plans/${planId}/enabled`,
        { enabled },
      )
      .then((r) => r.data),

  // 按计划立即触发一次今日决策（异步任务，返回 task_id）
  runPlan: (planId: string) =>
    api
      .post<TaskStartResponse>(`/api/live/plans/${planId}/run`)
      .then((r) => r.data),

  // 调度器运行状态
  getSchedulerStatus: () =>
    api.get<SchedulerStatus>('/api/live/scheduler/status').then((r) => r.data),
}

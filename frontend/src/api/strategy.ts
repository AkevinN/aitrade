// 规则策略回测 API 服务 — 仿 api/liveApi.ts 风格
import api from './client'
import type { SignalSourceInfo, StrategyBacktestRequest, StrategySweepRequest, StrategyWalkForwardRequest } from '../types/strategy'
import type { TaskStartResponse } from '../types/alpha'

/**
 * 规则策略回测 API 服务（对应后端 `/api/strategy/*` 路由组）。
 *
 * 所有写操作均返回异步任务，需通过 `useTask(task_id)` 轮询至终态。
 */
export const strategyService = {
  /**
   * 列出系统已注册的信号源。
   *
   * 结果用于填充信号源 Select，以及动态渲染信号参数表单（param_spec）。
   *
   * @returns 信号源元信息列表
   */
  listSources: (): Promise<SignalSourceInfo[]> =>
    api.get<SignalSourceInfo[]>('/api/strategy/sources').then((r) => r.data),

  /**
   * 启动单次规则策略回测任务。
   *
   * @param req - 回测请求体，包含信号源、标的池、策略参数、日期范围与成本假设
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   */
  runBacktest: (req: StrategyBacktestRequest): Promise<TaskStartResponse> =>
    api.post<TaskStartResponse>('/api/strategy/backtest/run', req).then((r) => r.data),

  /**
   * 启动参数网格扫描任务。
   *
   * 对 `grid` 中每个参数组合各跑一次回测，终态 `result.rows` 为 `SweepRow[]`。
   *
   * @param req - 扫描请求体（含基础回测配置 + grid 网格）
   * @returns 异步任务启动凭据
   */
  runSweep: (req: StrategySweepRequest): Promise<TaskStartResponse> =>
    api.post<TaskStartResponse>('/api/strategy/sweep/run', req).then((r) => r.data),

  /**
   * 启动 Walk-Forward 验证任务。
   *
   * 以滚动窗口评估策略参数的时序稳定性，终态 `result.rows` 为各折 `SweepRow[]`。
   *
   * @param req - Walk-Forward 请求体（含 train_days / test_days）
   * @returns 异步任务启动凭据
   */
  runWalkForward: (req: StrategyWalkForwardRequest): Promise<TaskStartResponse> =>
    api.post<TaskStartResponse>('/api/strategy/walkforward/run', req).then((r) => r.data),
}

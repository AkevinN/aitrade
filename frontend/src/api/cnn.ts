/**
 * CNN 模块 API 服务对象——封装所有与后端 `/api/cnn/` 端点的 HTTP 交互。
 *
 * 提供 CNN 模型的训练、查询、删除、架构探查、推理（predict）及回测能力。
 * 所有方法均返回 `Promise<T>`，错误由全局 Axios 拦截器透传。
 */
import api from './client'
import type {
  CNNArchitecture,
  CNNModelDetail,
  CNNModelInfo,
  CNNStatus,
  CNNTrainRequest,
  CNNBacktestRequest,
  CNNPredictRequest,
  TaskStartResponse,
} from '../types/cnn'
import type { CNNScreeningRequest } from '../types/screening'

export const cnnService = {
  /**
   * 查询 CNN 模块运行状态。
   *
   * 用于在训练/推理前探测后端是否具备 torch 环境与可用推理设备。
   *
   * @returns torch 安装情况与当前推理设备（`cpu` / `cuda`）
   */
  getStatus: () =>
    api.get<CNNStatus>('/api/cnn/status').then((r) => r.data),

  /**
   * 启动 CNN 模型训练任务。
   *
   * @param req - 训练请求体（标的、数据种类与间隔、标签规格、网络与训练超参等）
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   */
  train: (req: CNNTrainRequest) =>
    api.post<TaskStartResponse>('/api/cnn/train', req).then((r) => r.data),

  /**
   * 列出本地已保存的全部 CNN 模型。
   *
   * @returns 模型列表概要（名称、创建时间、最佳 epoch、预测目标等元信息）
   */
  listModels: () =>
    api.get<CNNModelInfo[]>('/api/cnn/models').then((r) => r.data),

  /**
   * 获取单个模型的完整详情。
   *
   * @param name - 模型名称（与 `listModels` 返回的 `name` 一致）
   * @returns 训练/网络配置、归一化参数、数据集信息及逐 epoch 训练历史
   */
  getModel: (name: string) =>
    api.get<CNNModelDetail>(`/api/cnn/models/${name}`).then((r) => r.data),

  /**
   * 探查模型的真实网络结构。
   *
   * 后端会重建模型实例并加载权重，做一次前向计算以捕获逐层输出张量形状；
   * 返回结果含 `verified` 真实性闸门，标记权重与重建结构是否严格一致。
   *
   * @param name - 模型名称
   * @returns 逐层结构、参数量、输入/输出张量形状及真实性校验结果
   */
  getModelArchitecture: (name: string) =>
    api.get<CNNArchitecture>(`/api/cnn/models/${name}/architecture`).then((r) => r.data),

  /**
   * 删除指定的本地模型。
   *
   * @param name - 待删除模型名称
   * @returns `{ deleted }` 为被删除模型的名称
   */
  deleteModel: (name: string) =>
    api.delete<{ deleted: string }>(`/api/cnn/models/${name}`).then((r) => r.data),

  /**
   * 启动 CNN 推理任务：对指定区间逐日生成概率信号并写入信号库。
   *
   * @param req - 推理请求体（信号名 name、模型名 model、起止日期 start/end）
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   */
  predict: (req: CNNPredictRequest) =>
    api.post<TaskStartResponse>('/api/cnn/predict', req).then((r) => r.data),

  /**
   * 启动 CNN 模型回测任务。
   *
   * 按请求中的出场模式（概率阈值 / 固定持有 / OCO 止盈止损 / 按 label 自动对齐）
   * 与成本假设（佣金、印花税、滑点、T+1）模拟成交。
   *
   * @param req - 回测请求体（模型、资金、起止日期、买卖阈值、出场模式与成本参数等）
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   */
  runBacktest: (req: CNNBacktestRequest) =>
    api.post<TaskStartResponse>('/api/cnn/backtest/run', req).then((r) => r.data),

  /**
   * 启动 CNN 批量选股任务（分层漏斗 Tier-1 廉价预筛 + 可选 Tier-2 WF/OOS 实证）。
   *
   * 后端异步执行：Tier-1 对 universe 批量打 CNN 适配度综合分，
   * Tier-2 对排名靠前的 top_k 只运行 WF/OOS 实证（`req.run_tier2=true` 时）。
   * 返回 task_id 后需通过 `useTask` 轮询至 completed，再从 `task.data.result`
   * 中读取 `ScreeningResult`。
   *
   * @param req - 选股请求体（标的池过滤、截止时间、漏斗配置与 Tier-2 超参）
   * @returns 异步任务启动凭据（含 task_id，需轮询至 completed）
   *
   * @example
   * ```ts
   * const res = await cnnService.runScreening({ name: 'screen_20250601', interval: 'd', as_of: '2025-06-01', lookback_days: 250, top_k: 15, run_tier2: true, min_bar_count: 250, objective: 'classification' })
   * setTaskId(res.task_id)
   * ```
   */
  runScreening: (req: CNNScreeningRequest) =>
    api.post<TaskStartResponse>('/api/cnn/screening/batch', req).then((r) => r.data),
}

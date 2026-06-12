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

export const cnnService = {
  // CNN status
  getStatus: () =>
    api.get<CNNStatus>('/api/cnn/status').then((r) => r.data),

  // Train
  train: (req: CNNTrainRequest) =>
    api.post<TaskStartResponse>('/api/cnn/train', req).then((r) => r.data),

  // Models
  listModels: () =>
    api.get<CNNModelInfo[]>('/api/cnn/models').then((r) => r.data),

  getModel: (name: string) =>
    api.get<CNNModelDetail>(`/api/cnn/models/${name}`).then((r) => r.data),

  // 真实网络结构（重建实例 + 加载权重 + 逐层形状）
  getModelArchitecture: (name: string) =>
    api.get<CNNArchitecture>(`/api/cnn/models/${name}/architecture`).then((r) => r.data),

  deleteModel: (name: string) =>
    api.delete<{ deleted: string }>(`/api/cnn/models/${name}`).then((r) => r.data),

  // 推理：生成概率信号并保存到信号库
  predict: (req: CNNPredictRequest) =>
    api.post<TaskStartResponse>('/api/cnn/predict', req).then((r) => r.data),

  // 回测
  runBacktest: (req: CNNBacktestRequest) =>
    api.post<TaskStartResponse>('/api/cnn/backtest/run', req).then((r) => r.data),
}

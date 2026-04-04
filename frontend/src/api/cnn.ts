import api from './client'
import type {
  CNNModelInfo,
  CNNStatus,
  CNNTrainRequest,
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
    api.get<CNNModelInfo>(`/api/cnn/models/${name}`).then((r) => r.data),

  deleteModel: (name: string) =>
    api.delete<{ deleted: string }>(`/api/cnn/models/${name}`).then((r) => r.data),
}

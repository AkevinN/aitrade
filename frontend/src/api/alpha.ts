import api from './client'
import type {
  AlphaStatus,
  Task,
  TaskStartResponse,
  DatasetDetail,
  BarDataList,
  BarDataDetail,
  BarDataDetailQuery,
  ModelDetail,
  SignalDetail,
  DataDownloadRequest,
  DatasetCreateRequest,
  ModelTrainRequest,
  SignalGenerateRequest,
  BacktestRunRequest,
  CNNTrainRequest,
  CsvPreviewResult,
  CsvImportResponse,
} from '../types/alpha'

export const alphaService = {
  // Module status
  getStatus: () =>
    api.get<AlphaStatus>('/api/alpha/status').then((r) => r.data),

  // Tasks
  listTasks: () =>
    api.get<Task[]>('/api/alpha/tasks').then((r) => r.data),

  getTask: (taskId: string) =>
    api.get<Task>(`/api/alpha/tasks/${taskId}`).then((r) => r.data),

  // Datasets
  listDatasets: () =>
    api.get<string[]>('/api/alpha/datasets').then((r) => r.data),

  getDataset: (name: string) =>
    api.get<DatasetDetail>(`/api/alpha/datasets/${name}`).then((r) => r.data),

  createDataset: (req: DatasetCreateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/datasets/create', req).then((r) => r.data),

  deleteDataset: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/datasets/${name}`).then((r) => r.data),

  // Models
  listModels: () =>
    api.get<string[]>('/api/alpha/models').then((r) => r.data),

  getModel: (name: string) =>
    api.get<ModelDetail>(`/api/alpha/models/${name}`).then((r) => r.data),

  trainModel: (req: ModelTrainRequest) =>
    api.post<TaskStartResponse>('/api/alpha/models/train', req).then((r) => r.data),

  deleteModel: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/models/${name}`).then((r) => r.data),

  // Signals
  listSignals: () =>
    api.get<string[]>('/api/alpha/signals').then((r) => r.data),

  getSignal: (name: string) =>
    api.get<SignalDetail>(`/api/alpha/signals/${name}`).then((r) => r.data),

  generateSignal: (req: SignalGenerateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/signals/generate', req).then((r) => r.data),

  deleteSignal: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/signals/${name}`).then((r) => r.data),

  // Backtest
  runBacktest: (req: BacktestRunRequest) =>
    api.post<TaskStartResponse>('/api/alpha/backtest/run', req).then((r) => r.data),

  // Data download
  downloadData: (req: DataDownloadRequest) =>
    api.post<TaskStartResponse>('/api/alpha/data/download', req).then((r) => r.data),

  // Contracts
  getContracts: () =>
    api.get<Record<string, unknown>>('/api/alpha/contracts').then((r) => r.data),

  // Bar data
  getBarData: () =>
    api.get<BarDataList>('/api/alpha/bar-data').then((r) => r.data),

  getBarDataDetail: (interval: string, vtSymbol: string, query?: BarDataDetailQuery) =>
    api
      .get<BarDataDetail>(`/api/alpha/bar-data/${interval}/${vtSymbol}`, {
        params: query as Record<string, string>,
      })
      .then((r) => r.data),

  deleteBarData: (interval: string, vtSymbol: string) =>
    api
      .delete<{ success: boolean; message: string }>(
        `/api/alpha/bar-data/${interval}/${vtSymbol}`,
      )
      .then((r) => r.data),

  // CSV Import
  previewCsvImport: (file: File, field_mapping?: Record<string, string>) => {
    const formData = new FormData()
    formData.append('file', file)
    if (field_mapping) {
      formData.append('field_mapping', JSON.stringify(field_mapping))
    }
    return api
      .post<CsvPreviewResult>('/api/alpha/bar-data/import/preview', formData)
      .then((r) => r.data)
  },

  importCsvData: (
    file: File,
    interval: 'd' | 'm',
    import_mode: 'merge' | 'replace',
    field_mapping?: Record<string, string>,
  ) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('interval', interval)
    formData.append('import_mode', import_mode)
    if (field_mapping) {
      formData.append('field_mapping', JSON.stringify(field_mapping))
    }
    return api
      .post<CsvImportResponse>('/api/alpha/bar-data/import', formData)
      .then((r) => r.data)
  },
}

/**
 * Alpha 模块 API 服务对象——封装所有与后端 `/api/alpha/` 端点的 HTTP 交互。
 *
 * 方法命名遵循 CRUD 惯例（`list*` / `get*` / `create*` / `delete*`），
 * 所有方法均返回 `Promise<T>`，T 为对应后端响应 DTO。
 *
 * @remarks
 * 文件上传（CSV 导入）使用 `FormData`，其余请求均为 JSON。
 * 错误由全局 Axios 拦截器记录并透传，调用方捕获 `AxiosError` 即可。
 */
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
  DataAggregateRequest,
  DatasetCreateRequest,
  ModelTrainRequest,
  SignalGenerateRequest,
  BacktestRunRequest,
  CsvPreviewResult,
  CsvImportResponse,
  CsvSaveMode,
  DataResourceList,
  DataResourceDetail,
  DataResourceMergeRequest,
  DataResourceMergePreview,
  DataResourceMergeResponse,
  DataResourceKind,
  RelocateBarIntervalResponse,
  CsvInterval,
  SymbolProfileRequest,
  SymbolProfileResponse,
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

  aggregateData: (req: DataAggregateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/data/aggregate', req).then((r) => r.data),

  // Symbol profiling
  runProfiling: (req: SymbolProfileRequest) =>
    api.post<SymbolProfileResponse>('/api/alpha/profiling', req).then((r) => r.data),

  listProfilingArtifacts: () =>
    api.get<string[]>('/api/alpha/profiling/artifacts').then((r) => r.data),

  getProfilingArtifact: (artifactId: string) =>
    api
      .get<SymbolProfileResponse>(`/api/alpha/profiling/${encodeURIComponent(artifactId)}`)
      .then((r) => r.data),

  // Contracts
  getContracts: () =>
    api.get<Record<string, unknown>>('/api/alpha/contracts').then((r) => r.data),

  // Unified data resources
  getDataResources: () =>
    api.get<DataResourceList>('/api/alpha/data/resources').then((r) => r.data),

  getDataResourceDetail: (kind: DataResourceKind, key: string, query?: BarDataDetailQuery) =>
    api
      .get<DataResourceDetail>(`/api/alpha/data/resources/${kind}/${encodeURIComponent(key)}`, {
        params: query as Record<string, string>,
      })
      .then((r) => r.data),

  deleteDataResource: (kind: DataResourceKind, key: string) =>
    api
      .delete<{ success: boolean; message: string }>(`/api/alpha/data/resources/${kind}/${encodeURIComponent(key)}`)
      .then((r) => r.data),

  relocateRawBarInterval: (key: string, interval: CsvInterval) =>
    api
      .patch<RelocateBarIntervalResponse>(
        `/api/alpha/data/resources/raw_bar/${encodeURIComponent(key)}/interval`,
        { interval },
      )
      .then((r) => r.data),

  previewDataResourceMerge: (req: DataResourceMergeRequest) =>
    api
      .post<DataResourceMergePreview>('/api/alpha/data/resources/merge/preview', req)
      .then((r) => r.data),

  mergeDataResourceBatches: (req: DataResourceMergeRequest) =>
    api
      .post<DataResourceMergeResponse>('/api/alpha/data/resources/merge', req)
      .then((r) => r.data),

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
  /**
   * 预览 K 线 CSV 文件的导入结果（列映射、样本行、符号与日期范围）。
   *
   * @param file - 待预览的 CSV 文件。
   * @param field_mapping - 自定义列名映射（{csv列名: 标准列名}）；缺省时后端自动推断。
   * @returns 预览结果，含匹配列、缺失列、样本行与符号列表。
   */
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

  /**
   * 正式导入 K 线 CSV 文件到本地 K 线库。
   *
   * @param file - 待导入的 CSV 文件。
   * @param interval - 目标周期（`d` 日线；`1m` / `5m` / `15m` / `30m` / `60m` 分钟线）。
   * @param import_mode - `merge` 合并（保留已有数据）；`replace` 全量覆盖。
   * @param save_mode - `batch` 暂存批次（默认）；`official` 直接写入官方库。
   * @param field_mapping - 自定义列名映射；缺省时后端自动推断。
   * @returns 导入结果，含成功/跳过/错误计数。
   */
  importCsvData: (
    file: File,
    interval: 'd' | '1m' | '5m' | '15m' | '30m' | '60m',
    import_mode: 'merge' | 'replace',
    save_mode: CsvSaveMode = 'batch',
    field_mapping?: Record<string, string>,
  ) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('interval', interval)
    formData.append('import_mode', import_mode)
    formData.append('save_mode', save_mode)
    if (field_mapping) {
      formData.append('field_mapping', JSON.stringify(field_mapping))
    }
    return api
      .post<CsvImportResponse>('/api/alpha/bar-data/import', formData)
      .then((r) => r.data)
  },

  /**
   * 预览 Tick CSV 文件的导入结果。
   *
   * @param file - 待预览的 Tick CSV 文件。
   * @param field_mapping - 自定义列名映射；缺省时后端自动推断。
   * @returns 预览结果，含匹配列、缺失列、样本行与符号列表。
   */
  previewTickCsvImport: (file: File, field_mapping?: Record<string, string>) => {
    const formData = new FormData()
    formData.append('file', file)
    if (field_mapping) {
      formData.append('field_mapping', JSON.stringify(field_mapping))
    }
    return api
      .post<CsvPreviewResult>('/api/alpha/ticks/import/preview', formData)
      .then((r) => r.data)
  },

  /**
   * 正式导入 Tick CSV 文件到本地 Tick 库。
   *
   * @param file - 待导入的 Tick CSV 文件。
   * @param import_mode - `merge` 合并；`replace` 全量覆盖。
   * @param save_mode - `batch` 暂存批次（默认）；`official` 直接写入官方库。
   * @param field_mapping - 自定义列名映射；缺省时后端自动推断。
   * @returns 导入结果，含成功/跳过/错误计数。
   */
  importTickCsvData: (
    file: File,
    import_mode: 'merge' | 'replace',
    save_mode: CsvSaveMode = 'batch',
    field_mapping?: Record<string, string>,
  ) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('import_mode', import_mode)
    formData.append('save_mode', save_mode)
    if (field_mapping) {
      formData.append('field_mapping', JSON.stringify(field_mapping))
    }
    return api
      .post<CsvImportResponse>('/api/alpha/ticks/import', formData)
      .then((r) => r.data)
  },
}

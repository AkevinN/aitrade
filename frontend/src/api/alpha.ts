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
  ParquetStageResult,
  ParquetImportRequest,
} from '../types/alpha'

export const alphaService = {
  // Module status
  /**
   * 查询 Alpha 模块的安装与就绪状态。
   *
   * @returns 模块状态，含 `installed` 等标志；前端据此决定是否禁用数据与训练功能。
   */
  getStatus: () =>
    api.get<AlphaStatus>('/api/alpha/status').then((r) => r.data),

  // Tasks
  /**
   * 拉取全部后台任务（下载/聚合/训练/回测等）的列表。
   *
   * @returns 任务数组，含状态、进度与时间戳；通常按需自行排序后展示。
   */
  listTasks: () =>
    api.get<Task[]>('/api/alpha/tasks').then((r) => r.data),

  /**
   * 按任务 ID 查询单个任务的最新状态，常用于轮询进度。
   *
   * @param taskId - 任务唯一标识，由各 `*Start` 接口返回。
   * @returns 任务详情，含当前状态、进度与失败原因。
   */
  getTask: (taskId: string) =>
    api.get<Task>(`/api/alpha/tasks/${taskId}`).then((r) => r.data),

  // Datasets
  /**
   * 列出所有已生成的数据集名称。
   *
   * @returns 数据集名称数组。
   */
  listDatasets: () =>
    api.get<string[]>('/api/alpha/datasets').then((r) => r.data),

  /**
   * 按名称获取单个数据集的详情。
   *
   * @param name - 数据集名称。
   * @returns 数据集详情，含样本规模、特征列等元信息。
   */
  getDataset: (name: string) =>
    api.get<DatasetDetail>(`/api/alpha/datasets/${name}`).then((r) => r.data),

  /**
   * 提交数据集创建任务（异步执行）。
   *
   * @param req - 数据集构建配置（标的、周期、特征、标签定义等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  createDataset: (req: DatasetCreateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/datasets/create', req).then((r) => r.data),

  /**
   * 删除指定数据集。
   *
   * @param name - 待删除的数据集名称。
   * @returns 操作结果，`success` 标识是否删除成功，`message` 为说明文案。
   */
  deleteDataset: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/datasets/${name}`).then((r) => r.data),

  // Models
  /**
   * 列出所有已训练的 Alpha 模型名称。
   *
   * @returns 模型名称数组。
   */
  listModels: () =>
    api.get<string[]>('/api/alpha/models').then((r) => r.data),

  /**
   * 按名称获取单个模型的详情。
   *
   * @param name - 模型名称。
   * @returns 模型详情，含训练配置、评估指标等元信息。
   */
  getModel: (name: string) =>
    api.get<ModelDetail>(`/api/alpha/models/${name}`).then((r) => r.data),

  /**
   * 提交模型训练任务（异步执行）。
   *
   * @param req - 训练配置（所用数据集、算法、超参数等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  trainModel: (req: ModelTrainRequest) =>
    api.post<TaskStartResponse>('/api/alpha/models/train', req).then((r) => r.data),

  /**
   * 删除指定模型。
   *
   * @param name - 待删除的模型名称。
   * @returns 操作结果，`success` 标识是否删除成功，`message` 为说明文案。
   */
  deleteModel: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/models/${name}`).then((r) => r.data),

  // Signals
  /**
   * 列出所有已生成的交易信号名称。
   *
   * @returns 信号名称数组。
   */
  listSignals: () =>
    api.get<string[]>('/api/alpha/signals').then((r) => r.data),

  /**
   * 按名称获取单个交易信号的详情。
   *
   * @param name - 信号名称。
   * @returns 信号详情，含生成来源、覆盖标的与时间范围等。
   */
  getSignal: (name: string) =>
    api.get<SignalDetail>(`/api/alpha/signals/${name}`).then((r) => r.data),

  /**
   * 基于模型提交信号生成任务（异步执行）。
   *
   * @param req - 信号生成配置（所用模型、标的范围、阈值等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  generateSignal: (req: SignalGenerateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/signals/generate', req).then((r) => r.data),

  /**
   * 删除指定交易信号。
   *
   * @param name - 待删除的信号名称。
   * @returns 操作结果，`success` 标识是否删除成功，`message` 为说明文案。
   */
  deleteSignal: (name: string) =>
    api.delete<{ success: boolean; message: string }>(`/api/alpha/signals/${name}`).then((r) => r.data),

  // Backtest
  /**
   * 提交回测任务（异步执行）。
   *
   * @param req - 回测配置（信号/策略、标的、区间、手续费与阈值等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  runBacktest: (req: BacktestRunRequest) =>
    api.post<TaskStartResponse>('/api/alpha/backtest/run', req).then((r) => r.data),

  // Data download
  /**
   * 提交行情下载任务（异步执行）。
   *
   * @param req - 下载配置（标的、周期、起止日期、数据源等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  downloadData: (req: DataDownloadRequest) =>
    api.post<TaskStartResponse>('/api/alpha/data/download', req).then((r) => r.data),

  /**
   * 提交周期聚合任务（异步执行），从 1m 或 Tick 派生出更大周期。
   *
   * @param req - 聚合配置（源周期、目标周期、标的范围等）。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  aggregateData: (req: DataAggregateRequest) =>
    api.post<TaskStartResponse>('/api/alpha/data/aggregate', req).then((r) => r.data),

  // Symbol profiling
  /**
   * 同步执行标的画像分析并直接返回结果（非任务化）。
   *
   * @param req - 画像配置（目标标的、观测维度等）。
   * @returns 画像分析结果。
   */
  runProfiling: (req: SymbolProfileRequest) =>
    api.post<SymbolProfileResponse>('/api/alpha/profiling', req).then((r) => r.data),

  /**
   * 列出已落盘的标的画像产物 ID。
   *
   * @returns 画像产物 ID 数组。
   */
  listProfilingArtifacts: () =>
    api.get<string[]>('/api/alpha/profiling/artifacts').then((r) => r.data),

  /**
   * 按产物 ID 读取一份已保存的标的画像结果。
   *
   * @param artifactId - 画像产物 ID；内部会做 URL 编码，可含特殊字符。
   * @returns 画像分析结果。
   */
  getProfilingArtifact: (artifactId: string) =>
    api
      .get<SymbolProfileResponse>(`/api/alpha/profiling/${encodeURIComponent(artifactId)}`)
      .then((r) => r.data),

  // Contracts
  /**
   * 获取后端可用的合约（标的）元数据映射。
   *
   * @returns 以合约代码为键的元数据对象；字段结构由后端决定，故为弱类型。
   */
  getContracts: () =>
    api.get<Record<string, unknown>>('/api/alpha/contracts').then((r) => r.data),

  // Unified data resources
  /**
   * 拉取统一数据资源清单（原始K线/历史Tick/派生周期等的汇总）。
   *
   * @returns 资源列表，含各类资源的键、周期与覆盖范围；用于资源页与就绪面板。
   */
  getDataResources: () =>
    api.get<DataResourceList>('/api/alpha/data/resources').then((r) => r.data),

  /**
   * 获取某类数据资源中单个条目的明细。
   *
   * @param kind - 资源类别（如原始K线/Tick/派生周期）。
   * @param key - 资源条目键；内部会做 URL 编码，可含特殊字符。
   * @param query - 可选的明细查询参数（如时间范围、行数上限）；缺省时返回默认范围。
   * @returns 资源明细，含数据行与统计信息。
   */
  getDataResourceDetail: (kind: DataResourceKind, key: string, query?: BarDataDetailQuery) =>
    api
      .get<DataResourceDetail>(`/api/alpha/data/resources/${kind}/${encodeURIComponent(key)}`, {
        params: query as Record<string, string>,
      })
      .then((r) => r.data),

  /**
   * 删除某类数据资源中的单个条目。
   *
   * @param kind - 资源类别。
   * @param key - 待删除条目键；内部会做 URL 编码。
   * @returns 操作结果，`success` 标识是否删除成功，`message` 为说明文案。
   */
  deleteDataResource: (kind: DataResourceKind, key: string) =>
    api
      .delete<{ success: boolean; message: string }>(`/api/alpha/data/resources/${kind}/${encodeURIComponent(key)}`)
      .then((r) => r.data),

  /**
   * 改写某条原始K线资源被归类到的周期（纠正错配的 interval）。
   *
   * @param key - 原始K线资源键；内部会做 URL 编码。
   * @param interval - 重新归类到的目标周期。
   * @returns 重定位结果，含变更后的资源标识。
   */
  relocateRawBarInterval: (key: string, interval: CsvInterval) =>
    api
      .patch<RelocateBarIntervalResponse>(
        `/api/alpha/data/resources/raw_bar/${encodeURIComponent(key)}/interval`,
        { interval },
      )
      .then((r) => r.data),

  /**
   * 预览多批次数据资源合并的结果，不落盘。
   *
   * @param req - 合并配置（参与合并的资源键、冲突处理策略等）。
   * @returns 合并预览，含预计行数、重叠与冲突信息，供用户确认。
   */
  previewDataResourceMerge: (req: DataResourceMergeRequest) =>
    api
      .post<DataResourceMergePreview>('/api/alpha/data/resources/merge/preview', req)
      .then((r) => r.data),

  /**
   * 正式执行多批次数据资源合并并落盘。
   *
   * @param req - 合并配置；通常与 {@link previewDataResourceMerge} 同参，确认后调用。
   * @returns 合并结果，含写入行数与生成的资源标识。
   */
  mergeDataResourceBatches: (req: DataResourceMergeRequest) =>
    api
      .post<DataResourceMergeResponse>('/api/alpha/data/resources/merge', req)
      .then((r) => r.data),

  // Bar data
  /**
   * 拉取本地K线库中已有的 (周期, 标的) 条目清单。
   *
   * @returns K线条目列表，按周期与标的组织。
   */
  getBarData: () =>
    api.get<BarDataList>('/api/alpha/bar-data').then((r) => r.data),

  /**
   * 获取指定周期、标的的K线明细数据。
   *
   * @param interval - K线周期（`d` 日线，`1m`/`5m`/`30m` 等分钟线）。
   * @param vtSymbol - 合约代码，如 `000001.SZSE`。
   * @param query - 可选的明细查询参数（如时间范围、行数上限）；缺省时返回默认范围。
   * @returns K线明细，含逐根K线与统计信息。
   */
  getBarDataDetail: (interval: string, vtSymbol: string, query?: BarDataDetailQuery) =>
    api
      .get<BarDataDetail>(`/api/alpha/bar-data/${interval}/${vtSymbol}`, {
        params: query as Record<string, string>,
      })
      .then((r) => r.data),

  /**
   * 删除指定周期、标的的K线数据。
   *
   * @param interval - K线周期。
   * @param vtSymbol - 合约代码。
   * @returns 操作结果，`success` 标识是否删除成功，`message` 为说明文案。
   */
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

  // Parquet Import
  /**
   * 上传一个或多个 Parquet 文件做暂存与解析预览，不落盘。
   *
   * 以 multipart/form-data 提交：每个文件追加到重复字段 `files`，外加一个
   * `data_kind` 字段。后端解析后返回会话 ID 与逐文件预览，供用户确认后再正式导入。
   *
   * @param files - 待暂存的 Parquet 文件数组（支持一次多选）。
   * @param data_kind - `bar` 解析为 K线；`tick` 解析为逐笔。
   * @returns 暂存结果，含 `session_id` 与逐文件 {@link ParquetStageResult.files} 预览。
   */
  stageParquet: (files: File[], data_kind: 'bar' | 'tick') => {
    const formData = new FormData()
    formData.append('data_kind', data_kind)
    files.forEach((file) => formData.append('files', file))
    return api
      .post<ParquetStageResult>('/api/alpha/parquet/stage', formData)
      .then((r) => r.data)
  },

  /**
   * 基于已暂存会话提交 Parquet 正式导入任务（异步执行）。
   *
   * @param req - 导入配置，含 `session_id`、`data_kind`、`interval` 与 `import_mode`。
   * @returns 任务启动响应，含可用于轮询的 `task_id`。
   */
  importParquet: (req: ParquetImportRequest) =>
    api.post<TaskStartResponse>('/api/alpha/parquet/import', req).then((r) => r.data),

  /**
   * 取消一个 Parquet 暂存会话并清理其临时文件。
   *
   * @param session_id - 待取消的暂存会话 ID；内部会做 URL 编码。
   * @returns 操作结果，`success` 标识是否清理成功，`message` 为说明文案。
   */
  cancelParquetStage: (session_id: string) =>
    api
      .delete<{ success: boolean; message?: string }>(
        `/api/alpha/parquet/stage/${encodeURIComponent(session_id)}`,
      )
      .then((r) => r.data),
}

// Backend response types — must match Pydantic models in backend/aitrade/models/alpha.py

export type TaskStatusValue = 'pending' | 'running' | 'completed' | 'failed'

export interface Task {
  id: string
  type: string
  status: TaskStatusValue
  progress: number
  message: string
  result?: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface TaskStartResponse {
  id: string
  message: string
}

// GET /api/alpha/status
export interface AlphaStatus {
  installed: boolean
  lab_path: string
  lab_exists: boolean
}

// GET /api/alpha/datasets/{name}
export interface DatasetDetail {
  name: string
  feature_count: number
  sample_count: number
  label_expression: string
}

// GET /api/alpha/bar-data
export interface BarDataList {
  daily: BarDataItem[]
  minute: BarDataItem[]
}

export interface BarDataItem {
  vt_symbol: string
  row_count: number
  start: string
  end: string
  file_size_kb: number
}

// GET /api/alpha/bar-data/{interval}/{vt_symbol}
export interface BarDataDetail {
  vt_symbol: string
  interval: string
  row_count: number
  start: string
  end: string
  columns: string[]
  preview: Record<string, unknown>[]
  loaded_count: number
  has_more: boolean
  next_before: string | null
}

export interface BarDataDetailQuery {
  limit?: number
  before?: string
}

// GET /api/alpha/models/{name}
export interface ModelDetail {
  name: string
  model_type: string
}

// GET /api/alpha/signals/{name}
export interface SignalDetail {
  name: string
  row_count: number
  columns: string[]
  preview: Record<string, unknown>[]
}

// Request types (match backend Pydantic models)

export interface DataDownloadRequest {
  vt_symbols: string[]
  start: string
  end: string
  interval?: string
}

export interface DatasetCreateRequest {
  name: string
  vt_symbols: string[]
  start: string
  end: string
  train_end?: string
  valid_end?: string
  features?: string[]
  label_period?: number
}

export interface ModelTrainRequest {
  name: string
  dataset: string
  model_type: string
  params?: Record<string, unknown>
}

export interface SignalGenerateRequest {
  name: string
  model: string
  vt_symbols: string[]
  start: string
  end: string
}

export interface BacktestRunRequest {
  name: string
  signal: string
  capital?: number
  start: string
  end: string
  benchmark?: string
}

export interface CNNTrainRequest {
  name: string
  vt_symbols: string[]
  start: string
  end: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
  lookback?: number
  dropout?: number
  train_ratio?: number
}

// CSV Import types

export interface CsvPreviewResult {
  columns: string[]
  sample_rows: Record<string, unknown>[]
  matched_fields: Record<string, string>
  unmapped_columns: string[]
  missing_required: string[]
  total_rows: number
  date_range: [string, string]
  symbols: string[]
}

export interface CsvImportResponse {
  success: boolean
  message: string
  imported_count: number
  skipped_count: number
  errors: string[]
}

export type CsvInterval = 'd' | 'm'
export type CsvImportMode = 'merge' | 'replace'

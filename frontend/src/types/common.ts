// Match backend SystemStatus from /api/status
export interface SystemStatus {
  version: string
  torch_available: boolean
  torch_device: string
  data_path: string
  tushare_token_set: boolean
  providers: ProviderInfo[]
}

export interface ProviderInfo {
  name: string
  priority: number
  status: string
  description: string
}

export interface ApiResponse<T> {
  data: T
  message?: string
}

export interface PaginationParams {
  page: number
  page_size: number
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

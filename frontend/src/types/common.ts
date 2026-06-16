/** 后端系统状态（`GET /api/status` 响应体，与后端 `SystemStatus` 对齐）。 */
export interface SystemStatus {
  /** 后端版本号。 */
  version: string
  /** PyTorch 是否可用。 */
  torch_available: boolean
  /** 当前 torch 推理设备（`cpu` / `cuda`）。 */
  torch_device: string
  /** 本地数据存储根路径。 */
  data_path: string
  /** Tushare token 是否已配置。 */
  tushare_token_set: boolean
  /** 已注册数据提供方列表。 */
  providers: ProviderInfo[]
}

/** 单个数据提供方信息。 */
export interface ProviderInfo {
  /** 提供方名称（如 `tushare`）。 */
  name: string
  /** 优先级（数字越小越优先）。 */
  priority: number
  /** 当前状态（`ok` / `degraded` 等）。 */
  status: string
  /** 可读说明。 */
  description: string
}

/**
 * 通用 API 响应包装（部分接口使用，非全量）。
 *
 * @template T - 实际数据体类型。
 */
export interface ApiResponse<T> {
  /** 响应数据体。 */
  data: T
  /** 可选的可读消息。 */
  message?: string
}

/** 通用分页查询参数。 */
export interface PaginationParams {
  /** 页码（从 1 开始）。 */
  page: number
  /** 每页条数。 */
  page_size: number
}

/**
 * 通用分页响应体。
 *
 * @template T - 列表元素类型。
 */
export interface PaginatedResponse<T> {
  /** 当前页数据列表。 */
  items: T[]
  /** 数据总条数。 */
  total: number
  /** 当前页码。 */
  page: number
  /** 每页条数。 */
  page_size: number
}

/**
 * 系统状态 API 服务对象——封装与后端 `/api/status` 端点的 HTTP 交互。
 *
 * 用于仪表板等场景，读取后端整体可用性（torch 是否安装、数据路径、数据提供方列表等）。
 */
import api from './client'
import type { SystemStatus } from '../types/common'

export const statusService = {
  /**
   * 获取后端系统状态快照。
   *
   * @returns {@link SystemStatus}，包含版本号、torch 可用性、数据路径及数据提供方列表。
   */
  getStatus: () =>
    api.get<SystemStatus>('/api/status').then((r) => r.data),
}

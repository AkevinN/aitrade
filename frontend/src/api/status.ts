import api from './client'
import type { SystemStatus } from '../types/common'

export const statusService = {
  getStatus: () =>
    api.get<SystemStatus>('/api/status').then((r) => r.data),
}

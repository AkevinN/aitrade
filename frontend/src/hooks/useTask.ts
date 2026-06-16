import { useQuery } from '@tanstack/react-query'
import { alphaService } from '../api/alpha'
import type { Task } from '../types/alpha'

/**
 * 轮询单条异步任务，直到任务达到终态（completed / failed）。
 *
 * 任务未到终态时每 2 秒重新请求一次；到达终态后停止轮询。
 * 常与 WS 推送配合：WS 更新写入 `taskStore`，本 Hook 提供首帧与兜底刷新。
 *
 * @param taskId - 任务 ID；为 `null` 时不发起请求。
 * @param enabled - 额外开关，可在外部条件不满足时临时挂起（默认 `true`）。
 * @returns TanStack Query 结果对象，`data` 为 {@link Task}。
 *
 * @example
 * ```ts
 * const task = useTask(taskId)
 * if (task.data?.status === 'completed') { ... }
 * ```
 */
export function useTask(taskId: string | null, enabled = true) {
  return useQuery<Task>({
    queryKey: ['task', taskId],
    queryFn: () => alphaService.getTask(taskId!),
    enabled: enabled && !!taskId,
    refetchInterval: (q) => {
      const task = q.state.data
      if (!task) return 2000
      if (task.status === 'completed' || task.status === 'failed') return false
      return 2000
    },
    retry: 1,
  })
}

/**
 * 拉取全量任务列表，每 5 秒自动刷新一次。
 *
 * 用于仪表板、资源页等需要展示近期任务汇总的场景。
 *
 * @returns TanStack Query 结果对象，`data` 为 {@link Task} 数组。
 */
export function useTaskList() {
  return useQuery<Task[]>({
    queryKey: ['tasks'],
    queryFn: () => alphaService.listTasks(),
    refetchInterval: 5000,
    retry: 1,
  })
}

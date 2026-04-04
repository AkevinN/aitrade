import { useQuery } from '@tanstack/react-query'
import { alphaService } from '../api/alpha'
import type { Task } from '../types/alpha'

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

export function useTaskList() {
  return useQuery<Task[]>({
    queryKey: ['tasks'],
    queryFn: () => alphaService.listTasks(),
    refetchInterval: 5000,
    retry: 1,
  })
}

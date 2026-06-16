import { create } from 'zustand'
import type { Task } from '../types/alpha'

/**
 * Zustand 任务状态 store 的形状定义。
 *
 * 以 `task_id` 为键在内存中维护异步任务记录；
 * WebSocket `task_update` 事件与轮询均通过 `addTask` / `updateTask` 写入。
 */
interface TaskState {
  /** 以 task_id 为键的任务字典。 */
  tasks: Record<string, Task>
  /**
   * 新增或全量覆盖一条任务记录。
   *
   * @param task - 完整的任务对象。
   */
  addTask: (task: Task) => void
  /**
   * 对已有任务做部分字段更新（浅合并）。
   *
   * @param taskId - 待更新的任务 ID。
   * @param updates - 需更新的字段子集。
   */
  updateTask: (taskId: string, updates: Partial<Task>) => void
  /** 清空所有任务记录（如退出登录或页面重置时调用）。 */
  clearTasks: () => void
}

/**
 * 全局任务状态 store（Zustand）。
 *
 * 全局单例：WebSocket hook 和轮询 hook 共享同一份任务快照，
 * 组件直接订阅感兴趣的 `task_id` 即可获得实时更新。
 *
 * @example
 * ```ts
 * // 在非 React 上下文（如 WS handler）写入
 * taskStore.getState().addTask(taskData)
 * // 在组件内读取
 * const tasks = useTaskStore().tasks
 * ```
 */
export const taskStore = create<TaskState>((set) => ({
  tasks: {},
  addTask: (task) =>
    set((state) => ({
      tasks: { ...state.tasks, [task.task_id]: task },
    })),
  updateTask: (taskId, updates) =>
    set((state) => ({
      tasks: {
        ...state.tasks,
        [taskId]: { ...state.tasks[taskId], ...updates },
      },
    })),
  clearTasks: () => set({ tasks: {} }),
}))

/**
 * 在 React 组件内订阅全局任务 store 的便捷 Hook。
 *
 * @returns 当前 `TaskState` 快照（Zustand 自动追踪依赖，状态变化时重渲染）。
 *
 * @example
 * ```tsx
 * const { tasks } = useTaskStore()
 * ```
 */
export const useTaskStore = () => taskStore()

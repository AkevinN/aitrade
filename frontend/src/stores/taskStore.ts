import { create } from 'zustand'
import type { Task } from '../types/alpha'

interface TaskState {
  tasks: Record<string, Task>
  addTask: (task: Task) => void
  updateTask: (taskId: string, updates: Partial<Task>) => void
  clearTasks: () => void
}

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

export const useTaskStore = () => taskStore()

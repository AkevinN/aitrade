/**
 * 运行历史拉取 Hook：从任务历史读取过去的回测/选股运行。
 *
 * 数据源是既有任务历史（`GET /api/alpha/tasks?include_history=true`，已永久落盘、已脱敏），
 * 而非新建库——回测无专门结果库、选股 persist 默认关闭，任务历史是二者统一的持久化源。
 * 纯只读；轻量轮询让"运行中"的运行进度也能刷新。
 */
import { useQuery } from '@tanstack/react-query'

import { alphaService } from '../api/alpha'
import type { Task } from '../types/alpha'
import {
  ALL_RUN_TYPES,
  BACKTEST_RUN_TYPES,
  SCREENING_RUN_TYPES,
} from '../pages/RunHistory/runTypes'

/** 运行历史类别过滤。 */
export type RunHistoryCategory = 'all' | 'backtest' | 'screening'

/**
 * 拉取指定类别的运行历史（按后端 updated_at 倒序）。
 *
 * @param category - 'all' / 'backtest' / 'screening'，决定请求哪些 task_type
 * @returns React Query 结果，data 为 Task 列表（含 result/status/duration_ms 等）
 */
export function useRunHistory(category: RunHistoryCategory) {
  const taskType =
    category === 'backtest'
      ? [...BACKTEST_RUN_TYPES]
      : category === 'screening'
        ? [...SCREENING_RUN_TYPES]
        : ALL_RUN_TYPES

  return useQuery<Task[]>({
    queryKey: ['runHistory', category],
    queryFn: () =>
      alphaService.listTasks({
        taskType,
        includeHistory: true,
        limit: 200,
        historyDays: 180,
      }),
    refetchInterval: 10_000,
  })
}

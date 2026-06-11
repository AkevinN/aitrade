// PlanManager 示例测试：列表/调度状态渲染、安全提示存在性、立即触发联动（Req 8.x / 9.5）。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import type { UseQueryResult } from '@tanstack/react-query'

import type { Task } from '../../types/alpha'
import { makeTask, makeResult, makePlanSummary, makeSchedulerStatus } from './testFixtures'

const taskState: { data: Task | null } = { data: null }
vi.mock('../../hooks/useTask', () => ({
  useTask: () => taskState as unknown as UseQueryResult<Task>,
}))

vi.mock('../../api/cnn', () => ({
  cnnService: { listModels: vi.fn().mockResolvedValue([]) },
}))

const runPlan = vi.fn().mockResolvedValue({ task_id: 'task-1', message: '已按计划触发' })
vi.mock('../../api/liveApi', () => ({
  liveService: {
    listPlans: vi.fn().mockResolvedValue([makePlanSummary()]),
    getSchedulerStatus: vi.fn().mockResolvedValue(makeSchedulerStatus()),
    runPlan: (...args: unknown[]) => runPlan(...args),
    getPlan: vi.fn(),
    createPlan: vi.fn(),
    updatePlan: vi.fn(),
    deletePlan: vi.fn(),
    togglePlan: vi.fn(),
  },
}))

import PlanManager from './PlanManager'

function renderManager() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <PlanManager />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('PlanManager', () => {
  beforeEach(() => {
    taskState.data = null
    runPlan.mockClear()
  })

  it('展示安全提示、计划列表与调度状态', async () => {
    renderManager()
    expect(screen.getByText('仅提醒，不自动下单')).toBeInTheDocument()
    expect(await screen.findByText('尾盘买入计划')).toBeInTheDocument()
    expect(screen.getByText('运行中')).toBeInTheDocument()
  })

  it('立即触发后调用 runPlan 并联动展示决策结果', async () => {
    taskState.data = makeTask({ status: 'completed', result: makeResult() })
    renderManager()
    fireEvent.click(await screen.findByText('立即触发'))
    await waitFor(() => expect(runPlan).toHaveBeenCalledWith('plan-1'))
    // 触发后展示结果卡片与买入决策。
    expect(await screen.findByText('立即触发结果')).toBeInTheDocument()
    expect(screen.getByText('买入')).toBeInTheDocument()
  })
})

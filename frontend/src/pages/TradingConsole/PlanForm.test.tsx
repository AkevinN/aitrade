// PlanForm 示例测试：字段存在性、凭证提示、提交装配请求（Req 8.4 / 9.5）。
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import PlanForm from './PlanForm'
import { makePlan } from './testFixtures'

vi.mock('../../api/cnn', () => ({
  cnnService: {
    listModels: vi
      .fn()
      .mockResolvedValue([{ name: 'demo' }, { name: 'm30', input_interval: '30m' }]),
  },
}))

function renderForm(props: Partial<React.ComponentProps<typeof PlanForm>> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onSubmit = props.onSubmit ?? vi.fn()
  render(
    <QueryClientProvider client={qc}>
      <PlanForm onSubmit={onSubmit} {...props} />
    </QueryClientProvider>,
  )
  return { onSubmit }
}

describe('PlanForm', () => {
  it('展示多唤醒时刻、添加按钮与凭证/多时刻/安全提示', () => {
    renderForm()
    expect(screen.getByText('唤醒时刻（可配置多个）')).toBeInTheDocument()
    expect(screen.getByText('添加时刻')).toBeInTheDocument()
    expect(screen.getByText(/凭证由后端环境变量管理/)).toBeInTheDocument()
    expect(screen.getByText(/当日至多产出一次决策/)).toBeInTheDocument()
    expect(screen.getByText(/不会向任何券商提交真实订单/)).toBeInTheDocument()
  })

  it('编辑模式回填计划名称', () => {
    renderForm({ initialPlan: makePlan({ name: '已存在计划' }) })
    expect(screen.getByDisplayValue('已存在计划')).toBeInTheDocument()
  })

  it('提交回调上抛装配好的请求体（trigger_times + bar_freq）', async () => {
    const onSubmit = vi.fn()
    renderForm({ initialPlan: makePlan(), onSubmit })
    fireEvent.click(screen.getByText('保存计划'))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    const req = onSubmit.mock.calls[0][0]
    expect(req.name).toBe('尾盘买入计划')
    expect(req.trigger_times).toEqual(['15:05'])
    expect(req.bar_freq).toBe('1d')
    expect(req.notify_channels).toEqual(['dingtalk'])
  })

  it('回填多个唤醒时刻并提交去重升序数组', async () => {
    const onSubmit = vi.fn()
    renderForm({ initialPlan: makePlan({ trigger_times: ['15:35', '15:05'] }), onSubmit })
    expect(screen.getByDisplayValue('15:05')).toBeInTheDocument()
    expect(screen.getByDisplayValue('15:35')).toBeInTheDocument()
    fireEvent.click(screen.getByText('保存计划'))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    const req = onSubmit.mock.calls[0][0]
    expect(req.trigger_times).toEqual(['15:05', '15:35'])
  })

  it('日内模型：bar_freq 由模型派生锁定，隐藏唤醒时刻并展示监控模式提示', async () => {
    renderForm({
      initialPlan: makePlan({ model: 'm30', bar_freq: '30m', trigger_times: [] }),
    })
    // bar_freq 只读展示派生值（模型 m30 训练间隔 30m）。
    await waitFor(() =>
      expect(screen.getByDisplayValue('盘中监控 · 30m')).toBeInTheDocument(),
    )
    expect(screen.queryByText('唤醒时刻（可配置多个）')).not.toBeInTheDocument()
    expect(screen.getByText(/盘中监控模式：交易时段内每根 30m bar 收盘后自动决策一次/)).toBeInTheDocument()
  })

  it('日内模型：提交 bar_freq=30m 且 trigger_times 为空列表', async () => {
    const onSubmit = vi.fn()
    renderForm({
      initialPlan: makePlan({ model: 'm30', bar_freq: '30m', trigger_times: [] }),
      onSubmit,
    })
    await waitFor(() =>
      expect(screen.getByDisplayValue('盘中监控 · 30m')).toBeInTheDocument(),
    )
    fireEvent.click(screen.getByText('保存计划'))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    const req = onSubmit.mock.calls[0][0]
    expect(req.bar_freq).toBe('30m')
    expect(req.trigger_times).toEqual([])
  })
})

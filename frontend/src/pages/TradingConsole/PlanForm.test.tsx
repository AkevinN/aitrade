// PlanForm 示例测试：字段存在性、凭证提示、提交装配请求（Req 8.4 / 9.5）。
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import userEvent from '@testing-library/user-event'

import PlanForm from './PlanForm'
import { makePlan } from './testFixtures'

vi.mock('../../api/cnn', () => ({
  cnnService: {
    listModels: vi
      .fn()
      .mockResolvedValue([{ name: 'demo' }, { name: 'm30', input_interval: '30m' }]),
  },
}))

vi.mock('../../api/strategy', () => ({
  strategyService: {
    listSources: vi.fn().mockResolvedValue([
      { name: 'momentum', description: '动量信号', param_spec: null },
      { name: 'mean_reversion', description: '均值回归', param_spec: null },
    ]),
  },
}))

function renderForm(props: Partial<React.ComponentProps<typeof PlanForm>> = {}) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onSubmit = props.onSubmit ?? vi.fn()
  render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <PlanForm onSubmit={onSubmit} {...props} />
      </AntApp>
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

  // ---- rule 模式测试（v2）----

  it('切换到 rule 模式后 signal_source 必填校验阻止提交', async () => {
    const onSubmit = vi.fn()
    renderForm({ onSubmit })

    // 点击「规则调仓」Radio
    fireEvent.click(screen.getByText('规则调仓'))
    // 等待 rule 模式字段出现
    await waitFor(() => expect(screen.getByText('信号源')).toBeInTheDocument())

    // 先填写计划名称（避免 name 必填阻断）
    const nameInput = screen.getByPlaceholderText('如：平安银行尾盘买入计划')
    await userEvent.clear(nameInput)
    await userEvent.type(nameInput, '测试规则计划')

    // 填写组合 ID
    const portfolioInput = screen.getByPlaceholderText('portfolio-001')
    await userEvent.type(portfolioInput, 'p-001')

    // 不填 signal_source，直接提交
    fireEvent.click(screen.getByText('创建计划'))
    // onSubmit 不应被调用（校验失败）
    await new Promise((r) => setTimeout(r, 100))
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('rule 模式提交载荷含 trigger_times 且 model/vt_symbol/scheme 为空串', async () => {
    const onSubmit = vi.fn()
    // 用已有 rule 计划回填，绕过 jsdom 中 antd Select pointer-events 限制
    const rulePlan = makePlan({
      strategy_type: 'rule' as const,
      model: '',
      vt_symbol: '',
      scheme: '',
      signal_source: 'etf_momentum',
      trigger_schedule: 'daily' as const,
      portfolio_id: 'p-001',
      trigger_times: ['15:05'],
    } as Parameters<typeof makePlan>[0])
    renderForm({ initialPlan: rulePlan, onSubmit })

    // 等待 rule 模式字段出现
    await waitFor(() => expect(screen.getByText('信号源')).toBeInTheDocument())

    fireEvent.click(screen.getByText('保存计划'))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    const req = onSubmit.mock.calls[0][0]
    // 契约：model/vt_symbol/scheme 为空串（后端已接受此形态）
    expect(req.model).toBe('')
    expect(req.vt_symbol).toBe('')
    expect(req.scheme).toBe('')
    // 契约：trigger_times 非空（["15:05"]）
    expect(req.trigger_times).toEqual(['15:05'])
    expect(req.strategy_type).toBe('rule')
    expect(req.signal_source).toBe('etf_momentum')
  })

  it('cnn 模式载荷不含 rule 专属字段（strategy_type=cnn）', async () => {
    const onSubmit = vi.fn()
    renderForm({ initialPlan: makePlan(), onSubmit })

    // 确保还在 cnn 模式（默认）
    expect(screen.getByText('CNN 决策')).toBeInTheDocument()

    fireEvent.click(screen.getByText('保存计划'))
    await waitFor(() => expect(onSubmit).toHaveBeenCalled())
    const req = onSubmit.mock.calls[0][0]
    // cnn 模式：strategy_type=cnn，不含 signal_source / trigger_schedule
    expect(req.strategy_type).toBe('cnn')
    expect(req.signal_source).toBeUndefined()
    expect(req.trigger_schedule).toBeUndefined()
    expect(req.portfolio_id).toBeUndefined()
  })
})

// 规则策略回测页面测试 — RTL 风格，仿 TradingConsole 组件测试
// mock API 模块 + antd App 包裹 + role/文本断言

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import type { UseQueryResult } from '@tanstack/react-query'

import type { Task } from '../../types/alpha'
import type { SignalSourceInfo } from '../../types/strategy'

// ── mock strategy API ──────────────────────────────────────────────────────
const mockListSources = vi.fn<() => Promise<SignalSourceInfo[]>>()
const mockRunBacktest = vi.fn()
const mockRunSweep = vi.fn()
const mockRunWalkForward = vi.fn()

vi.mock('../../api/strategy', () => ({
  strategyService: {
    listSources: () => mockListSources(),
    runBacktest: (...args: unknown[]) => mockRunBacktest(...args),
    runSweep: (...args: unknown[]) => mockRunSweep(...args),
    runWalkForward: (...args: unknown[]) => mockRunWalkForward(...args),
  },
}))

// ── mock useTask ────────────────────────────────────────────────────────────
// 每个用例通过 taskState 控制 useTask 返回值
interface TaskState {
  data: Task | null
}
const taskState: TaskState = { data: null }
vi.mock('../../hooks/useTask', () => ({
  useTask: () => taskState as unknown as UseQueryResult<Task>,
}))

// ── sample fixtures ─────────────────────────────────────────────────────────
const ETF_MOMENTUM_SOURCE: SignalSourceInfo = {
  name: 'etf_momentum',
  description: 'ETF 动量轮动信号',
  param_spec: {
    lookback: { type: 'int', default: 20, label: '动量回看天数' },
  },
}

function makeCompletedTask(statistics?: Record<string, unknown>): Task {
  return {
    task_id: 'task-rule-1',
    type: 'strategy_backtest',
    title: '规则策略回测',
    entity_type: 'strategy',
    entity_name: 'etf_momentum',
    status: 'completed',
    progress: 100,
    message: '完成',
    result: {
      statistics: statistics ?? {
        total_return: 0.25,
        annual_return: 0.12,
        sharpe_ratio: 1.35,
        max_ddpercent: -0.15,
        total_trade_count: 42,
        total_net_pnl: 250000,
        end_balance: 1250000,
        start_date: '2024-01-01',
        end_date: '2025-01-01',
        total_days: 252,
        profit_days: 130,
        loss_days: 122,
      },
    },
    created_at: '2026-06-11T10:00:00',
    updated_at: '2026-06-11T10:05:00',
  }
}

import RuleBacktest from './RuleBacktest'

// ── render helper ───────────────────────────────────────────────────────────
function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <RuleBacktest />
      </AntApp>
    </QueryClientProvider>,
  )
}

// antd Select 选项渲染到 document.body portal，用 document.querySelector 查找
async function selectOption(comboboxIndex: number, optionText: RegExp | string) {
  const combos = screen.getAllByRole('combobox')
  const combo = combos[comboboxIndex]
  // Click to open
  fireEvent.mouseDown(combo)
  // Wait for option in body
  await waitFor(() => {
    const opts = document.querySelectorAll('.ant-select-item-option')
    const found = Array.from(opts).some((el) =>
      typeof optionText === 'string'
        ? el.textContent?.includes(optionText)
        : optionText.test(el.textContent ?? ''),
    )
    if (!found) throw new Error(`Option "${String(optionText)}" not found in dropdown`)
  })
  // Click the matching option
  const opts = document.querySelectorAll('.ant-select-item-option')
  const target = Array.from(opts).find((el) =>
    typeof optionText === 'string'
      ? el.textContent?.includes(optionText)
      : optionText.test(el.textContent ?? ''),
  ) as HTMLElement | undefined
  if (target) fireEvent.click(target)
}

// ────────────────────────────────────────────────────────────────────────────

describe('RuleBacktest 页面', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    taskState.data = null
    mockListSources.mockResolvedValue([ETF_MOMENTUM_SOURCE])
    mockRunBacktest.mockResolvedValue({ task_id: 'task-rule-1' })
    mockRunSweep.mockResolvedValue({ task_id: 'task-sweep-1' })
    mockRunWalkForward.mockResolvedValue({ task_id: 'task-wf-1' })
  })

  // ① 渲染后信号源列表加载并可选（mock listSources 返回 etf_momentum）
  it('渲染后信号源列表加载，Select 显示 etf_momentum 选项', async () => {
    renderPage()

    // 页面标题应在
    expect(screen.getByText('规则策略回测配置')).toBeInTheDocument()

    // 等待 listSources 响应
    await waitFor(() => expect(mockListSources).toHaveBeenCalledTimes(1))

    // 展开信号源 Select，验证 etf_momentum 出现在下拉选项中
    const combos = screen.getAllByRole('combobox')
    fireEvent.mouseDown(combos[0])

    // 下拉选项渲染到 portal（document.body）
    await waitFor(() => {
      const opts = document.querySelectorAll('.ant-select-item-option')
      const found = Array.from(opts).some((el) => /etf_momentum/.test(el.textContent ?? ''))
      expect(found).toBe(true)
    })
  })

  // ② 填表提交 → runBacktest 被调且载荷含 signal_params.universe 数组 / cost.stamp_duty=0.0005
  it('填表后提交回测，runBacktest 收到正确的 signal_params.universe 和 cost.stamp_duty', async () => {
    renderPage()

    // 等待信号源加载
    await waitFor(() => expect(mockListSources).toHaveBeenCalled())

    // 选择信号源
    await selectOption(0, 'etf_momentum')

    // 点击「启动回测」
    const runBtn = screen.getByRole('button', { name: /启动回测/ })
    fireEvent.click(runBtn)

    await waitFor(() => expect(mockRunBacktest).toHaveBeenCalledTimes(1))

    const payload = mockRunBacktest.mock.calls[0][0]
    // universe 应为数组（从 TextArea 按行分割）
    expect(Array.isArray(payload.signal_params.universe)).toBe(true)
    expect(payload.signal_params.universe.length).toBeGreaterThan(0)
    // 印花税默认 0.0005
    expect(payload.cost.stamp_duty).toBe(0.0005)
    // 信号源字段
    expect(payload.signal_source).toBe('etf_momentum')
  })

  // ③ task completed 后 BacktestResults 出现统计卡
  it('task 完成后，BacktestResults 展示统计数据', async () => {
    taskState.data = makeCompletedTask()

    renderPage()

    // BacktestResults 在 completed 状态时应展示统计数字
    // 统计区不应再显示空状态
    expect(screen.queryByText('启动回测查看统计结果')).not.toBeInTheDocument()

    // 关键统计指标文字（BacktestResults 的 Statistic title）
    expect(screen.getByText('总收益')).toBeInTheDocument()
    expect(screen.getByText('夏普比率')).toBeInTheDocument()
  })

  // ④ Walk-Forward 载荷包含 train_days / test_days（对齐后端 StrategyWalkForwardRequest）
  it('Walk-Forward 提交载荷含 train_days 和 test_days', async () => {
    renderPage()

    await waitFor(() => expect(mockListSources).toHaveBeenCalled())
    await selectOption(0, 'etf_momentum')

    // 展开折叠面板
    const collapseHeader = screen.getByText('参数扫描 / Walk-Forward')
    fireEvent.click(collapseHeader)

    // 点击「启动 Walk-Forward」
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /启动 Walk-Forward/ })).toBeInTheDocument()
    })
    const wfBtn = screen.getByRole('button', { name: /启动 Walk-Forward/ })
    fireEvent.click(wfBtn)

    await waitFor(() => expect(mockRunWalkForward).toHaveBeenCalledTimes(1))

    const payload = mockRunWalkForward.mock.calls[0][0]
    // 载荷应含 train_days / test_days，不含旧的 n_splits / train_ratio
    expect(payload).toHaveProperty('train_days')
    expect(payload).toHaveProperty('test_days')
    expect(payload).not.toHaveProperty('n_splits')
    expect(payload).not.toHaveProperty('train_ratio')
    // 默认值
    expect(payload.train_days).toBe(180)
    expect(payload.test_days).toBe(60)
  })

  // ⑤ sweep 网格 JSON 非法行 → 友好报错不提交
  it('sweep 网格含非法 JSON 时，显示错误提示且不调用 runSweep', async () => {
    renderPage()

    // 先选信号源
    await waitFor(() => expect(mockListSources).toHaveBeenCalled())
    await selectOption(0, 'etf_momentum')

    // 展开折叠面板
    const collapseHeader = screen.getByText('参数扫描 / Walk-Forward')
    fireEvent.click(collapseHeader)

    // 等待折叠面板展开（找到 TextArea）
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/strategy_params/)).toBeInTheDocument()
    })

    // 输入非法 JSON
    const gridTextarea = screen.getByPlaceholderText(/strategy_params/)
    fireEvent.change(gridTextarea, { target: { value: 'not-valid-json' } })

    // 点击「启动参数扫描」
    const sweepBtn = screen.getByRole('button', { name: /启动参数扫描/ })
    fireEvent.click(sweepBtn)

    // 应出现错误提示
    await waitFor(() => {
      expect(screen.getByText(/JSON 解析失败/)).toBeInTheDocument()
    })

    // runSweep 不应被调用
    expect(mockRunSweep).not.toHaveBeenCalled()
  })
})

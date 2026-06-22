// RunHistory 页 RTL 测试
//
// Feature: backtest-screening-run-history
// 覆盖：列表渲染 + 类别过滤（Property 5 路由前置）、点开回测→BacktestResults、
// 点开选股→只读榜单、点开 failed→错误堆栈且不渲染结果（Property 5）、
// 运行参数脱敏展示 ***（Property 4）、空列表空态（Req 1.4）。

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import { MemoryRouter } from 'react-router-dom'

// mock alphaService.listTasks：按 taskType 集合过滤（模拟后端多值过滤）。
// 另外补 getBarDataDetail（回测详情的 BacktestCharts 会拉 K 线）与 getDataResources（选股
// 详情复用 live 组件，import 链路涉及），保持离线确定性。
const mockListTasks = vi.fn()
const mockDeleteTask = vi.fn()
vi.mock('../../api/alpha', () => ({
  alphaService: {
    listTasks: (p: unknown) => mockListTasks(p),
    deleteTask: (id: string) => mockDeleteTask(id),
    getBarDataDetail: () => Promise.resolve({ preview: [] }),
    getDataResources: () =>
      Promise.resolve({ raw_bars: [], raw_ticks: [], derived_bars: [], raw_bar_intervals: [], derived_intervals: [] }),
  },
}))
// 选股详情复用 live 榜单 + Tier2DetailDrawer，其 import 链涉及这两个 service；mock 防真实请求。
vi.mock('../../api/cnn', () => ({ cnnService: { listModels: vi.fn().mockResolvedValue([]) } }))
vi.mock('../../api/screening', () => ({ screeningService: { getScreeningReport: vi.fn() } }))

import RunHistory from './index'

const backtestTask = {
  task_id: 'bt1', type: 'cnn_backtest', title: '回测任务A', entity_type: '', entity_name: '',
  status: 'completed', progress: 100, message: '',
  result: {
    statistics: { total_return: 0.12, annual_return: 0.2, sharpe_ratio: 1.5, max_ddpercent: -0.08, total_trade_count: 10, total_net_pnl: 1200, end_balance: 1120000 },
    target_symbol: '600519.SSE',
    equity_curve: [
      { date: '2025-01-01', balance: 1000000, drawdown: 0, ddpercent: 0, net_pnl: 0 },
      { date: '2025-06-20', balance: 1120000, drawdown: 0, ddpercent: 0, net_pnl: 120000 },
    ],
    trades: [],
  },
  created_at: '2026-06-22T10:00:00', updated_at: '2026-06-22T10:01:00', finished_at: '2026-06-22T10:01:00',
  duration_ms: 5000, params: { name: 'bt', token: '***' },
}
const screeningTask = {
  task_id: 'sc1', type: 'cnn_screening', title: '选股任务B', entity_type: '', entity_name: '',
  status: 'completed', progress: 100, message: '',
  result: { run_id: 'scr_x', status: 'draft', universe_size: 1, excluded: [], effective_right_bound: null, eval_window: null, leaderboard: [
    { rank: 1, tier1: { vt_symbol: '600519.SSE', fitness_score: 0.82, contributions: [], overall_confidence: 'high', available: true, note: null }, promoted_to_tier2: true, tier2: { vt_symbol: '600519.SSE', evaluable: true, edge_ok: true, avg_score: 0.12, pos_fold_ratio: 0.75, avg_cross_seed_std: null, report_id: 'r', note: null } },
  ] },
  created_at: '2026-06-22T09:00:00', updated_at: '2026-06-22T09:02:00', finished_at: '2026-06-22T09:02:00',
  duration_ms: 120000, params: { name: 'sc' },
}
const failedTask = {
  task_id: 'f1', type: 'backtest_run', title: '回测任务C', entity_type: '', entity_name: '',
  status: 'failed', progress: 0, message: 'boom failure', result: null,
  error_traceback: 'Traceback (most recent call last):\n  ValueError: boom',
  created_at: '2026-06-22T08:00:00', updated_at: '2026-06-22T08:00:30', finished_at: '2026-06-22T08:00:30',
  duration_ms: 30000,
}
const ALL = [backtestTask, screeningTask, failedTask]

beforeEach(() => {
  mockListTasks.mockReset()
  mockDeleteTask.mockReset()
  mockDeleteTask.mockResolvedValue({ success: true, task_id: 'bt1', purged: { reports: 0, models: 0, result_files: 0 } })
  mockListTasks.mockImplementation((params: { taskType?: string | string[] } | undefined) => {
    const t = params?.taskType
    const set = new Set<string>(Array.isArray(t) ? t : t ? [t] : [])
    return Promise.resolve(set.size === 0 ? ALL : ALL.filter((x) => set.has(x.type)))
  })
})

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <MemoryRouter>
          <RunHistory />
        </MemoryRouter>
      </AntApp>
    </QueryClientProvider>,
  )
}

/** 点击类别 Segmented 中的某个选项（避开与行内"类别"Tag 同名）。 */
async function pickCategory(user: ReturnType<typeof userEvent.setup>, label: string) {
  const seg = document.querySelector('.ant-segmented') as HTMLElement
  await user.click(within(seg).getByText(label))
}

describe('RunHistory 页', () => {
  it('列出回测与选股运行，类别过滤切换正确', async () => {
    const user = userEvent.setup()
    renderPage()

    // 默认"全部"→ 三条都在
    expect(await screen.findByText('回测任务A')).toBeInTheDocument()
    expect(screen.getByText('选股任务B')).toBeInTheDocument()
    expect(screen.getByText('回测任务C')).toBeInTheDocument()

    // 切"选股"→ 只剩选股
    await pickCategory(user, '选股')
    await waitFor(() => expect(screen.queryByText('回测任务A')).not.toBeInTheDocument())
    expect(screen.getByText('选股任务B')).toBeInTheDocument()

    // 切"回测"→ 回测在、选股不在
    await pickCategory(user, '回测')
    await waitFor(() => expect(screen.getByText('回测任务A')).toBeInTheDocument())
    expect(screen.queryByText('选股任务B')).not.toBeInTheDocument()
  })

  it('点开回测运行 → 展示完整结果（统计 + 净值曲线，与刚跑完一样）', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByText('回测任务A'))

    // BacktestResults 统计卡 + BacktestCharts 净值曲线（复用 live 回测组件）
    expect(await screen.findByText('夏普比率')).toBeInTheDocument()
    expect(screen.getByText('总收益')).toBeInTheDocument()
    expect(await screen.findByText('净值 / 回撤曲线')).toBeInTheDocument()
  })

  it('点开选股运行 → 展示完整榜单（含查看详情，与刚选完一样）', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByText('选股任务B'))

    // 复用 live 榜单：标的行 + 每行"查看详情"钻取按钮（Tier-2 折级实证）
    expect(await screen.findByText('600519.SSE')).toBeInTheDocument()
    expect(screen.getAllByText('查看详情').length).toBeGreaterThan(0)
    // 选股详情不应渲染回测统计
    expect(screen.queryByText('夏普比率')).not.toBeInTheDocument()
  })

  it('点开失败运行 → 展示错误堆栈，不渲染结果', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByText('回测任务C'))

    expect(await screen.findByText('该次运行失败')).toBeInTheDocument()
    expect(screen.getByText('错误堆栈')).toBeInTheDocument()
    expect(screen.queryByText('夏普比率')).not.toBeInTheDocument()
  })

  it('运行参数以脱敏值展示（*** 不回退为明文）', async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByText('回测任务A'))

    // 展开"运行参数"折叠
    await user.click(await screen.findByText('运行参数'))
    expect(
      await screen.findByText((c) => c.includes('***')),
    ).toBeInTheDocument()
  })

  it('无运行历史时渲染空态', async () => {
    mockListTasks.mockResolvedValue([])
    renderPage()
    expect(await screen.findByText('暂无运行历史')).toBeInTheDocument()
  })

  it('删除一条运行 → 确认后调用 deleteTask', async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByText('回测任务A')

    // 第一行（回测任务A，completed）的"删除" → Popconfirm → 确认
    const deleteLinks = screen.getAllByText('删除')
    await user.click(deleteLinks[0])
    // Popconfirm 确认按钮文案为"删除"（antd 会在两个汉字间插空格，故用正则匹配）
    const confirmBtn = await screen.findByRole('button', { name: /删\s*除/ })
    await user.click(confirmBtn)

    await waitFor(() => expect(mockDeleteTask).toHaveBeenCalledTimes(1))
    expect(mockDeleteTask).toHaveBeenCalledWith('bt1')
  })
})

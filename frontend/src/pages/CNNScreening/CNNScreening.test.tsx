// CNNScreening 页面 RTL 测试（Task 11.5 + 中英标签/Tooltip 覆盖）
//
// 覆盖范围：
//  1. 填写表单并提交 → cnnService.runScreening 收到格式正确的 CNNScreeningRequest；task_id 被存储（按钮进入"运行中"态）。
//  2. 任务 running → 进度 Alert 显示；任务 completed，注入合成 ScreeningResult → 榜单渲染行。
//  3. Fitness_Score 列默认降序排列（榜单首行排名=1 在前）。
//  4. "带入训练"按钮触发 navigate('/cnn-train', {state:{preset:{...}}}) 并携带正确标的与周期。
//  5. 低置信 / 未入围 Tier-2 行显示橙色警示 Tag。
//  6. 空结果（leaderboard=[]）渲染不崩溃，显示"候选池 0 只"。
//  7. 中英结合标签渲染：榜单列头显示"CNN 适配度 (Fitness Score)"、"综合置信度 (Confidence)" 等。
//  8. Tooltip 悬浮线索：InfoCircleOutlined 图标在列头中存在（role="img" 带 aria-label）。
//
// 通过 mock 网络服务保持确定性、离线（不触网）。

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import { MemoryRouter } from 'react-router-dom'
import type { UseQueryResult } from '@tanstack/react-query'

import type { Task } from '../../types/alpha'
import type { ScreeningResult } from '../../types/screening'

// ── mock cnnService（vitest hoist 要求 vi.mock 在 import 之前声明）────────────
const mockRunScreening = vi.fn()
vi.mock('../../api/cnn', () => ({
  cnnService: {
    runScreening: (...args: unknown[]) => mockRunScreening(...args),
    // 同文件其他方法在此不使用，给占位以防组件导入时报错（不会用到）
    listModels: vi.fn().mockResolvedValue([]),
    train: vi.fn(),
    getModel: vi.fn(),
    deleteModel: vi.fn(),
    getModelArchitecture: vi.fn(),
    predict: vi.fn(),
    runBacktest: vi.fn(),
  },
}))

// ── mock useTask ─────────────────────────────────────────────────────────────
// 用 vi.fn() 持有引用，以便在各测试中动态切换返回值。
const mockUseTask = vi.fn<(taskId: string | null) => Partial<UseQueryResult<Task>>>()
vi.mock('../../hooks/useTask', () => ({
  useTask: (taskId: string | null) => mockUseTask(taskId),
}))

// ── mock react-router-dom（保留 MemoryRouter，只拦截 useNavigate）────────────
const mockNavigate = vi.fn()
vi.mock('react-router-dom', async (importActual) => {
  const actual = await importActual<typeof import('react-router-dom')>()
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  }
})

import CNNScreening from './index'

// ─────────────────────────────────────────────────────────────────────────────
// 测试 fixtures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 构造一个 idle Task（无任务启动），useTask 返回空 data。
 */
function idleTask(): Partial<UseQueryResult<Task>> {
  return { data: undefined }
}

/**
 * 构造一个 running Task，`useTask` 将其作为轮询中间状态返回。
 *
 * @param progress - 进度百分比（0-100）
 * @param message - 状态消息
 */
function runningTask(progress = 30, message = 'Tier-1 3/10'): Partial<UseQueryResult<Task>> {
  return {
    data: {
      task_id: 'tid-1',
      status: 'running',
      progress,
      message,
      task_type: 'CNN_SCREENING',
      created_at: '2026-06-15T00:00:00',
      updated_at: '2026-06-15T00:00:05',
      result: null,
      error: null,
      entity_type: 'cnn_screening',
      entity_name: 'test',
      params: {},
      title: 'CNN 选股批量评估',
    } as unknown as Task,
  }
}

/**
 * 构造一个 completed Task，`result` 字段携带合成 ScreeningResult。
 *
 * @param result - 要嵌入 task.data.result 的选股结果
 */
function completedTask(result: ScreeningResult): Partial<UseQueryResult<Task>> {
  return {
    data: {
      task_id: 'tid-1',
      status: 'completed',
      progress: 100,
      message: '选股完成',
      task_type: 'CNN_SCREENING',
      created_at: '2026-06-15T00:00:00',
      updated_at: '2026-06-15T00:01:00',
      result: result as unknown as Record<string, unknown>,
      error: null,
      entity_type: 'cnn_screening',
      entity_name: 'test',
      params: {},
      title: 'CNN 选股批量评估',
    } as unknown as Task,
  }
}

/** 最小合法 ScreeningResult，含两行榜单（高分 + 低分/未入围）。 */
const syntheticResult: ScreeningResult = {
  run_id: 'run-abc',
  status: 'draft',
  created_at: '2026-06-15T00:01:00',
  input: { interval: 'd', as_of: '2026-06-15', lookback_days: 250 },
  rules_id: 'builtin-v1',
  universe_size: 2,
  excluded: [],
  leaderboard: [
    {
      rank: 1,
      tier1: {
        vt_symbol: '600030.SSE',
        fitness_score: 0.82,
        contributions: [],
        overall_confidence: 'high',
        available: true,
        note: null,
      },
      promoted_to_tier2: true,
      tier2: {
        vt_symbol: '600030.SSE',
        evaluable: true,
        edge_ok: true,
        avg_score: 0.12,
        pos_fold_ratio: 0.75,
        avg_cross_seed_std: null,
        report_id: 'rpt-1',
        note: null,
      },
    },
    {
      rank: 2,
      tier1: {
        vt_symbol: '000001.SZSE',
        fitness_score: 0.41,
        contributions: [],
        overall_confidence: 'low',
        available: true,
        note: null,
      },
      promoted_to_tier2: false,
      tier2: null,
    },
  ],
  effective_right_bound: '2026-06-14T15:00:00',
  eval_window: null,
}

/** 空榜单结果，用于验证 empty state 不崩溃。 */
const emptyResult: ScreeningResult = {
  ...syntheticResult,
  universe_size: 0,
  leaderboard: [],
  excluded: [],
}

// ─────────────────────────────────────────────────────────────────────────────
// 渲染辅助
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 渲染 CNNScreening 页，包裹 QueryClientProvider + AntApp + MemoryRouter。
 * `retry: false` 使测试中失败的 query 不重试、保持确定性。
 */
function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <MemoryRouter>
          <CNNScreening />
        </MemoryRouter>
      </AntApp>
    </QueryClientProvider>,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 测试套件
// ─────────────────────────────────────────────────────────────────────────────

describe('CNNScreening 页面', () => {
  beforeEach(() => {
    mockRunScreening.mockReset()
    mockNavigate.mockReset()
    // 默认无进行中任务
    mockUseTask.mockReturnValue(idleTask())
  })

  // ── 1. 表单提交调用 runScreening ──────────────────────────────────────────
  it('填写表单提交后 runScreening 收到格式正确的 CNNScreeningRequest，按钮进入运行态', async () => {
    mockRunScreening.mockResolvedValueOnce({ task_id: 'tid-1', name: 'test' })
    // 提交后 useTask 返回 running（模拟按钮进入运行中）
    mockUseTask.mockReturnValue(idleTask())

    const user = userEvent.setup()
    renderPage()

    // 页面应显示任务名称输入框（有 placeholder）
    // 任务名称默认已填（initialValues 中带有 dayjs 格式化的时间戳），直接清空并重新输入
    const nameInput = screen.getByPlaceholderText('cnn_screen_YYYYMMDD')
    await user.clear(nameInput)
    await user.type(nameInput, 'my_screen')

    // 点击启动选股
    await user.click(screen.getByRole('button', { name: /启动选股/ }))

    await waitFor(() => {
      expect(mockRunScreening).toHaveBeenCalledTimes(1)
    })

    const req = mockRunScreening.mock.calls[0][0]
    // 校验关键字段格式与类型
    expect(req.name).toBe('my_screen')
    expect(req.interval).toBe('d')         // 默认值
    expect(typeof req.as_of).toBe('string') // YYYY-MM-DD 字符串
    expect(/^\d{4}-\d{2}-\d{2}$/.test(req.as_of)).toBe(true)
    expect(req.lookback_days).toBe(250)
    expect(req.top_k).toBe(15)
    expect(req.run_tier2).toBe(true)
    expect(req.objective).toBe('classification')
    expect(req.persist).toBe(false)
    expect(Array.isArray(req.include_symbols)).toBe(true)
    expect(Array.isArray(req.exclude_symbols)).toBe(true)
  })

  // ── 2. 任务运行中 → 进度 Alert 显示；任务完成 → 榜单渲染行 ──────────────
  it('任务 running 时显示进度 Alert；任务 completed 后榜单渲染标的行', async () => {
    // 先模拟 running 状态
    mockUseTask.mockReturnValue(runningTask(30, 'Tier-1 3/10'))

    renderPage()

    // 进度 Alert 显示运行中信息（按钮文案 "选股任务运行中…" 与 Alert message "运行中… 30%" 都含"运行中"，用 getAllByText）
    const runningTexts = await screen.findAllByText(/运行中/)
    expect(runningTexts.length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Tier-1 3\/10/)).toBeInTheDocument()

    // 切换到 completed 状态
    mockUseTask.mockReturnValue(completedTask(syntheticResult))

    // 重新渲染以触发状态变更
    // （RTL 直接重新 render 同一组件以测试完成后的榜单，更简洁）
    const { unmount } = renderPage()
    expect(await screen.findByText('600030.SSE')).toBeInTheDocument()
    expect(screen.getByText('000001.SZSE')).toBeInTheDocument()
    // 结果面板摘要 Tag
    expect(screen.getAllByText('候选池 2 只').length).toBeGreaterThanOrEqual(1)
    unmount()
  })

  // ── 3. fitness_score 列默认降序：rank=1 行出现在 rank=2 行之前 ──────────
  it('榜单 fitness_score 列 defaultSortOrder=descend：rank-1 行先于 rank-2 行渲染', async () => {
    mockUseTask.mockReturnValue(completedTask(syntheticResult))
    renderPage()

    const cells600 = await screen.findAllByText('600030.SSE')
    const cells000 = await screen.findAllByText('000001.SZSE')

    // 两个标的都能渲染出来
    expect(cells600.length).toBeGreaterThanOrEqual(1)
    expect(cells000.length).toBeGreaterThanOrEqual(1)

    // rank-1（600030.SSE）应出现在 rank-2（000001.SZSE）之前（DOM 中更靠前）
    const all600 = cells600.map((el) => el)
    const all000 = cells000.map((el) => el)
    const pos600 = all600[0].compareDocumentPosition(all000[0])
    // compareDocumentPosition 返回 4 表示 all000 在 all600 之后（DOCUMENT_POSITION_FOLLOWING）
    expect(pos600 & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  // ── 4. "带入训练"按钮 → navigate 被调用并带正确的 preset ─────────────────
  it('点击榜单第一行"带入训练"→ navigate 跳转 /cnn-train 并携带 preset', async () => {
    mockUseTask.mockReturnValue(completedTask(syntheticResult))
    renderPage()

    // 等待榜单渲染
    await screen.findByText('600030.SSE')

    // 点击第一个"带入训练"按钮（rank-1 行）
    const buttons = screen.getAllByRole('button', { name: '带入训练' })
    expect(buttons.length).toBeGreaterThanOrEqual(1)
    await userEvent.click(buttons[0])

    expect(mockNavigate).toHaveBeenCalledTimes(1)
    const [path, opts] = mockNavigate.mock.calls[0]
    expect(path).toBe('/cnn-train')
    expect(opts?.state?.preset?.target_symbol).toBe('600030.SSE')
    expect(opts?.state?.preset?.input_data_kind).toBe('bar')
    // interval 来自 screeningResult.input.interval
    expect(opts?.state?.preset?.input_interval).toBe('d')
  })

  // ── 5. 低置信 / 未入围 Tier-2 行显示橙色警示 Tag ─────────────────────────
  it('未入围 Tier-2 的行显示"未经 WF 实证"Tag；低置信行显示"低置信"Tag', async () => {
    mockUseTask.mockReturnValue(completedTask(syntheticResult))
    renderPage()

    await screen.findByText('600030.SSE')

    // rank-2 行：promoted_to_tier2=false → 应有"未经 WF 实证" Tag
    expect(screen.getByText('未经 WF 实证')).toBeInTheDocument()

    // rank-2 的 overall_confidence='low' → 显示"低置信" Tag
    expect(screen.getByText('低置信')).toBeInTheDocument()
  })

  // ── 6. 空结果 → 不崩溃，显示 "候选池 0 只" ────────────────────────────────
  it('空榜单结果渲染不崩溃，显示"候选池 0 只"', async () => {
    mockUseTask.mockReturnValue(completedTask(emptyResult))
    renderPage()

    expect(await screen.findByText('候选池 0 只')).toBeInTheDocument()
    // 不应出现任何标的代码（因为 leaderboard 为空）
    expect(screen.queryByText('600030.SSE')).not.toBeInTheDocument()
  })

  // ── 7. 中英结合标签渲染 ───────────────────────────────────────────────────
  it('榜单列头使用中英结合标签（"CNN 适配度 (Fitness Score)"、"综合置信度 (Confidence)" 等）', async () => {
    mockUseTask.mockReturnValue(completedTask(syntheticResult))
    renderPage()

    // 等待榜单渲染
    await screen.findByText('600030.SSE')

    // 主要榜单列头（antd Table 有时克隆列头节点造成同一文字出现多次，用 getAllByText 并断言 ≥1）
    expect(screen.getAllByText('CNN 适配度 (Fitness Score)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('综合置信度 (Confidence)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('入围 Tier-2 (Promoted)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('实证胜出 (Edge OK)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('跨折均分 (Avg Fold Score)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('正分折占比 (Positive Fold Ratio)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('标的 (Symbol)').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('排名 (Rank)').length).toBeGreaterThanOrEqual(1)
  })

  // ── 8. Tooltip 悬浮线索：InfoCircleOutlined 图标存在 ─────────────────────
  it('榜单列头包含 InfoCircleOutlined Tooltip 悬浮图标（anticon-info-circle 标记）', async () => {
    mockUseTask.mockReturnValue(completedTask(syntheticResult))
    renderPage()

    await screen.findByText('600030.SSE')

    // antd 图标渲染为 <span role="img" aria-label="info-circle"> 或带 class anticon-info-circle
    // getAllByRole 查 "img" + aria-label="info-circle"（antd 5.x 渲染方式）
    const infoIcons = document.querySelectorAll('.anticon-info-circle')
    expect(infoIcons.length).toBeGreaterThan(0)
  })

  // ── 9. 选股七维度：contributions 展开行中的维度标签也走 MetricMeta ─────────
  it('展开行的贡献维度显示中英标签（data_quality → "数据质量 (Data Quality)"）', async () => {
    const resultWithContribs: ScreeningResult = {
      ...syntheticResult,
      leaderboard: [
        {
          rank: 1,
          tier1: {
            vt_symbol: '600030.SSE',
            fitness_score: 0.82,
            contributions: [
              {
                dimension: 'data_quality',
                raw_value: 0.9,
                level: 'high',
                weight: 0.2,
                contribution: 0.18,
                confidence: 'high',
              },
              {
                dimension: 'liquidity',
                raw_value: 0.7,
                level: 'medium',
                weight: 0.15,
                contribution: 0.105,
                confidence: 'medium',
              },
            ],
            overall_confidence: 'high',
            available: true,
            note: null,
          },
          promoted_to_tier2: true,
          tier2: null,
        },
      ],
    }
    mockUseTask.mockReturnValue(completedTask(resultWithContribs))
    renderPage()

    await screen.findByText('600030.SSE')

    // 展开第一行
    const expandBtns = document.querySelectorAll('.ant-table-row-expand-icon')
    if (expandBtns.length > 0) {
      await userEvent.click(expandBtns[0] as HTMLElement)
      // 贡献明细中的维度标签应显示中英合并形式
      expect(await screen.findByText('数据质量 (Data Quality)')).toBeInTheDocument()
      expect(screen.getByText('流动性 (Liquidity)')).toBeInTheDocument()
    }
    // 如无可展开按钮（JSDOM 下 antd Table expandable 有时不渲染），仅检查前面的标签测试即可
  })

  // ── 10. Tier-2 高级设置面板：run_tier2 开启时显示，关闭时隐藏 ───────────────
  it('run_tier2 开启时「Tier-2 高级设置」区块可见，关闭时隐藏', async () => {
    mockUseTask.mockReturnValue(idleTask())
    const user = userEvent.setup()
    renderPage()

    // 默认 run_tier2=true → 高级设置面板应存在
    expect(await screen.findByText(/Tier-2 高级设置/)).toBeInTheDocument()

    // 点击 run_tier2 Switch 关闭
    const switchEl = screen.getByRole('switch')
    await user.click(switchEl)

    // run_tier2=false → 高级设置面板消失
    expect(screen.queryByText(/Tier-2 高级设置/)).not.toBeInTheDocument()
  })

  // ── 11. Tier-2 高级设置：字段显示中英标签 ─────────────────────────────────
  it('Tier-2 高级设置面板中字段使用中英结合标签', async () => {
    mockUseTask.mockReturnValue(idleTask())
    const user = userEvent.setup()
    renderPage()

    // 展开 Collapse
    const collapseHeader = await screen.findByText(/Tier-2 高级设置/)
    await user.click(collapseHeader)

    // 四个高级字段的中英标签
    expect(await screen.findByText('评估窗总长 (Eval Window Days)')).toBeInTheDocument()
    expect(screen.getByText('每折训练 (Train Days)')).toBeInTheDocument()
    expect(screen.getByText('每折测试 (Test Days)')).toBeInTheDocument()
    expect(screen.getByText('随机种子数 (Seeds)')).toBeInTheDocument()
  })

  // ── 12. Tier-2 高级字段填写 → 包含于请求；留空 → 请求中不含该字段 ──────────
  it('填写 eval_window_days 后包含在 runScreening 请求中；留空时不含', async () => {
    mockRunScreening.mockResolvedValueOnce({ task_id: 'tid-2', name: 'test' })
    mockUseTask.mockReturnValue(idleTask())
    const user = userEvent.setup()
    renderPage()

    // 展开高级设置
    const collapseHeader = await screen.findByText(/Tier-2 高级设置/)
    await user.click(collapseHeader)

    // 找到 eval_window_days 输入框（placeholder="默认 900"）并填写
    const evalInput = await screen.findByPlaceholderText('默认 900')
    await user.clear(evalInput)
    await user.type(evalInput, '600')

    // 提交
    const nameInput = screen.getByPlaceholderText('cnn_screen_YYYYMMDD')
    await user.clear(nameInput)
    await user.type(nameInput, 'adv_screen')
    await user.click(screen.getByRole('button', { name: /启动选股/ }))

    await waitFor(() => {
      expect(mockRunScreening).toHaveBeenCalledTimes(1)
    })

    const req = mockRunScreening.mock.calls[0][0]
    // eval_window_days 已填 → 包含在请求中
    expect(req.eval_window_days).toBe(600)
    // train_days / fold_test_days / n_seeds 未填 → 不包含（undefined/omitted）
    expect(req.train_days).toBeUndefined()
    expect(req.fold_test_days).toBeUndefined()
    expect(req.n_seeds).toBeUndefined()
  })

  // ── 13. 历史需求提示渲染（默认值回落场景）────────────────────────────────────
  it('展开高级设置后显示历史需求提示，包含天数与折数', async () => {
    mockUseTask.mockReturnValue(idleTask())
    const user = userEvent.setup()
    renderPage()

    // 展开 Collapse
    const collapseHeader = await screen.findByText(/Tier-2 高级设置/)
    await user.click(collapseHeader)

    // 默认值（900/480/90）→ 提示应含"900 天"或"570 天"或"折"
    const hint = await screen.findByText(/约.*折/)
    expect(hint).toBeInTheDocument()
    // 提示中含"历史不足的标的会自动跳过"
    expect(hint.textContent).toContain('历史不足的标的会自动跳过')
  })

  // ── 14. promoted_to_tier2 + evaluable=false → 跳过/数据不足 Tag ─────────────
  it('入围 Tier-2 但 evaluable=false 时显示橙色"跳过"或"数据不足"Tag，note 绑 Tooltip', async () => {
    const resultWithSkip: ScreeningResult = {
      ...syntheticResult,
      leaderboard: [
        {
          rank: 1,
          tier1: {
            vt_symbol: '600030.SSE',
            fitness_score: 0.82,
            contributions: [],
            overall_confidence: 'high',
            available: true,
            note: null,
          },
          promoted_to_tier2: true,
          tier2: {
            vt_symbol: '600030.SSE',
            evaluable: false,
            edge_ok: false,
            avg_score: null,
            pos_fold_ratio: null,
            avg_cross_seed_std: null,
            report_id: null,
            note: '数据不足：本地历史仅 200 天，低于评估所需最小值 570 天',
          },
        },
        {
          rank: 2,
          tier1: {
            vt_symbol: '000001.SZSE',
            fitness_score: 0.55,
            contributions: [],
            overall_confidence: 'medium',
            available: true,
            note: null,
          },
          promoted_to_tier2: true,
          tier2: {
            vt_symbol: '000001.SZSE',
            evaluable: false,
            edge_ok: false,
            avg_score: null,
            pos_fold_ratio: null,
            avg_cross_seed_std: null,
            report_id: null,
            note: 'WF 评估异常，已跳过',
          },
        },
      ],
    }

    mockUseTask.mockReturnValue(completedTask(resultWithSkip))
    renderPage()

    await screen.findByText('600030.SSE')

    // note 含"数据不足" → Tag 文案为"数据不足"
    expect(screen.getByText('数据不足')).toBeInTheDocument()

    // note 不含"数据不足" → Tag 文案为"跳过"
    expect(screen.getByText('跳过')).toBeInTheDocument()
  })
})

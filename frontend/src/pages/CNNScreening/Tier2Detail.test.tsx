// Tier-2 折级详情：GateVerdictHeader / FoldTable / Tier2DetailDrawer 的 RTL 测试。
//
// 覆盖（Feature: cnn-screening-tier2-detail）：
//  - GateVerdictHeader（Property 4）：edge_ok 两条件忠实映射；evaluable=false → 不可评估。
//  - FoldTable：渲染折行、点行回调 onSelect；无生产模型隐藏生产对照列、有则显示。
//  - Tier2DetailDrawer（Property 5 / R4.4 / R4.8）：拉报告渲染全部折、点行切换指标卡、404 空态。
//
// 通过 mock screeningService 保持确定性、离线。

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'

import type { BacktestStatistics } from '../../types/alpha'
import type { ScreeningFold, ScreeningWfReport, Tier2Verdict } from '../../types/screening'

// ── mock screeningService（vitest hoist：vi.mock 在被测组件 import 之前）────────
const mockGetReport = vi.fn()
vi.mock('../../api/screening', () => ({
  screeningService: {
    getScreeningReport: (...args: unknown[]) => mockGetReport(...args),
  },
}))

// ── mock alphaService.getBarDataDetail（抽屉里 BacktestCharts 拉 K 线 OHLC）──────
// 返回空 preview → K 线图渲染空态、不触网，保持确定性。用具名 mock 以断言拉行情时的周期实参。
const mockGetBarDataDetail = vi.fn().mockResolvedValue({ preview: [] })
vi.mock('../../api/alpha', () => ({
  alphaService: {
    getBarDataDetail: (...args: unknown[]) => mockGetBarDataDetail(...args),
  },
}))

import GateVerdictHeader from './GateVerdictHeader'
import FoldTable from './FoldTable'
import Tier2DetailDrawer from './Tier2DetailDrawer'

// ─────────────────────────────────────────────────────────────────────────────
// fixtures
// ─────────────────────────────────────────────────────────────────────────────

function mkVerdict(overrides: Partial<Tier2Verdict> = {}): Tier2Verdict {
  return {
    vt_symbol: '600030.SSE',
    evaluable: true,
    edge_ok: true,
    avg_score: 0.12,
    pos_fold_ratio: 0.75,
    avg_cross_seed_std: 0.05,
    report_id: 'rpt-1',
    note: null,
    ...overrides,
  }
}

function mkFold(
  i: number,
  stats: Partial<BacktestStatistics> = {},
  opts: { production_score?: number } = {},
): ScreeningFold {
  return {
    fold: i,
    train: { start: '2024-01-01', end: '2024-05-31' },
    test: { start: '2024-06-01', end: '2024-08-30' },
    candidate_model: `m${i}`,
    candidate_models: [`m${i}`],
    candidate_statistics: { total_return: 5, sharpe_ratio: 1, ...stats } as BacktestStatistics,
    candidate_equity_curve: [],
    candidate_trades: [],
    candidate_seed_statistics: [{ total_return: 5 } as BacktestStatistics],
    candidate_seed_scores: [0.5],
    candidate_score: 0.5,
    cross_seed: { mean: 0.5, std: 0.04, n: 1 },
    production_model: opts.production_score != null ? 'p' : null,
    production_statistics: null,
    production_score: opts.production_score ?? null,
    score_delta: opts.production_score != null ? 0.1 : null,
  }
}

function mkReport(folds: ScreeningFold[]): ScreeningWfReport {
  return {
    report_id: 'rpt-1',
    type: 'walk_forward',
    name: 'x',
    created_at: '2025-06-22T15:30:12',
    request: {},
    production_model: null,
    folds,
    summary: {
      fold_count: folds.length,
      candidate_win_count: 0,
      candidate_win_rate: 0,
      avg_score_delta: null,
      n_seeds: 1,
      avg_cross_seed_std: 0.04,
      passed: false,
      reasons: [],
    },
  }
}

function renderDrawer(verdict: Tier2Verdict | null) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <Tier2DetailDrawer open verdict={verdict} onClose={vi.fn()} />
      </AntApp>
    </QueryClientProvider>,
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// GateVerdictHeader（Property 4）
// ─────────────────────────────────────────────────────────────────────────────

describe('GateVerdictHeader', () => {
  it('edge_ok 通过：两条件均为成功 Tag', () => {
    render(<AntApp><GateVerdictHeader verdict={mkVerdict()} /></AntApp>)
    expect(screen.getByText('✓ edge_ok 通过')).toBeInTheDocument()
    // 两条件 chip（平均分>0、正分折≥阈值）均通过 → 两个 ant-tag-success
    expect(document.querySelectorAll('.ant-tag-success').length).toBe(2)
    expect(screen.getByText(/平均分/)).toBeInTheDocument()
    expect(screen.getByText(/正分折/)).toBeInTheDocument()
  })

  it('edge_ok 未通过：平均分≤0 且正分折不足 → 两条件均为失败 Tag', () => {
    render(
      <AntApp>
        <GateVerdictHeader verdict={mkVerdict({ edge_ok: false, avg_score: -0.1, pos_fold_ratio: 0.4 })} />
      </AntApp>,
    )
    expect(screen.getByText('✗ edge_ok 未通过')).toBeInTheDocument()
    expect(document.querySelectorAll('.ant-tag-error').length).toBe(2)
  })

  it('evaluable=false：渲染不可评估并附 note', () => {
    render(
      <AntApp>
        <GateVerdictHeader verdict={mkVerdict({ evaluable: false, report_id: null, note: '数据不足，跳过 Tier-2' })} />
      </AntApp>,
    )
    expect(screen.getByText('不可评估')).toBeInTheDocument()
    expect(screen.getByText('数据不足，跳过 Tier-2')).toBeInTheDocument()
  })
})

// ─────────────────────────────────────────────────────────────────────────────
// FoldTable
// ─────────────────────────────────────────────────────────────────────────────

describe('FoldTable', () => {
  it('渲染折行，点行触发 onSelect', async () => {
    const onSelect = vi.fn()
    render(
      <AntApp>
        <FoldTable folds={[mkFold(0), mkFold(1)]} selectedFold={0} onSelect={onSelect} />
      </AntApp>,
    )
    expect(screen.getByText('#0')).toBeInTheDocument()
    await userEvent.click(screen.getByText('#1'))
    expect(onSelect).toHaveBeenCalledWith(1)
  })

  it('无生产模型时隐藏生产对照列，有则显示', () => {
    const { unmount } = render(
      <AntApp>
        <FoldTable folds={[mkFold(0)]} selectedFold={0} onSelect={vi.fn()} />
      </AntApp>,
    )
    expect(screen.queryByText('生产分 (Prod Score)')).not.toBeInTheDocument()
    unmount()

    render(
      <AntApp>
        <FoldTable folds={[mkFold(0, {}, { production_score: 0.3 })]} selectedFold={0} onSelect={vi.fn()} />
      </AntApp>,
    )
    expect(screen.getByText('生产分 (Prod Score)')).toBeInTheDocument()
    expect(screen.getByText('分差 (Δ Score)')).toBeInTheDocument()
  })
})

// ─────────────────────────────────────────────────────────────────────────────
// Tier2DetailDrawer（Property 5 / R4.4 / R4.8）
// ─────────────────────────────────────────────────────────────────────────────

describe('Tier2DetailDrawer', () => {
  beforeEach(() => {
    mockGetReport.mockReset()
    mockGetBarDataDetail.mockClear()
  })

  it('拉报告渲染全部折，点折行切换回测指标卡', async () => {
    mockGetReport.mockResolvedValueOnce(
      mkReport([
        mkFold(0, { start_date: '2024-06-02' }),
        mkFold(1, { start_date: '2024-09-02' }),
      ]),
    )
    renderDrawer(mkVerdict())

    // 折级表渲染全部折
    expect(await screen.findByText('#0')).toBeInTheDocument()
    expect(screen.getByText('#1')).toBeInTheDocument()
    // 默认选中首折 → BacktestResults 显示折0 的开始日期
    expect(await screen.findByText('2024-06-02')).toBeInTheDocument()

    // 点折1行 → 指标卡切换到折1
    await userEvent.click(screen.getByText('#1'))
    expect(await screen.findByText('2024-09-02')).toBeInTheDocument()
    expect(mockGetReport).toHaveBeenCalledWith('rpt-1')
  })

  it('报告加载失败 → 空态占位，不崩溃', async () => {
    mockGetReport.mockRejectedValueOnce(new Error('404'))
    renderDrawer(mkVerdict())
    // 门禁头部仍以 verdict 渲染
    expect(await screen.findByText('✓ edge_ok 通过')).toBeInTheDocument()
    expect(await screen.findByText('报告不存在或加载失败')).toBeInTheDocument()
  })

  it('渲染折级净值曲线卡 + K线买卖点卡 + 成交明细（第三波）', async () => {
    mockGetReport.mockResolvedValueOnce(mkReport([mkFold(0, { start_date: '2024-06-02' })]))
    renderDrawer(mkVerdict())
    // BacktestCharts 复用：净值/回撤曲线卡 + K线买卖点卡
    expect(await screen.findByText('净值 / 回撤曲线')).toBeInTheDocument()
    expect(screen.getByText('K 线与买卖点')).toBeInTheDocument()
    // 折级成交明细折叠面板
    expect(screen.getByText(/成交明细/)).toBeInTheDocument()
  })

  it('K 线复用回测实际所用的 bar 周期（report.request.input_interval），而非默认 d', async () => {
    // 选股跑在 30m 线上、该标的无 d 行情：抽屉应按报告里回测的 30m 周期拉 OHLC，而非硬编码 d。
    const report = mkReport([mkFold(0, { start_date: '2024-06-02' })])
    report.request = { input_interval: '30m' }
    mockGetReport.mockResolvedValueOnce(report)
    renderDrawer(mkVerdict({ vt_symbol: '000415.SZSE' }))

    expect(await screen.findByText('K 线与买卖点')).toBeInTheDocument()
    await waitFor(() =>
      expect(mockGetBarDataDetail).toHaveBeenCalledWith('30m', '000415.SZSE'),
    )
    // 绝不应再用日线 d 去拉这只无 d 行情的标的
    expect(mockGetBarDataDetail).not.toHaveBeenCalledWith('d', '000415.SZSE')
  })
})

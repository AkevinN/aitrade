import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import dayjs from 'dayjs'

import type { SymbolProfileResponse } from '../../types/alpha'
import ProfilingPanel, { GroupProfileView, SuggestionView } from './ProfilingPanel'

const runProfiling = vi.fn()
const getProfilingArtifact = vi.fn()
const listProfilingArtifacts = vi.fn()
vi.mock('../../api/alpha', () => ({
  alphaService: {
    runProfiling: (req: unknown) => runProfiling(req),
    getProfilingArtifact: (id: string) => getProfilingArtifact(id),
    listProfilingArtifacts: () => listProfilingArtifacts(),
  },
}))

function renderWithClient(ui: React.ReactElement) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>{ui}</AntApp>
    </QueryClientProvider>,
  )
}

const profile: SymbolProfileResponse = {
  input: {
    vt_symbol: '600030.SSE',
    interval: '30m',
    as_of: '2024-01-01T00:00:00',
    lookback_days: 250,
    effective_right_bound: '2023-12-29T15:00:00',
    effective_bar_count: 120,
    rules_id: 'builtin-v1',
  },
  available: true,
  blocks: [
    {
      block: 'data_quality',
      metrics: [
        { key: 'count_valid_bars', value: 120, effective_sample: 120, confidence: 'high' },
        { key: 'gap_ratio', value: null, effective_sample: 0, confidence: 'insufficient', note: 'insufficient_sample' },
      ],
    },
    { block: 'liquidity', level: 'medium', metrics: [{ key: 'avg_turnover', value: 1, effective_sample: 120, confidence: 'high' }] },
    { block: 'volatility', level: 'low', metrics: [{ key: 'realized_volatility', value: 0.01, effective_sample: 120, confidence: 'medium' }] },
    { block: 'predictability', level: 'trending', metrics: [{ key: 'variance_ratio', value: 1.2, effective_sample: 120, confidence: 'medium' }] },
  ],
  group_profile: {
    target: '600030.SSE',
    members: ['000300.SSE'],
    alignment_coverage: 0.5,
    correlation_summary: { '000300.SSE': 0.02 },
  },
  suggestion: {
    status: 'draft',
    interval: '30m',
    vt_symbols: ['600030.SSE'],
    degraded: false,
    items: [
      { field: 'label_spec.mode', value: 'oco', reason: 'vol', based_on_confidence: 'high' },
      { field: 'label_spec.take_profit', value: 0.03, reason: 'vol', based_on_confidence: 'high' },
    ],
  },
  overall_confidence: 'medium',
  created_at: '2024-01-01T01:00:00',
  artifact_id: '600030.SSE__30m__20240101T000000',
}

describe('ProfilingPanel', () => {
  beforeEach(() => {
    runProfiling.mockReset()
    getProfilingArtifact.mockReset()
    listProfilingArtifacts.mockReset()
    listProfilingArtifacts.mockResolvedValue([])
  })

  it('blocks run when target symbol is missing', async () => {
    const user = userEvent.setup()
    renderWithClient(
      <ProfilingPanel
        open
        onClose={vi.fn()}
        targetSymbol=""
        interval="30m"
        defaultAsOf={dayjs('2024-01-01')}
        observationGroups={[]}
        onApplySuggestion={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /开始评估/ })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /开始评估/ }))
    expect(runProfiling).not.toHaveBeenCalled()
  })

  it('renders unavailable response as normal 200 result', async () => {
    runProfiling.mockResolvedValueOnce({
      ...profile,
      available: false,
      unavailable_reason: '窗口内无可用行情',
      blocks: [],
      suggestion: null,
      group_profile: null,
      overall_confidence: 'insufficient',
    })
    const user = userEvent.setup()
    renderWithClient(
      <ProfilingPanel
        open
        onClose={vi.fn()}
        targetSymbol="600030.SSE"
        interval="30m"
        defaultAsOf={dayjs('2024-01-01')}
        observationGroups={[]}
        onApplySuggestion={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /开始评估/ }))
    expect(await screen.findByText('画像数据不可用')).toBeInTheDocument()
    expect(screen.getByText('窗口内无可用行情')).toBeInTheDocument()
  })

  it('renders four blocks, null placeholder and group feedback', async () => {
    runProfiling.mockResolvedValueOnce(profile)
    const user = userEvent.setup()
    renderWithClient(
      <ProfilingPanel
        open
        onClose={vi.fn()}
        targetSymbol="600030.SSE"
        interval="30m"
        defaultAsOf={dayjs('2024-01-01')}
        observationGroups={[{ role: 'market', name: 'm', symbols: ['000300.SSE'] }]}
        onApplySuggestion={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /开始评估/ }))
    expect(await screen.findByText('数据质量')).toBeInTheDocument()
    expect(screen.getByText('流动性')).toBeInTheDocument()
    expect(screen.getByText('波动性')).toBeInTheDocument()
    expect(screen.getByText('可预测性')).toBeInTheDocument()
    expect(screen.getByText('insufficient_sample')).toBeInTheDocument()
    expect(screen.getByText('建议剔除 / 可能为噪声')).toBeInTheDocument()
    expect(screen.getByText('600030.SSE__30m__20240101T000000')).toBeInTheDocument()
  })

  it('loads historical artifact through the same render path', async () => {
    listProfilingArtifacts.mockResolvedValueOnce(['artifact-1'])
    getProfilingArtifact.mockResolvedValueOnce(profile)
    const user = userEvent.setup()
    renderWithClient(
      <ProfilingPanel
        open
        onClose={vi.fn()}
        targetSymbol="600030.SSE"
        interval="30m"
        defaultAsOf={dayjs('2024-01-01')}
        observationGroups={[]}
        onApplySuggestion={vi.fn()}
      />,
    )

    await user.click(screen.getByText('查看历史画像'))
    await waitFor(() => expect(listProfilingArtifacts).toHaveBeenCalled())
    const historyButton = screen.getByRole('button', { name: /history查看历史/ })
    await waitFor(() => expect(historyButton).toBeEnabled())
    await user.click(historyButton)
    await waitFor(() => expect(getProfilingArtifact).toHaveBeenCalledWith('artifact-1'))
    expect(await screen.findByText('历史画像 · 创建于 2024-01-01T01:00:00')).toBeInTheDocument()
  })
})

describe('SuggestionView and GroupProfileView', () => {
  beforeEach(() => {
    runProfiling.mockReset()
    getProfilingArtifact.mockReset()
    listProfilingArtifacts.mockReset()
    listProfilingArtifacts.mockResolvedValue([])
  })

  it('applies mapped suggestions without invoking train service', async () => {
    const onApply = vi.fn()
    const user = userEvent.setup()
    renderWithClient(<SuggestionView suggestion={profile.suggestion} onApply={onApply} />)
    await user.click(screen.getByRole('button', { name: /填充.*训练表单/ }))
    expect(onApply).toHaveBeenCalledWith(
      expect.objectContaining({ label_mode: 'oco', oco_take_profit_pct: 3 }),
      0,
    )
  })

  it('shows degraded alert and low-confidence mark', () => {
    renderWithClient(
      <SuggestionView
        suggestion={{
          status: 'draft',
          interval: '30m',
          vt_symbols: ['600030.SSE'],
          degraded: true,
          items: [
            { field: 'data.lookback_days', value: 'increase', reason: 'few bars', based_on_confidence: 'low' },
          ],
        }}
        onApply={vi.fn()}
      />,
    )
    expect(screen.getByText('样本或置信度不足，仅展示前置建议')).toBeInTheDocument()
    expect(screen.getByText('低置信')).toBeInTheDocument()
  })

  it('does not render group section when absent', () => {
    renderWithClient(<GroupProfileView groupProfile={null} />)
    expect(screen.queryByText('观测组关联性')).not.toBeInTheDocument()
  })
})

describe('ProfilingPanel retry state', () => {
  beforeEach(() => {
    runProfiling.mockReset()
    getProfilingArtifact.mockReset()
    listProfilingArtifacts.mockReset()
    listProfilingArtifacts.mockResolvedValue([])
  })

  it('keeps parameters and can retry after failure', async () => {
    runProfiling.mockRejectedValueOnce(new Error('boom')).mockResolvedValueOnce(profile)
    const user = userEvent.setup()
    renderWithClient(
      <ProfilingPanel
        open
        onClose={vi.fn()}
        targetSymbol="600030.SSE"
        interval="30m"
        defaultAsOf={dayjs('2024-01-01')}
        observationGroups={[]}
        onApplySuggestion={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /开始评估/ }))
    expect(await screen.findByText('画像评估失败')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /重试/ }))
    await waitFor(() => expect(runProfiling).toHaveBeenCalledTimes(2))
    expect(await screen.findByText('数据质量')).toBeInTheDocument()
  })
})

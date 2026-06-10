// CNNTrain 回归冒烟测试（task 8.2）：验证「合约可用性渲染」在改用共享
// Hook `useAvailableSymbols` 后不回归。
//
// - mock alphaService.getDataResources 返回小型 DataResourceList（含
//   raw_bars / raw_ticks / derived_bars 三类，互不相同的合约）。
// - 断言可用证券数量文案渲染（“已有 N 个证券的本地数据可用”）。
// - 打开目标证券 Select，断言来自三类来源的合约都作为可选项出现，
//   证明经由共享 Hook 的归并端到端可用。
//
// 通过 mock 网络服务保持确定性、离线（不触网）。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import { MemoryRouter } from 'react-router-dom'

import type { DataResourceList, DataResourceSummary } from '../../types/alpha'

// 构造一条 DataResourceSummary（仅填测试关心的字段，其余给占位值）。
function makeResource(
  overrides: Partial<DataResourceSummary> & Pick<DataResourceSummary, 'vt_symbol' | 'interval'>,
): DataResourceSummary {
  return {
    key: `${overrides.vt_symbol}-${overrides.interval}`,
    kind: 'raw_bar',
    row_count: 100,
    start: '2023-01-01 00:00:00',
    end: '2023-12-31 00:00:00',
    file_size_kb: 1,
    source_kind: '',
    source_interval: '',
    target_interval: '',
    ...overrides,
  } as DataResourceSummary
}

// 三类来源、互不相同的合约，验证归并覆盖 raw_bars / raw_ticks / derived_bars。
const resources: DataResourceList = {
  raw_bars: [makeResource({ vt_symbol: 'AAA.SZSE', interval: '1m' })],
  raw_ticks: [makeResource({ vt_symbol: 'BBB.SSE', interval: 'tick' })],
  derived_bars: [
    makeResource({ vt_symbol: 'CCC.SZSE', interval: '1m', target_interval: '30m' }),
  ],
  raw_bar_intervals: ['1m'],
  derived_intervals: ['30m'],
}

const getDataResources = vi.fn().mockResolvedValue(resources)
const runProfiling = vi.fn()
const listProfilingArtifacts = vi.fn().mockResolvedValue([])
vi.mock('../../api/alpha', () => ({
  alphaService: {
    getDataResources: () => getDataResources(),
    runProfiling: (req: unknown) => runProfiling(req),
    getProfilingArtifact: vi.fn(),
    listProfilingArtifacts: () => listProfilingArtifacts(),
  },
}))

// CNNTrain 拉取 CNN 模型列表；mock 为离线。
const train = vi.fn()
vi.mock('../../api/cnn', () => ({
  cnnService: {
    listModels: vi.fn().mockResolvedValue([]),
    train: (req: unknown) => train(req),
    getModel: vi.fn(),
    deleteModel: vi.fn(),
  },
}))

import CNNTrain from './index'

function renderPage(state?: Record<string, unknown>) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <MemoryRouter initialEntries={[{ pathname: '/', state }]}>
          <CNNTrain />
        </MemoryRouter>
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('CNNTrain 合约可用性渲染（共享 Hook 回归）', () => {
  beforeEach(() => {
    getDataResources.mockClear()
    runProfiling.mockReset()
    listProfilingArtifacts.mockReset()
    listProfilingArtifacts.mockResolvedValue([])
    train.mockReset()
  })

  it('渲染可用证券数量文案（三类来源归并为 3 个证券）', async () => {
    renderPage()
    // resources 异步返回后，extra 文案由 availableSymbols.length 驱动。
    expect(
      await screen.findByText('已有 3 个证券的本地数据可用'),
    ).toBeInTheDocument()
  })

  it('raw_bars/raw_ticks/derived_bars 的合约都作为目标证券可选项出现', async () => {
    const user = userEvent.setup()
    renderPage()
    // 等数据加载完成。
    await screen.findByText('已有 3 个证券的本地数据可用')

    // 打开目标证券 Select（showSearch 单选）。步骤 1 的「输入数据类型」「输入
    // 周期」两个 Select 在 DOM 中先出现，目标证券是第三个 combobox。
    const comboboxes = screen.getAllByRole('combobox')
    const target = comboboxes[2]
    await user.click(target)
    await screen.findByRole('listbox')

    // 下拉使用虚拟滚动，jsdom 下不会一次性渲染全部选项；改为对每个合约用
    // 子串搜索过滤（optionFilterProp="label"），逐一断言其作为可选项出现。
    // 三类来源分别贡献 AAA.SZSE(raw_bars)/BBB.SSE(raw_ticks)/CCC.SZSE(derived_bars)。
    for (const symbol of ['AAA.SZSE', 'BBB.SSE', 'CCC.SZSE']) {
      await user.clear(target)
      await user.type(target, symbol)
      const options = await screen.findAllByText(symbol)
      expect(options.length).toBeGreaterThanOrEqual(1)
    }
  })

  it('画像入口打开抽屉，Apply 仅回填表单且不提交训练', async () => {
    runProfiling.mockResolvedValueOnce({
      input: {
        vt_symbol: 'AAA.SZSE',
        interval: '1m',
        as_of: '2023-01-01T00:00:00',
        lookback_days: 250,
        effective_right_bound: '2023-01-01T00:00:00',
        effective_bar_count: 120,
        rules_id: 'builtin-v1',
      },
      available: true,
      blocks: [
        {
          block: 'data_quality',
          metrics: [{ key: 'count_valid_bars', value: 120, effective_sample: 120, confidence: 'high' }],
        },
        { block: 'liquidity', metrics: [] },
        { block: 'volatility', metrics: [] },
        { block: 'predictability', metrics: [] },
      ],
      suggestion: {
        status: 'draft',
        interval: '1m',
        vt_symbols: ['AAA.SZSE'],
        degraded: false,
        items: [
          { field: 'label_spec.mode', value: 'oco', reason: 'vol', based_on_confidence: 'high' },
          { field: 'label_spec.take_profit', value: 0.03, reason: 'vol', based_on_confidence: 'high' },
        ],
      },
      group_profile: null,
      overall_confidence: 'high',
      created_at: '2023-01-01T00:00:00',
      artifact_id: 'AAA.SZSE__1m__20230101T000000',
    })
    const user = userEvent.setup()
    renderPage({
      preset: {
        target_symbol: 'AAA.SZSE',
        input_data_kind: 'bar',
        input_interval: '1m',
      },
    })
    await screen.findByText('已有 3 个证券的本地数据可用')

    await user.click(screen.getByRole('button', { name: /评估该标的/ }))
    expect(await screen.findByText('标的画像评估')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /开始评估/ }))
    expect(await screen.findByText('方案建议')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /填充.*训练表单/ }))

    expect(train).not.toHaveBeenCalled()
    expect(screen.getByText('OCO 止盈止损（路径依赖）')).toBeInTheDocument()

    await user.click(screen.getByLabelText('Close'))
    expect(await screen.findByText('最近画像')).toBeInTheDocument()
    const summary = within(screen.getByLabelText('最近画像摘要'))
    expect(summary.getByText('AAA.SZSE')).toBeInTheDocument()
    expect(summary.getByText('本次评估')).toBeInTheDocument()
    expect(summary.getByText('AAA.SZSE__1m__20230101T000000')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '查看详情' }))
    expect(await screen.findByText('标的画像评估')).toBeInTheDocument()
    expect(screen.getByText('方案建议')).toBeInTheDocument()
  })
})

// 组件测试：本地聚合工作区的降级四态互斥渲染与提交行为。
//
// 测试运行器为仓库已配置的 Vitest，组件渲染使用 @testing-library/react +
// @testing-library/user-event。本文件覆盖：
//  - 降级四态（loading / error / empty / ready）的互斥渲染与优先级（Req 8.1/8.3/8.4/8.5）；
//  - 无效组合提交时不调用 alphaService.aggregateData 且保留配置（Req 7.2）；
//  - 有效组合提交时以正确的 DataAggregateRequest 调用（bar 含 source_interval、
//    tick 省略 source_interval）（Req 7.3/7.4/7.5）。
//
// 与 AggregationWorkspace.fields.test.tsx（task 9.1，字段级联）互补，互不重叠。

import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App as AntApp } from 'antd'
import dayjs from 'dayjs'

import type {
  DataAggregateRequest,
  DataResourceList,
  DataResourceSummary,
} from '../../types/alpha'

// alphaService.aggregateData 被 mock，避免真实网络请求；返回固定 task_id。
vi.mock('../../api/alpha', () => ({
  alphaService: {
    aggregateData: vi.fn(),
  },
}))

import { alphaService } from '../../api/alpha'
import AggregationWorkspace, {
  type AggregationWorkspaceProps,
} from './AggregationWorkspace'

const aggregateDataMock = vi.mocked(alphaService.aggregateData)

// ---------------------------------------------------------------------------
// 夹具构造
// ---------------------------------------------------------------------------

const JAN = '2024-01-01T00:00:00Z'
const JUN = '2024-06-01T00:00:00Z'
const FEB = '2024-02-01T00:00:00Z'
const MAY = '2024-05-01T00:00:00Z'

function makeResource(
  kind: DataResourceSummary['kind'],
  vt_symbol: string,
  interval: string,
  start: string,
  end: string,
): DataResourceSummary {
  return {
    key: `${kind}:${vt_symbol}:${interval}`,
    kind,
    vt_symbol,
    interval,
    row_count: 100,
    start,
    end,
    file_size_kb: 1,
    source_kind: '',
    source_interval: '',
    target_interval: '',
    session_profile: 'cn_equity',
  }
}

function makeResourceList(
  partial: Partial<DataResourceList>,
): DataResourceList {
  return {
    raw_bars: [],
    raw_ticks: [],
    derived_bars: [],
    raw_bar_intervals: [],
    derived_intervals: [],
    ...partial,
  }
}

const EMPTY_RESOURCES = makeResourceList({})

// 单合约 1m bar：bar 来源有效组合可达。
const BAR_RESOURCES = makeResourceList({
  raw_bars: [makeResource('raw_bar', 'AAA.SZSE', '1m', JAN, JUN)],
  raw_bar_intervals: ['1m'],
})

// 单合约仅 tick：tick 来源有效组合可达。
const TICK_RESOURCES = makeResourceList({
  raw_ticks: [makeResource('raw_tick', 'TKR.SZSE', 'tick', JAN, JUN)],
})

// 两合约 1m bar 但区间无重叠：组合可启用提交（有 target），但校验为 no-range-overlap。
const NON_OVERLAP_RESOURCES = makeResourceList({
  raw_bars: [
    makeResource('raw_bar', 'AAA.SZSE', '1m', JAN, FEB),
    makeResource('raw_bar', 'BBB.SZSE', '1m', MAY, JUN),
  ],
  raw_bar_intervals: ['1m'],
})

// ---------------------------------------------------------------------------
// 渲染辅助
// ---------------------------------------------------------------------------

function renderWorkspace(overrides: Partial<AggregationWorkspaceProps> = {}) {
  const onTaskStarted = overrides.onTaskStarted ?? vi.fn()
  const onRetry = overrides.onRetry ?? vi.fn()
  const props: AggregationWorkspaceProps = {
    resources: undefined,
    isLoading: false,
    error: undefined,
    onTaskStarted,
    onRetry,
    ...overrides,
  }
  const utils = render(
    <AntApp>
      <AggregationWorkspace {...props} />
    </AntApp>,
  )
  return { ...utils, onTaskStarted, onRetry }
}

const LOADING_TEXT = /正在读取数据资源/
const ERROR_TEXT = /数据资源加载失败/
const EMPTY_TEXT = /暂无本地数据/
const SUBMIT_NAME = /生成派生周期/

beforeEach(() => {
  aggregateDataMock.mockReset()
  aggregateDataMock.mockResolvedValue({ task_id: 'task-123' })
})

// ---------------------------------------------------------------------------
// 降级四态：互斥渲染与优先级（Requirements 8.1, 8.3, 8.4, 8.5）
// ---------------------------------------------------------------------------

describe('AggregationWorkspace 降级四态互斥渲染', () => {
  // Requirement 8.1：加载中只渲染加载态，不渲染表单/提交按钮。
  it('loading 态仅渲染加载提示，无提交表单（Req 8.1）', () => {
    renderWorkspace({ isLoading: true, resources: BAR_RESOURCES })

    expect(screen.getByText(LOADING_TEXT)).toBeInTheDocument()
    expect(screen.queryByText(ERROR_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText(EMPTY_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SUBMIT_NAME })).not.toBeInTheDocument()
  })

  // Requirement 8.3：失败态渲染失败 Alert，重试按钮触发 onRetry，且不渲染表单。
  it('error 态渲染失败提示，重试调用 onRetry（Req 8.3）', async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()
    renderWorkspace({ error: new Error('boom'), resources: undefined, onRetry })

    expect(screen.getByText(ERROR_TEXT)).toBeInTheDocument()
    expect(screen.queryByText(LOADING_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText(EMPTY_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SUBMIT_NAME })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /重试/ }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  // Requirement 8.3：未提供 onRetry 时不渲染重试按钮，但仍显示失败提示。
  it('error 态在未提供 onRetry 时不渲染重试按钮（Req 8.3）', () => {
    renderWorkspace({ error: new Error('boom'), onRetry: undefined })
    expect(screen.getByText(ERROR_TEXT)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /重试/ })).not.toBeInTheDocument()
  })

  // Requirement 8.4：空态渲染 Empty 提示，不渲染表单/提交按钮。
  it('empty 态渲染空数据提示，无提交表单（Req 8.4）', () => {
    renderWorkspace({ resources: EMPTY_RESOURCES })

    expect(screen.getByText(EMPTY_TEXT)).toBeInTheDocument()
    expect(screen.queryByText(LOADING_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText(ERROR_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: SUBMIT_NAME })).not.toBeInTheDocument()
  })

  // Requirement 8.5：就绪态渲染完整表单（含提交按钮），退出 loading/error/empty。
  it('ready 态渲染完整配置表单（Req 8.5）', () => {
    renderWorkspace({ resources: BAR_RESOURCES })

    expect(screen.getByRole('button', { name: SUBMIT_NAME })).toBeInTheDocument()
    expect(screen.getByText('合约')).toBeInTheDocument()
    expect(screen.queryByText(LOADING_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText(ERROR_TEXT)).not.toBeInTheDocument()
    expect(screen.queryByText(EMPTY_TEXT)).not.toBeInTheDocument()
  })

  // Requirement 8.1/8.3：优先级 loading > error —— 同时为真时只渲染 loading。
  it('loading 优先于 error（Req 8.1 > 8.3）', () => {
    renderWorkspace({ isLoading: true, error: new Error('boom'), resources: BAR_RESOURCES })
    expect(screen.getByText(LOADING_TEXT)).toBeInTheDocument()
    expect(screen.queryByText(ERROR_TEXT)).not.toBeInTheDocument()
  })

  // Requirement 8.3/8.4：优先级 error > empty —— 错误且无数据时只渲染 error。
  it('error 优先于 empty（Req 8.3 > 8.4）', () => {
    renderWorkspace({ error: new Error('boom'), resources: EMPTY_RESOURCES })
    expect(screen.getByText(ERROR_TEXT)).toBeInTheDocument()
    expect(screen.queryByText(EMPTY_TEXT)).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// 提交行为：无效不调用、有效以正确请求调用（Requirements 7.2, 7.3, 7.4, 7.5）
// ---------------------------------------------------------------------------

/** 当前唯一可见（未被隐藏）的 antd 下拉容器。 */
function visibleDropdown(): HTMLElement {
  const dropdowns = Array.from(
    document.querySelectorAll<HTMLElement>('.ant-select-dropdown'),
  ).filter((el) => !el.className.includes('ant-select-dropdown-hidden'))
  const last = dropdowns[dropdowns.length - 1]
  if (!last) throw new Error('no visible ant-select dropdown')
  return last
}

/** 打开合约多选下拉并点击可见下拉中的合约项（多选模式点击后下拉保持打开）。 */
async function selectSymbol(user: ReturnType<typeof userEvent.setup>, symbol: string) {
  const combo = screen.getAllByRole('combobox')[0]
  await user.click(combo)
  const option = within(visibleDropdown()).getByText(symbol, { selector: 'span' })
  await user.click(option)
  await user.keyboard('{Escape}')
}

describe('AggregationWorkspace 提交行为', () => {
  // Requirement 7.2：无效组合（区间无重叠）提交时不调用 aggregateData 且保留配置。
  it('无效组合提交不调用 aggregateData 且保留已选合约（Req 7.2）', async () => {
    const user = userEvent.setup()
    const { onTaskStarted } = renderWorkspace({ resources: NON_OVERLAP_RESOURCES })

    await selectSymbol(user, 'AAA.SZSE')
    await selectSymbol(user, 'BBB.SZSE')

    // 两合约 1m 有公共 bar 来源与目标周期，提交按钮可用，但区间无重叠 -> 无效。
    const submit = screen.getByRole('button', { name: SUBMIT_NAME })
    await waitFor(() => expect(submit).toBeEnabled())
    await user.click(submit)

    // 无效组合：不应调用聚合接口，也不应回调任务启动。
    expect(aggregateDataMock).not.toHaveBeenCalled()
    expect(onTaskStarted).not.toHaveBeenCalled()

    // 配置保留：已选合约标签仍在文档中（未被清空）。
    const symbolBox = screen.getAllByRole('combobox')[0].closest('.ant-select') as HTMLElement
    expect(within(symbolBox).getByTitle('AAA.SZSE')).toBeInTheDocument()
    expect(within(symbolBox).getByTitle('BBB.SZSE')).toBeInTheDocument()
  })

  // Requirement 7.3/7.4：有效 bar 组合提交以含 source_interval 的请求调用 aggregateData。
  it('有效 bar 组合提交以正确 DataAggregateRequest 调用（含 source_interval）（Req 7.3/7.4）', async () => {
    const user = userEvent.setup()
    const { onTaskStarted } = renderWorkspace({ resources: BAR_RESOURCES })

    await selectSymbol(user, 'AAA.SZSE')

    const submit = screen.getByRole('button', { name: SUBMIT_NAME })
    await waitFor(() => expect(submit).toBeEnabled())
    await user.click(submit)

    await waitFor(() => expect(aggregateDataMock).toHaveBeenCalledTimes(1))

    const request = aggregateDataMock.mock.calls[0][0] as DataAggregateRequest
    expect(request).toMatchObject({
      vt_symbols: ['AAA.SZSE'],
      source_kind: 'bar',
      source_interval: '1m',
      target_interval: '5m',
      start: dayjs(JAN).format('YYYY-MM-DD'),
      end: dayjs(JUN).format('YYYY-MM-DD'),
      session_profile: 'cn_equity',
    })

    await waitFor(() => expect(onTaskStarted).toHaveBeenCalledWith('task-123'))
  })

  // Requirement 7.5：有效 tick 组合提交时省略 source_interval。
  it('有效 tick 组合提交省略 source_interval（Req 7.5）', async () => {
    const user = userEvent.setup()
    const { onTaskStarted } = renderWorkspace({ resources: TICK_RESOURCES })

    await selectSymbol(user, 'TKR.SZSE')

    const submit = screen.getByRole('button', { name: SUBMIT_NAME })
    await waitFor(() => expect(submit).toBeEnabled())
    await user.click(submit)

    await waitFor(() => expect(aggregateDataMock).toHaveBeenCalledTimes(1))

    const request = aggregateDataMock.mock.calls[0][0] as DataAggregateRequest
    expect(request).toMatchObject({
      vt_symbols: ['TKR.SZSE'],
      source_kind: 'tick',
      target_interval: '5m',
      start: dayjs(JAN).format('YYYY-MM-DD'),
      end: dayjs(JUN).format('YYYY-MM-DD'),
      session_profile: 'cn_equity',
    })
    expect(request).not.toHaveProperty('source_interval')

    await waitFor(() => expect(onTaskStarted).toHaveBeenCalledWith('task-123'))
  })
})

// HistoryTable 示例测试：列表渲染、归档式删除（Popconfirm 确认 → 调用
// deleteDecision → 列表刷新），以及删除按钮不触发行点击（详情弹窗）。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'

// mock liveApi 服务，使测试离线且可断言调用参数/次数。
const listDecisions = vi.fn()
const getDecision = vi.fn()
const deleteDecision = vi.fn()
const batchDeleteDecisions = vi.fn()
const getDecisionTrace = vi.fn()
vi.mock('../../api/liveApi', () => ({
  liveService: {
    listDecisions: () => listDecisions(),
    getDecision: (signalId: string) => getDecision(signalId),
    deleteDecision: (signalId: string) => deleteDecision(signalId),
    batchDeleteDecisions: (signalIds: string[]) => batchDeleteDecisions(signalIds),
    getDecisionTrace: (signalId: string) => getDecisionTrace(signalId),
  },
}))

import HistoryTable from './HistoryTable'

const SIGNAL_ID = '2026-06-09:eod_buy_v1@v3'

function renderTable() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <HistoryTable />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('HistoryTable', () => {
  beforeEach(() => {
    listDecisions.mockReset()
    getDecision.mockReset()
    deleteDecision.mockReset()
    batchDeleteDecisions.mockReset()
    getDecisionTrace.mockReset()
  })

  it('渲染 signal_id 行与删除按钮', async () => {
    listDecisions.mockResolvedValue([SIGNAL_ID])
    renderTable()

    expect(await screen.findByText(SIGNAL_ID)).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: `删除决策 ${SIGNAL_ID}` }),
    ).toBeInTheDocument()
  })

  it('确认删除后调用 deleteDecision 并刷新列表', async () => {
    listDecisions.mockResolvedValueOnce([SIGNAL_ID]).mockResolvedValueOnce([])
    deleteDecision.mockResolvedValue({
      signal_id: SIGNAL_ID,
      deleted: true,
      trace_archived: true,
    })
    renderTable()

    await userEvent.click(
      await screen.findByRole('button', { name: `删除决策 ${SIGNAL_ID}` }),
    )
    await userEvent.click(
      await screen.findByRole('button', { name: '确认删除' }),
    )

    await waitFor(() => expect(deleteDecision).toHaveBeenCalledWith(SIGNAL_ID))
    // 删除成功 → 列表刷新（第二次返回空），行消失。
    await waitFor(() => expect(listDecisions).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(screen.queryByText(SIGNAL_ID)).not.toBeInTheDocument(),
    )
    // 点删除按钮不应触发行点击（不拉取详情、不开弹窗）。
    expect(getDecision).not.toHaveBeenCalled()
  })

  it('未勾选时批量删除按钮禁用', async () => {
    listDecisions.mockResolvedValue([SIGNAL_ID])
    renderTable()

    await screen.findByText(SIGNAL_ID)
    expect(screen.getByRole('button', { name: /批量删除/ })).toBeDisabled()
  })

  it('勾选多行后批量删除并刷新列表', async () => {
    const SECOND_ID = '2026-06-10:eod_buy_v1@v3'
    listDecisions
      .mockResolvedValueOnce([SIGNAL_ID, SECOND_ID])
      .mockResolvedValueOnce([])
    batchDeleteDecisions.mockResolvedValue({
      deleted: [SIGNAL_ID, SECOND_ID],
      missing: [],
    })
    renderTable()

    await screen.findByText(SIGNAL_ID)
    // 表头第一个 checkbox 为全选。
    await userEvent.click(screen.getAllByRole('checkbox')[0])
    expect(screen.getByRole('button', { name: /批量删除 \(2\)/ })).toBeEnabled()

    await userEvent.click(screen.getByRole('button', { name: /批量删除/ }))
    await userEvent.click(await screen.findByRole('button', { name: '确认删除' }))

    await waitFor(() =>
      expect(batchDeleteDecisions).toHaveBeenCalledWith([SIGNAL_ID, SECOND_ID]),
    )
    await waitFor(() => expect(listDecisions).toHaveBeenCalledTimes(2))
    await waitFor(() =>
      expect(screen.queryByText(SIGNAL_ID)).not.toBeInTheDocument(),
    )
    // 勾选 checkbox 不应触发行点击（不开详情弹窗）。
    expect(getDecision).not.toHaveBeenCalled()
  })

  it('取消删除不调用 deleteDecision', async () => {
    listDecisions.mockResolvedValue([SIGNAL_ID])
    renderTable()

    await userEvent.click(
      await screen.findByRole('button', { name: `删除决策 ${SIGNAL_ID}` }),
    )
    await userEvent.click(await screen.findByRole('button', { name: /取\s*消/ }))

    expect(deleteDecision).not.toHaveBeenCalled()
    expect(screen.getByText(SIGNAL_ID)).toBeInTheDocument()
  })
})

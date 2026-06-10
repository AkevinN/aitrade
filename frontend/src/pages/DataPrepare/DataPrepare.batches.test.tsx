import { describe, it, expect, beforeEach, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { App as AntApp } from 'antd'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

import type { DataResourceList, DataResourceSummary } from '../../types/alpha'
import DataPrepare from './index'

const mocks = vi.hoisted(() => ({
  getDataResourcesMock: vi.fn(),
  previewDataResourceMergeMock: vi.fn(),
  mergeDataResourceBatchesMock: vi.fn(),
}))

vi.mock('../../api/alpha', () => ({
  alphaService: {
    getDataResources: mocks.getDataResourcesMock,
    getDataResourceDetail: vi.fn(),
    downloadData: vi.fn(),
    previewCsvImport: vi.fn(),
    previewTickCsvImport: vi.fn(),
    importCsvData: vi.fn(),
    importTickCsvData: vi.fn(),
    deleteDataResource: vi.fn(),
    relocateRawBarInterval: vi.fn(),
    previewDataResourceMerge: mocks.previewDataResourceMergeMock,
    mergeDataResourceBatches: mocks.mergeDataResourceBatchesMock,
  },
}))

vi.mock('../../api/status', () => ({
  statusService: {
    getStatus: vi.fn().mockResolvedValue({ providers: [] }),
  },
}))

vi.mock('../../hooks/useTask', () => ({
  useTask: () => ({ data: null }),
}))

const batch = (
  key: string,
  fileName: string,
): DataResourceSummary => ({
  key,
  kind: 'raw_bar_batch',
  vt_symbol: '000001.SZSE',
  interval: 'd',
  row_count: 3,
  start: '2024-01-02 00:00:00',
  end: '2024-01-04 00:00:00',
  file_size_kb: 1,
  source_kind: 'bar',
  source_interval: 'd',
  target_interval: 'd',
  created_at: '2024-01-05T00:00:00',
  status: 'pending',
  batch_id: key.split('__').at(-1),
  file_name: fileName,
  batch_resource_kind: 'raw_bar',
})

const resources: DataResourceList = {
  raw_bars: [],
  raw_ticks: [],
  raw_bar_batches: [
    batch('batch__raw_bar__d__000001.SZSE__a', 'first.csv'),
    batch('batch__raw_bar__d__000001.SZSE__b', 'second.csv'),
  ],
  raw_tick_batches: [],
  derived_bars: [],
  raw_bar_intervals: [],
  derived_intervals: [],
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <AntApp>
        <MemoryRouter>
          <DataPrepare />
        </MemoryRouter>
      </AntApp>
    </QueryClientProvider>,
  )
}

beforeEach(() => {
  mocks.getDataResourcesMock.mockReset()
  mocks.previewDataResourceMergeMock.mockReset()
  mocks.mergeDataResourceBatchesMock.mockReset()
  mocks.getDataResourcesMock.mockResolvedValue(resources)
  mocks.previewDataResourceMergeMock.mockResolvedValue({
    can_merge: true,
    reason: '',
    errors: [],
    kind: 'raw_bar',
    keys: resources.raw_bar_batches?.map((item) => item.key),
    vt_symbol: '000001.SZSE',
    interval: 'd',
    intersection_start: '2024-01-03T00:00:00',
    intersection_end: '2024-01-04T00:00:00',
    conflict_count: 0,
    estimated_rows: 4,
    batch_count: 2,
  })
})

describe('DataPrepare 上传批次合并', () => {
  it('选择两个 K线批次后调用合并预检接口', async () => {
    const { container } = renderPage()

    await screen.findByText('K线批次 (2)')
    fireEvent.click(screen.getByText('K线批次 (2)'))

    expect(await screen.findByText(/first.csv/)).toBeInTheDocument()
    expect(screen.getByText(/second.csv/)).toBeInTheDocument()

    const table = Array.from(container.querySelectorAll<HTMLElement>('.ant-table-tbody'))
      .find((item) => item.textContent?.includes('first.csv')) as HTMLElement
    const checkboxes = table.querySelectorAll<HTMLElement>('.ant-table-row .ant-checkbox')
    expect(checkboxes.length).toBeGreaterThanOrEqual(2)

    fireEvent.click(checkboxes[0])
    fireEvent.click(checkboxes[1])

    const tabPanel = table.closest('[role="tabpanel"]') as HTMLElement
    fireEvent.click(within(tabPanel).getByRole('button', { name: '合并到正式K线' }))

    await waitFor(() => {
      expect(mocks.previewDataResourceMergeMock).toHaveBeenCalledWith({
        kind: 'raw_bar',
        keys: [
          'batch__raw_bar__d__000001.SZSE__a',
          'batch__raw_bar__d__000001.SZSE__b',
        ],
      })
    })
  })
})

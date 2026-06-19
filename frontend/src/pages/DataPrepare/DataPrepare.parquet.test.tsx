import { describe, it, expect, beforeEach, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { App as AntApp } from 'antd'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

import type {
  DataResourceList,
  ParquetStageResult,
  Task,
} from '../../types/alpha'
import DataPrepare from './index'

const mocks = vi.hoisted(() => ({
  getDataResourcesMock: vi.fn(),
  previewCsvImportMock: vi.fn(),
  stageParquetMock: vi.fn(),
  importParquetMock: vi.fn(),
  cancelParquetStageMock: vi.fn(),
  getTaskMock: vi.fn(),
}))

vi.mock('../../api/alpha', () => ({
  alphaService: {
    getDataResources: mocks.getDataResourcesMock,
    getDataResourceDetail: vi.fn(),
    downloadData: vi.fn(),
    previewCsvImport: mocks.previewCsvImportMock,
    previewTickCsvImport: vi.fn(),
    importCsvData: vi.fn(),
    importTickCsvData: vi.fn(),
    deleteDataResource: vi.fn(),
    relocateRawBarInterval: vi.fn(),
    previewDataResourceMerge: vi.fn(),
    mergeDataResourceBatches: vi.fn(),
    stageParquet: mocks.stageParquetMock,
    importParquet: mocks.importParquetMock,
    cancelParquetStage: mocks.cancelParquetStageMock,
    getTask: mocks.getTaskMock,
  },
}))

vi.mock('../../api/status', () => ({
  statusService: {
    getStatus: vi.fn().mockResolvedValue({ providers: [] }),
  },
}))

// 真实使用 useTask：它内部走被 mock 的 alphaService.getTask，按 task_id 轮询。

const emptyResources: DataResourceList = {
  raw_bars: [],
  raw_ticks: [],
  raw_bar_batches: [],
  raw_tick_batches: [],
  derived_bars: [],
  raw_bar_intervals: [],
  derived_intervals: [],
}

const stageResult: ParquetStageResult = {
  session_id: 'sess-abc',
  files: [
    {
      file_name: 'good.parquet',
      vt_symbol: '000001.SZSE',
      row_count: 1200,
      date_range: ['2024-01-02', '2024-06-28'],
      columns: ['datetime', 'open', 'high', 'low', 'close', 'volume'],
      missing_required: [],
      importable: true,
      reason: '',
    },
    {
      file_name: 'bad.parquet',
      vt_symbol: '',
      row_count: 0,
      date_range: ['', ''],
      columns: ['foo'],
      missing_required: ['close', 'datetime'],
      importable: false,
      reason: '缺少必填列',
    },
  ],
}

const completedTask: Task = {
  task_id: 'task-1',
  type: 'parquet_import',
  title: 'Parquet 导入',
  entity_type: 'data',
  entity_name: 'parquet',
  status: 'completed',
  progress: 100,
  message: '导入完成',
  result: { total: 2, success: 1, failed: 1, failed_files: [], batches: [], saved_as: 'raw_bar' },
  created_at: '2024-01-05T00:00:00',
  updated_at: '2024-01-05T00:01:00',
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

/** 通过 antd Upload 的隐藏 <input type=file> 注入文件，模拟用户选择。 */
function selectFiles(input: HTMLInputElement, files: File[]) {
  fireEvent.change(input, { target: { files } })
}

beforeEach(() => {
  mocks.getDataResourcesMock.mockReset()
  mocks.previewCsvImportMock.mockReset()
  mocks.stageParquetMock.mockReset()
  mocks.importParquetMock.mockReset()
  mocks.cancelParquetStageMock.mockReset()
  mocks.getTaskMock.mockReset()
  mocks.getDataResourcesMock.mockResolvedValue(emptyResources)
  mocks.stageParquetMock.mockResolvedValue(stageResult)
  mocks.importParquetMock.mockResolvedValue({ task_id: 'task-1', message: 'started' })
  mocks.cancelParquetStageMock.mockResolvedValue({ success: true })
  mocks.getTaskMock.mockResolvedValue(completedTask)
  mocks.previewCsvImportMock.mockResolvedValue({
    columns: ['datetime', 'close'],
    sample_rows: [],
    matched_fields: {},
    unmapped_columns: [],
    missing_required: [],
    total_rows: 10,
    date_range: ['2024-01-01', '2024-02-01'],
    symbols: ['000001.SZSE'],
  })
})

describe('DataPrepare Parquet 上传导入', () => {
  it('选择 .parquet 文件后暂存并渲染逐文件预览汇总，确认后调用导入', async () => {
    const { container } = renderPage()

    // 切到「K线CSV」标签，里面同时承载 parquet 上传入口。
    fireEvent.click(await screen.findByText('K线CSV'))

    const barUploadInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    expect(barUploadInput).toBeTruthy()

    const parquetFile = new File([new Uint8Array([1, 2, 3])], 'good.parquet', {
      type: 'application/octet-stream',
    })
    const badFile = new File([new Uint8Array([4])], 'bad.parquet', {
      type: 'application/octet-stream',
    })
    selectFiles(barUploadInput!, [parquetFile, badFile])

    // 暂存接口应被调用，且 data_kind=bar。
    await waitFor(() => {
      expect(mocks.stageParquetMock).toHaveBeenCalledTimes(1)
    })
    const [filesArg, kindArg] = mocks.stageParquetMock.mock.calls[0]
    expect(kindArg).toBe('bar')
    expect(filesArg).toHaveLength(2)

    // 预览汇总表渲染两行：识别代码与行数。
    expect(await screen.findByText('000001.SZSE')).toBeInTheDocument()
    expect(screen.getByText('1200')).toBeInTheDocument()
    // 不可导入行展示原因。
    expect(screen.getByText('缺少必填列')).toBeInTheDocument()

    // 点击「确认导入」，应带 session_id 调用 importParquet。
    fireEvent.click(screen.getByRole('button', { name: '确认导入' }))

    await waitFor(() => {
      expect(mocks.importParquetMock).toHaveBeenCalledTimes(1)
    })
    expect(mocks.importParquetMock.mock.calls[0][0]).toMatchObject({
      session_id: 'sess-abc',
      data_kind: 'bar',
      import_mode: 'merge',
    })

    // useTask 被驱动到 completed，导入摘要可见。
    await waitFor(() => {
      expect(mocks.getTaskMock).toHaveBeenCalledWith('task-1')
    })
  })

  it('点击取消会调用 cancelParquetStage 清理服务端暂存会话', async () => {
    const { container } = renderPage()

    fireEvent.click(await screen.findByText('K线CSV'))
    const barUploadInput = container.querySelector<HTMLInputElement>('input[type="file"]')
    const parquetFile = new File([new Uint8Array([1, 2, 3])], 'good.parquet', {
      type: 'application/octet-stream',
    })
    selectFiles(barUploadInput!, [parquetFile])

    await waitFor(() => {
      expect(mocks.stageParquetMock).toHaveBeenCalledTimes(1)
    })
    await screen.findByText('000001.SZSE')

    // antd 会在两个 CJK 字符间插入空格，按钮可访问名为「取 消」。
    fireEvent.click(screen.getByRole('button', { name: /取\s*消/ }))

    await waitFor(() => {
      expect(mocks.cancelParquetStageMock).toHaveBeenCalledWith('sess-abc')
    })
    expect(mocks.importParquetMock).not.toHaveBeenCalled()
  })

  it('单个 .csv 文件仍走原有 previewCsv 流程而非 parquet 暂存', async () => {
    const { container } = renderPage()

    fireEvent.click(await screen.findByText('K线CSV'))
    const barUploadInput = container.querySelector<HTMLInputElement>(
      'input[type="file"]',
    )
    const csvFile = new File(['datetime,close\n2024-01-01,1'], 'data.csv', {
      type: 'text/csv',
    })
    selectFiles(barUploadInput!, [csvFile])

    await waitFor(() => {
      expect(mocks.previewCsvImportMock).toHaveBeenCalledTimes(1)
    })
    expect(mocks.stageParquetMock).not.toHaveBeenCalled()
  })
})

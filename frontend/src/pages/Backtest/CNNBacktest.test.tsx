// CNNBacktest 页面测试（Task 7.4）：验证 path_class 模型的 veto 控件显隐逻辑。
//
// - mock cnnService.listModels 返回两个模型：
//   一个 objective='path_class'，一个 objective='classification'。
// - 选中 path_class 模型时，veto_threshold 控件出现；
// - 选中 classification 模型时，veto_threshold 控件不出现。
//
// 通过 mock 网络服务保持确定性、离线（不触网）。

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { App as AntApp } from 'antd'
import type { UseQueryResult } from '@tanstack/react-query'

import type { CNNModelInfo } from '../../types/cnn'
import type { Task } from '../../types/alpha'

// ── mock cnnService（必须在 import 之前声明，vitest hoist） ─────────────────
const mockListModels = vi.fn<() => Promise<CNNModelInfo[]>>()
const mockRunBacktest = vi.fn()

vi.mock('../../api/cnn', () => ({
  cnnService: {
    listModels: () => mockListModels(),
    runBacktest: (...args: unknown[]) => mockRunBacktest(...args),
  },
}))

// ── mock useTask（始终返回 null，只测 UI 显隐不测回测执行） ──────────────────
vi.mock('../../hooks/useTask', () => ({
  useTask: () => ({ data: null } as unknown as UseQueryResult<Task>),
}))

import CNNBacktest from './CNNBacktest'

// ── 两个测试用模型 fixture ──────────────────────────────────────────────────
const PATH_CLASS_MODEL: CNNModelInfo = {
  name: 'cnn_path_v1',
  created_at: '2025-01-01T00:00:00',
  target_symbol: '000001.SZSE',
  input_interval: 'd',
  objective: 'path_class',
  best_epoch: 30,
  best_val_loss: 0.8,
}

const CLASSIFICATION_MODEL: CNNModelInfo = {
  name: 'cnn_cls_v1',
  created_at: '2025-01-01T00:00:00',
  target_symbol: '000001.SZSE',
  input_interval: 'd',
  objective: 'classification',
  best_epoch: 25,
  best_val_loss: 0.55,
}

// ── 渲染辅助 ────────────────────────────────────────────────────────────────
function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  // CNNBacktest 不依赖 Router（无 useLocation），直接包 QueryClient + AntApp 即可。
  return render(
    <QueryClientProvider client={qc}>
      <AntApp>
        <CNNBacktest />
      </AntApp>
    </QueryClientProvider>,
  )
}

describe('CNNBacktest veto_threshold 控件显隐', () => {
  beforeEach(() => {
    mockListModels.mockReset()
    mockRunBacktest.mockReset()
    mockListModels.mockResolvedValue([PATH_CLASS_MODEL, CLASSIFICATION_MODEL])
  })

  it('选中 path_class 模型时 veto_threshold 控件出现', async () => {
    const user = userEvent.setup()
    renderPage()

    // 等待模型列表加载完成（Select 可交互）
    const comboboxes = await screen.findAllByRole('combobox')
    // 第一个 combobox 是「CNN 模型」选择器
    const modelSelect = comboboxes[0]
    await user.click(modelSelect)

    // 选 path_class 模型
    const listbox = await screen.findByRole('listbox')
    await user.click(listbox.querySelector(`[title="${PATH_CLASS_MODEL.name}"]`) ??
      screen.getByText(`${PATH_CLASS_MODEL.name} (${PATH_CLASS_MODEL.target_symbol})`))

    // veto 控件说明文案应出现
    await waitFor(() => {
      expect(screen.getByText(/否决阈值 veto_threshold/)).toBeInTheDocument()
    })
  })

  it('选中 classification 模型时 veto_threshold 控件不出现', async () => {
    const user = userEvent.setup()
    renderPage()

    const comboboxes = await screen.findAllByRole('combobox')
    const modelSelect = comboboxes[0]
    await user.click(modelSelect)

    const listbox = await screen.findByRole('listbox')
    await user.click(listbox.querySelector(`[title="${CLASSIFICATION_MODEL.name}"]`) ??
      screen.getByText(`${CLASSIFICATION_MODEL.name} (${CLASSIFICATION_MODEL.target_symbol})`))

    // 等待模型信息显示
    await waitFor(() => {
      expect(screen.getByText(/方向分类/)).toBeInTheDocument()
    })

    // veto 控件不应出现
    expect(screen.queryByText(/否决阈值 veto_threshold/)).not.toBeInTheDocument()
  })
})

// 组件测试（task 9.1）：AggregationWorkspace 的 Symbol_Selector 与字段联动。
//
// 覆盖数据驱动配置的交互行为（@testing-library/react + user-event）：
// - 合约选择器：多选、大小写不敏感子串搜索、空查询展示全部、无匹配时
//   notFoundContent="暂无本地数据"，且不存在手填 TextArea。
// - 来源类型：bar/tick 两项始终展示，不可用项 disabled。
// - 来源周期：bar 时显示、tick 时隐藏。
// - 目标周期：随来源周期变化刷新。
// - 时间范围：公共可用区间文本展示 + 默认区间填充。
// - 时段规则：仅 cn_equity 且默认值为 cn_equity。
//
// 组件经由 props 接收 resources（不触网）；message 上下文由 antd <App> 提供。
// _Requirements: 1.2, 1.3, 1.4, 1.5, 2.4, 3.3, 4.4, 5.3, 5.4, 6.1, 6.2_

import { describe, it, expect, vi } from 'vitest'
import { render, screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App as AntApp } from 'antd'

import type { DataResourceList, DataResourceSummary } from '../../types/alpha'
import AggregationWorkspace from './AggregationWorkspace'

// ---------------------------------------------------------------------------
// Fixtures：构造 useAvailableSymbols 能归并出所需可用性的 DataResourceList。
// ---------------------------------------------------------------------------

const RANGE_START = '2024-01-01T00:00:00Z'
const RANGE_END = '2024-06-01T00:00:00Z'

// 构造一条 DataResourceSummary（仅填测试关心字段，其余占位）。
function makeResource(
  overrides: Partial<DataResourceSummary> &
    Pick<DataResourceSummary, 'vt_symbol' | 'interval'>,
): DataResourceSummary {
  return {
    key: `${overrides.vt_symbol}-${overrides.interval}`,
    kind: 'raw_bar',
    row_count: 100,
    start: RANGE_START,
    end: RANGE_END,
    file_size_kb: 1,
    source_kind: '',
    source_interval: '',
    target_interval: '',
    ...overrides,
  } as DataResourceSummary
}

// AAA.SZSE / AAB.SZSE：拥有 1m、30m K线 + tick（bar 与 tick 均可用）。
// BBB.SSE：仅有 1m K线（bar 可用、tick 不可用）。
const resources: DataResourceList = {
  raw_bars: [
    makeResource({ vt_symbol: 'AAA.SZSE', interval: '1m' }),
    makeResource({ vt_symbol: 'AAA.SZSE', interval: '30m' }),
    makeResource({ vt_symbol: 'AAB.SZSE', interval: '1m' }),
    makeResource({ vt_symbol: 'AAB.SZSE', interval: '30m' }),
    makeResource({ vt_symbol: 'BBB.SSE', interval: '1m' }),
  ],
  raw_ticks: [
    makeResource({ vt_symbol: 'AAA.SZSE', interval: 'tick', kind: 'raw_tick' }),
    makeResource({ vt_symbol: 'AAB.SZSE', interval: 'tick', kind: 'raw_tick' }),
  ],
  derived_bars: [],
  raw_bar_intervals: ['1m', '30m'],
  derived_intervals: [],
}

function renderWorkspace(overrides: Partial<Parameters<typeof AggregationWorkspace>[0]> = {}) {
  return render(
    <AntApp>
      <AggregationWorkspace
        resources={resources}
        isLoading={false}
        error={undefined}
        onTaskStarted={vi.fn()}
        onRetry={vi.fn()}
        {...overrides}
      />
    </AntApp>,
  )
}

// 当前唯一可见（未被隐藏）的 antd 下拉容器；切换 Select 时旧下拉被标记隐藏。
function visibleDropdown(): HTMLElement {
  const dropdowns = Array.from(
    document.querySelectorAll<HTMLElement>('.ant-select-dropdown'),
  ).filter((el) => !el.className.includes('ant-select-dropdown-hidden'))
  const last = dropdowns[dropdowns.length - 1]
  if (!last) throw new Error('no visible ant-select dropdown')
  return last
}

// 可见下拉中真正可交互的选项项（.ant-select-item-option），而非 a11y 镜像。
function optionItems(dd: HTMLElement): HTMLElement[] {
  return Array.from(dd.querySelectorAll<HTMLElement>('.ant-select-item-option'))
}

// 取某个选项项渲染出的标签文本（optionRender 内容）。
function optionLabel(el: HTMLElement): string {
  return el.querySelector('.ant-select-item-option-content')?.textContent ?? el.textContent ?? ''
}

// 点击可见下拉中标签包含指定文本的选项项。
async function clickOption(
  user: ReturnType<typeof userEvent.setup>,
  dd: HTMLElement,
  text: string,
): Promise<void> {
  const item = optionItems(dd).find((el) => optionLabel(el).includes(text))
  if (!item) throw new Error(`option not found: ${text}`)
  await user.click(item)
}

// 选中一个合约（合约选择器为第一个 combobox）。
async function selectSymbol(
  user: ReturnType<typeof userEvent.setup>,
  label: string,
): Promise<void> {
  const combo = screen.getAllByRole('combobox')[0]
  await user.click(combo)
  await clickOption(user, visibleDropdown(), label)
  await user.keyboard('{Escape}')
}

// ---------------------------------------------------------------------------
// Requirement 1.2 / 1.3：多选 + 大小写不敏感子串搜索 + 空查询展示全部。
// ---------------------------------------------------------------------------

describe('Symbol_Selector 多选与搜索过滤（Req 1.2, 1.3）', () => {
  it('空查询时展示全部 Available_Symbol（Req 1.3）', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    const combo = screen.getAllByRole('combobox')[0]
    await user.click(combo)

    const labels = optionItems(visibleDropdown()).map(optionLabel)
    expect(labels.some((t) => t.includes('AAA.SZSE'))).toBe(true)
    expect(labels.some((t) => t.includes('AAB.SZSE'))).toBe(true)
    expect(labels.some((t) => t.includes('BBB.SSE'))).toBe(true)
  })

  it('按合约代码大小写不敏感子串过滤（Req 1.2, 1.3）', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    const combo = screen.getAllByRole('combobox')[0]
    await user.click(combo)
    // 小写 "aa" 应命中 AAA.SZSE 与 AAB.SZSE，不命中 BBB.SSE。
    await user.type(combo, 'aa')

    await waitFor(() => {
      const labels = optionItems(visibleDropdown()).map(optionLabel)
      expect(labels.some((t) => t.includes('AAA.SZSE'))).toBe(true)
      expect(labels.some((t) => t.includes('AAB.SZSE'))).toBe(true)
      expect(labels.some((t) => t.includes('BBB.SSE'))).toBe(false)
    })
  })

  it('支持多选：选中两个合约后两者均保留为已选项（Req 1.2）', async () => {
    const user = userEvent.setup()
    const { container } = renderWorkspace()

    await selectSymbol(user, 'AAA.SZSE')
    await selectSymbol(user, 'AAB.SZSE')

    const multiple = container.querySelector('.ant-select-multiple') as HTMLElement
    const items = multiple.querySelectorAll('.ant-select-selection-item')
    expect(items.length).toBe(2)
    const itemTexts = Array.from(items).map((i) => i.textContent ?? '')
    expect(itemTexts.some((t) => t.includes('AAA.SZSE'))).toBe(true)
    expect(itemTexts.some((t) => t.includes('AAB.SZSE'))).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// Requirement 1.4：移除手填 TextArea。
// ---------------------------------------------------------------------------

describe('合约输入方式（Req 1.4）', () => {
  it('工作区不存在 TextArea 手填输入', () => {
    const { container } = renderWorkspace()
    expect(container.querySelector('textarea')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Requirement 1.5：无匹配时 notFoundContent="暂无本地数据"。
// ---------------------------------------------------------------------------

describe('Symbol_Selector 无匹配提示（Req 1.5）', () => {
  it('搜索无匹配合约时下拉展示「暂无本地数据」', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    const combo = screen.getAllByRole('combobox')[0]
    await user.click(combo)
    await user.type(combo, 'ZZZ-NOPE')

    await waitFor(() => {
      expect(within(visibleDropdown()).getByText('暂无本地数据')).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// Requirement 2.4：bar/tick 两项始终展示，不可用项 disabled。
// ---------------------------------------------------------------------------

describe('来源类型选项（Req 2.4）', () => {
  it('两个来源类型始终展示，不可用项置为 disabled', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    // BBB.SSE 仅有 bar 数据 -> tick 应不可选。
    await selectSymbol(user, 'BBB.SSE')
    await screen.findByText('原始K线') // 等默认来源类型 = bar 应用完成

    // 打开来源类型 Select（symbol=0, sourceKind=1）。
    const sourceKind = screen.getAllByRole('combobox')[1]
    await user.click(sourceKind)

    const dd = visibleDropdown()
    const items = optionItems(dd)
    // bar 与 tick 两个选项均展示。
    expect(items.length).toBe(2)
    const labels = items.map(optionLabel)
    expect(labels.some((t) => t.includes('原始K线'))).toBe(true)
    expect(labels.some((t) => t.includes('历史Tick'))).toBe(true)

    // tick 不可用 -> disabled。
    const tickItem = items.find((el) => optionLabel(el).includes('历史Tick'))
    expect(tickItem?.className).toContain('ant-select-item-option-disabled')
    // bar 可用 -> 非 disabled。
    const barItem = items.find((el) => optionLabel(el).includes('原始K线'))
    expect(barItem?.className).not.toContain('ant-select-item-option-disabled')
  })
})

// ---------------------------------------------------------------------------
// Requirement 3.3：bar 时显示来源周期、tick 时隐藏。
// ---------------------------------------------------------------------------

describe('来源周期显隐（Req 3.3）', () => {
  it('bar 时展示来源周期，切换到 tick 后隐藏', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    // AAA.SZSE 同时拥有 bar 与 tick；默认来源类型 = bar。
    await selectSymbol(user, 'AAA.SZSE')
    await screen.findByText('原始K线')

    // bar 来源下「来源周期」字段（标签）可见。
    expect(screen.getByText('来源周期')).toBeInTheDocument()

    // 切换来源类型为 tick。
    const sourceKind = screen.getAllByRole('combobox')[1]
    await user.click(sourceKind)
    await clickOption(user, visibleDropdown(), '历史Tick')

    // tick 来源下「来源周期」字段被隐藏。
    await waitFor(() => {
      expect(screen.queryByText('来源周期')).toBeNull()
    })
  })
})

// ---------------------------------------------------------------------------
// Requirement 4.4：目标周期随来源刷新。
// ---------------------------------------------------------------------------

describe('目标周期随来源刷新（Req 4.4）', () => {
  it('来源周期由 1m 改为 30m 后目标周期可选项刷新', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    // AAA.SZSE：来源周期可选 {1m, 30m}，默认 1m -> 目标含多档。
    await selectSymbol(user, 'AAA.SZSE')
    await screen.findByText('原始K线')

    const targetSelectionText = () =>
      screen
        .getAllByRole('combobox')[3]
        .closest('.ant-select')!
        .querySelector('.ant-select-selection-item')?.textContent ?? ''

    // 1m 来源下目标默认归一为最细 5m。
    await waitFor(() => {
      expect(targetSelectionText()).toContain('5分钟')
    })

    // 打开目标周期 Select（symbol=0, sourceKind=1, sourceInterval=2, target=3）。
    const targetBefore = screen.getAllByRole('combobox')[3]
    await user.click(targetBefore)
    const beforeLabels = optionItems(visibleDropdown()).map(optionLabel)
    // 1m 来源 -> {5m,10m,15m,30m,60m}。
    expect(beforeLabels).toEqual(['5分钟', '10分钟', '15分钟', '30分钟', '60分钟'])
    await user.keyboard('{Escape}')

    // 将来源周期改为 30m。
    const sourceInterval = screen.getAllByRole('combobox')[2]
    await user.click(sourceInterval)
    await clickOption(user, visibleDropdown(), '30分钟')

    // 目标周期可选项随来源刷新：30m 来源下唯一目标为 60m，旧选中 5m 被归一为 60m。
    await waitFor(() => {
      expect(targetSelectionText()).toContain('60分钟')
    })
  })
})

// ---------------------------------------------------------------------------
// Requirement 5.3 / 5.4：公共可用区间文本 + 默认区间填充。
// ---------------------------------------------------------------------------

describe('时间范围联动（Req 5.3, 5.4）', () => {
  it('展示公共可用区间文本并默认填充时间范围', async () => {
    const user = userEvent.setup()
    const { container } = renderWorkspace()

    await selectSymbol(user, 'AAA.SZSE')
    await screen.findByText('原始K线')

    // Req 5.4：附近展示公共可用区间文本。
    await waitFor(() => {
      expect(
        screen.getByText((content) =>
          content.includes('本地数据可用：2024-01-01 ~ 2024-06-01'),
        ),
      ).toBeInTheDocument()
    })

    // Req 5.3：时间范围默认被填充（RangePicker 两个输入均有值）。
    const pickerInputs = container.querySelectorAll<HTMLInputElement>(
      '.ant-picker-input input',
    )
    expect(pickerInputs.length).toBe(2)
    expect(pickerInputs[0].value).toMatch(/\d{4}-\d{2}-\d{2}/)
    expect(pickerInputs[1].value).toMatch(/\d{4}-\d{2}-\d{2}/)
  })
})

// ---------------------------------------------------------------------------
// Requirement 6.1 / 6.2：时段规则仅 cn_equity 且默认 cn_equity。
// ---------------------------------------------------------------------------

describe('时段规则（Req 6.1, 6.2）', () => {
  it('默认值为 cn_equity 且可选项仅 cn_equity', async () => {
    const user = userEvent.setup()
    renderWorkspace()

    // Req 6.1：默认展示 cn_equity（标签「A股日内时段」）。
    expect(screen.getByText('A股日内时段')).toBeInTheDocument()

    // Req 6.2：打开时段规则 Select，仅有唯一可选项。
    const combos = screen.getAllByRole('combobox')
    const sessionCombo = combos[combos.length - 1]
    await user.click(sessionCombo)

    const items = optionItems(visibleDropdown())
    expect(items.length).toBe(1)
    expect(optionLabel(items[0])).toContain('A股日内时段')
  })
})

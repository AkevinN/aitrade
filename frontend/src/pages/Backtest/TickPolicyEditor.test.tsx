// TickPolicyEditor RTL：多策略增删、kind 切换、label 唯一标红、条件策略内嵌与回传。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { App as AntApp } from 'antd'

import TickPolicyEditor from './TickPolicyEditor'
import type { TickPolicyCfg } from '../../types/t0'

const FIXED = (label: string): TickPolicyCfg => ({ kind: 'fixed', label, sell_tick: 0.02, buy_tick: 0.02 })
const COND: TickPolicyCfg = {
  kind: 'conditional', label: '条件', rules: [
    { name: '高开', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.07, buy_tick: 0.01 },
  ], default_sell_tick: 0.03, default_buy_tick: 0.03, pricetick: 0.01,
}

function renderEditor(value: TickPolicyCfg[]) {
  const onChange = vi.fn()
  render(
    <AntApp>
      <TickPolicyEditor value={value} onChange={onChange} signalNames={['mdl_prob']} />
    </AntApp>,
  )
  return { onChange }
}

// 驱动 antd Select（选项渲染到 body portal）
async function selectOption(comboboxIndex: number, optionText: string) {
  const combo = screen.getAllByRole('combobox')[comboboxIndex]
  fireEvent.mouseDown(combo)
  await waitFor(() => {
    const opts = document.querySelectorAll('.ant-select-item-option')
    if (!Array.from(opts).some((el) => el.textContent?.includes(optionText)))
      throw new Error(`option ${optionText} not found`)
  })
  const target = Array.from(document.querySelectorAll('.ant-select-item-option'))
    .find((el) => el.textContent?.includes(optionText))!
  fireEvent.click(target)
}

describe('TickPolicyEditor', () => {
  beforeEach(() => vi.clearAllMocks())

  it('固定策略以分展示', () => {
    renderEditor([FIXED('固定2分')])
    expect((screen.getByLabelText('策略0卖档') as HTMLInputElement).value).toBe('2')
    expect((screen.getByLabelText('策略0买档') as HTMLInputElement).value).toBe('2')
  })

  it('添加策略：追加一个固定档', () => {
    const { onChange } = renderEditor([FIXED('a')])
    fireEvent.click(screen.getByText('添加档位策略'))
    const next = onChange.mock.calls[0][0]
    expect(next).toHaveLength(2)
    expect(next[1].kind).toBe('fixed')
  })

  it('删除策略：移除该条', () => {
    const { onChange } = renderEditor([FIXED('a'), FIXED('b')])
    fireEvent.click(screen.getByLabelText('策略0删除'))
    expect(onChange.mock.calls[0][0].map((p: TickPolicyCfg) => p.label)).toEqual(['b'])
  })

  it('单个策略时删除按钮禁用', () => {
    renderEditor([FIXED('only')])
    expect(screen.getByLabelText('策略0删除')).toBeDisabled()
  })

  it('改名：回传新 label', () => {
    const { onChange } = renderEditor([FIXED('a')])
    fireEvent.change(screen.getByLabelText('策略0名称'), { target: { value: '我的策略' } })
    expect(onChange.mock.calls.at(-1)![0][0].label).toBe('我的策略')
  })

  it('label 重复时标红提示', () => {
    renderEditor([FIXED('dup'), FIXED('dup')])
    expect(screen.getAllByText('名称重复').length).toBe(2)
  })

  it('kind 切到条件规则：替换为带默认规则的 conditional', async () => {
    const { onChange } = renderEditor([FIXED('a')])
    await selectOption(0, '条件规则')
    const next = onChange.mock.calls.at(-1)![0][0]
    expect(next.kind).toBe('conditional')
    expect(next.rules.length).toBeGreaterThanOrEqual(1)
    expect(next.label).toBe('a')   // 保留名称
  })

  it('波动缩放策略显示 k/窗口/回退档', () => {
    renderEditor([{ kind: 'vol_scaled', label: 'v', k: 0.4, n: 20, fallback: 0.02 }])
    expect(screen.getByLabelText('策略0系数k')).toBeInTheDocument()
    expect(screen.getByLabelText('策略0回退档')).toBeInTheDocument()
  })

  it('条件策略内嵌规则编辑器并回传默认档变更', () => {
    const { onChange } = renderEditor([COND])
    expect(screen.getByText('添加规则')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('默认卖档'), { target: { value: '5' } })
    expect(onChange.mock.calls.at(-1)![0][0].default_sell_tick).toBeCloseTo(0.05, 9)
  })
})

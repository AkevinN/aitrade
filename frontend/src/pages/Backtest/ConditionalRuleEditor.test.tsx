// ConditionalRuleEditor RTL：增删/上下移规则、LHS 联动 window/signal、分↔元 / %↔小数 换算。
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { App as AntApp } from 'antd'

import ConditionalRuleEditor from './ConditionalRuleEditor'
import type { RuleCfg } from '../../types/t0'

const GAP_RULE: RuleCfg = { name: '', lhs: 'gap', op: 'gt', threshold: 0.003, sell_tick: 0.07, buy_tick: 0.01 }

function renderEditor(props: Partial<React.ComponentProps<typeof ConditionalRuleEditor>> = {}) {
  const onChange = vi.fn()
  render(
    <AntApp>
      <ConditionalRuleEditor
        rules={props.rules ?? [GAP_RULE]}
        defaultSellTick={props.defaultSellTick ?? 0.03}
        defaultBuyTick={props.defaultBuyTick ?? 0.03}
        signalNames={props.signalNames ?? ['mdl_prob']}
        onChange={props.onChange ?? onChange}
      />
    </AntApp>,
  )
  return { onChange: props.onChange ?? onChange }
}

describe('ConditionalRuleEditor', () => {
  beforeEach(() => vi.clearAllMocks())

  it('以友好单位渲染：gap 阈值显示为 %、档位显示为 分', () => {
    renderEditor()
    expect((screen.getByLabelText('规则0阈值') as HTMLInputElement).value).toBe('0.3')   // 0.003→0.3%
    expect((screen.getByLabelText('规则0卖档') as HTMLInputElement).value).toBe('7')      // 0.07→7分
    expect((screen.getByLabelText('规则0买档') as HTMLInputElement).value).toBe('1')      // 0.01→1分
  })

  it('添加规则：追加一条默认 gap 规则', () => {
    const { onChange } = renderEditor()
    fireEvent.click(screen.getByText('添加规则'))
    expect(onChange).toHaveBeenCalledTimes(1)
    const next = onChange.mock.calls[0][0]
    expect(next.rules).toHaveLength(2)
    expect(next.rules[1].lhs).toBe('gap')
  })

  it('删除规则：移除该条', () => {
    const { onChange } = renderEditor({ rules: [GAP_RULE, { ...GAP_RULE, op: 'lt' }] })
    fireEvent.click(screen.getByLabelText('规则0删除'))
    expect(onChange.mock.calls[0][0].rules).toHaveLength(1)
    expect(onChange.mock.calls[0][0].rules[0].op).toBe('lt')   // 删的是第0条
  })

  it('仅剩一条规则时删除按钮禁用（不能清空规则）', () => {
    renderEditor()   // 默认 1 条规则
    expect(screen.getByLabelText('规则0删除')).toBeDisabled()
  })

  it('下移规则：交换顺序', () => {
    const { onChange } = renderEditor({ rules: [{ ...GAP_RULE, name: 'A' }, { ...GAP_RULE, name: 'B' }] })
    fireEvent.click(screen.getByLabelText('规则0下移'))
    expect(onChange.mock.calls[0][0].rules.map((r: RuleCfg) => r.name)).toEqual(['B', 'A'])
  })

  it('gap 阈值按 % 输入、存为小数', () => {
    const { onChange } = renderEditor()
    fireEvent.change(screen.getByLabelText('规则0阈值'), { target: { value: '0.5' } })
    expect(onChange.mock.calls.at(-1)![0].rules[0].threshold).toBeCloseTo(0.005, 9)
  })

  it('卖档按分输入、存为元', () => {
    const { onChange } = renderEditor()
    fireEvent.change(screen.getByLabelText('规则0卖档'), { target: { value: '8' } })
    expect(onChange.mock.calls.at(-1)![0].rules[0].sell_tick).toBeCloseTo(0.08, 9)
  })

  it('默认档位可编辑、存为元', () => {
    const { onChange } = renderEditor()
    fireEvent.change(screen.getByLabelText('默认卖档'), { target: { value: '5' } })
    expect(onChange.mock.calls.at(-1)![0].defaultSellTick).toBeCloseTo(0.05, 9)
  })

  it('lhs=mean_range 显示窗口输入；lhs=gap 不显示', () => {
    renderEditor({ rules: [{ ...GAP_RULE, lhs: 'mean_range', window: 20 }] })
    expect(screen.getByLabelText('规则0窗口')).toBeInTheDocument()
  })

  it('lhs=gap 不显示窗口/信号字段', () => {
    renderEditor()
    expect(screen.queryByLabelText('规则0窗口')).toBeNull()
    expect(screen.queryByText('mdl_prob')).toBeNull()
  })

  it('lhs=signal 显示已选信号名', () => {
    renderEditor({ rules: [{ ...GAP_RULE, lhs: 'signal', signal_name: 'mdl_prob' }] })
    expect(screen.getByText('mdl_prob')).toBeInTheDocument()
  })
})

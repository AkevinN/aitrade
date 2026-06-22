// DayBarField RTL 测试：按天模式自动换算 bar 并 emit；按 bar 模式直填；自定义周期强制手填。

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App as AntApp } from 'antd'

import DayBarField from './DayBarField'

function renderField(props: Partial<React.ComponentProps<typeof DayBarField>> = {}) {
  const onChange = vi.fn()
  render(
    <AntApp>
      <DayBarField interval="30m" maxBars={120} minBars={1} defaultDays={5} onChange={onChange} {...props} />
    </AntApp>,
  )
  return { onChange }
}

describe('DayBarField', () => {
  beforeEach(() => vi.clearAllMocks())

  it('按天模式：挂载即按 days×每日bar数 换算并 emit bar 数（30m: 5×8=40）', async () => {
    const { onChange } = renderField()
    // 实时换算文案
    expect(await screen.findByText(/每交易日 8 根 × 5 日 → 40 根 bar/)).toBeInTheDocument()
    // emit 换算后的 bar 数
    await waitFor(() => expect(onChange).toHaveBeenCalledWith(40))
  })

  it('改观测交易日数 → 重新换算 emit（10×8=80）', async () => {
    const { onChange } = renderField()
    await waitFor(() => expect(onChange).toHaveBeenCalled())
    const dayInput = screen.getByRole('spinbutton')
    // antd InputNumber 的多字符 type 不稳定，用 fireEvent.change 原子设值。
    fireEvent.change(dayInput, { target: { value: '10' } })
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith(80))
    expect(await screen.findByText(/× 10 日 → 80 根 bar/)).toBeInTheDocument()
  })

  it('切到「按 bar」模式 → 直填 bar 根数', async () => {
    const { onChange } = renderField({ value: 40 })
    await userEvent.click(screen.getByText('按 bar'))
    // 出现 bar 输入框（addonAfter=bar），手填 12 → emit 12
    const barInput = screen.getByRole('spinbutton')
    await userEvent.clear(barInput)
    await userEvent.type(barInput, '12')
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith(12))
  })

  it('换算表外的周期：强制手填 bar，提示不可换算', async () => {
    renderField({ interval: '3m' })
    expect(screen.getByText(/周期「3m」不在换算表内/)).toBeInTheDocument()
    // 「按天」选项被禁用（Segmented disabled）
    const segItems = document.querySelectorAll('.ant-segmented-item-disabled')
    expect(segItems.length).toBeGreaterThan(0)
  })
})

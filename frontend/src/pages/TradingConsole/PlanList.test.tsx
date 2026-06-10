// PlanList 示例测试：列表渲染、空态、启停/触发/编辑/删除回调（Req 8.2）。
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

import PlanList from './PlanList'
import { makePlanSummary } from './testFixtures'

const noop = () => {}

function baseProps(over: Record<string, unknown> = {}) {
  return {
    plans: [makePlanSummary()],
    onToggle: vi.fn(),
    onRun: vi.fn(),
    onEdit: vi.fn(),
    onDelete: vi.fn(),
    ...over,
  }
}

describe('PlanList', () => {
  it('空列表展示占位提示', () => {
    render(<PlanList plans={[]} onToggle={noop} onRun={noop} onEdit={noop} onDelete={noop} />)
    expect(screen.getByText(/暂无交易计划/)).toBeInTheDocument()
  })

  it('展示计划名称、标的与决策时点', () => {
    render(<PlanList {...baseProps()} />)
    expect(screen.getByText('尾盘买入计划')).toBeInTheDocument()
    expect(screen.getByText(/000001\.SZSE/)).toBeInTheDocument()
    expect(screen.getByText('15:05')).toBeInTheDocument()
  })

  it('展示多个唤醒时刻标签', () => {
    render(
      <PlanList
        {...baseProps({ plans: [makePlanSummary({ trigger_times: ['09:35', '15:05'] })] })}
      />,
    )
    expect(screen.getByText('09:35')).toBeInTheDocument()
    expect(screen.getByText('15:05')).toBeInTheDocument()
  })

  it('日内计划展示「盘中监控 · 30m」且不显示唤醒时刻', () => {
    render(
      <PlanList
        {...baseProps({
          plans: [makePlanSummary({ bar_freq: '30m', trigger_times: [] })],
        })}
      />,
    )
    expect(screen.getByText('盘中监控 · 30m')).toBeInTheDocument()
    expect(screen.queryByText('15:05')).not.toBeInTheDocument()
  })

  it('点击立即触发回调 onRun', () => {
    const props = baseProps()
    render(<PlanList {...props} />)
    fireEvent.click(screen.getByText('立即触发'))
    expect(props.onRun).toHaveBeenCalledWith('plan-1')
  })

  it('点击编辑回调 onEdit', () => {
    const props = baseProps()
    render(<PlanList {...props} />)
    fireEvent.click(screen.getByText('编辑'))
    expect(props.onEdit).toHaveBeenCalledWith('plan-1')
  })
})

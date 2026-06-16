// SchedulerStatusCard 示例测试：运行中/未运行/未知状态展示，以及 last_triggered 渲染（TSO Wave 4 / R6.3）。
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import SchedulerStatusCard from './SchedulerStatusCard'
import { makeSchedulerStatus } from './testFixtures'

describe('SchedulerStatusCard', () => {
  it('运行中状态展示运行中与轮询周期', () => {
    render(<SchedulerStatusCard status={makeSchedulerStatus({ running: true, tick_seconds: 30 })} />)
    expect(screen.getByText('运行中')).toBeInTheDocument()
    expect(screen.getByText('30 秒')).toBeInTheDocument()
  })

  it('未运行状态展示未运行', () => {
    render(<SchedulerStatusCard status={makeSchedulerStatus({ running: false })} />)
    expect(screen.getByText('未运行')).toBeInTheDocument()
  })

  it('无状态展示未知', () => {
    render(<SchedulerStatusCard status={null} />)
    expect(screen.getByText('调度器状态未知')).toBeInTheDocument()
  })

  it('last_triggered 为空时展示「暂无触发记录」', () => {
    render(<SchedulerStatusCard status={makeSchedulerStatus({ last_triggered: {} })} />)
    expect(screen.getByText('暂无触发记录')).toBeInTheDocument()
  })

  it('last_triggered 有条目时展示 plan_id 与日期', () => {
    render(
      <SchedulerStatusCard
        status={makeSchedulerStatus({
          last_triggered: { 'plan-1': '2026-06-12', 'plan-2': '2026-06-11' },
        })}
      />,
    )
    expect(screen.getByText('plan-1')).toBeInTheDocument()
    expect(screen.getByText('2026-06-12')).toBeInTheDocument()
    expect(screen.getByText('plan-2')).toBeInTheDocument()
    expect(screen.getByText('2026-06-11')).toBeInTheDocument()
  })

  it('既有「上次触发」标签行正常展示', () => {
    render(
      <SchedulerStatusCard
        status={makeSchedulerStatus({ last_triggered: { 'plan-x': '2026-06-10' } })}
      />,
    )
    expect(screen.getByText('上次触发')).toBeInTheDocument()
  })
})

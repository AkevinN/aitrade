// SchedulerStatusCard 示例测试：运行中/未运行/未知状态展示（Req 8.5）。
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
})

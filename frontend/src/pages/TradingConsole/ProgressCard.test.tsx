// ProgressCard 示例测试：覆盖进度联动（Req 6.2）与错误展示（Req 6.7）。
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import ProgressCard from './ProgressCard'
import { makeTask } from './testFixtures'

describe('ProgressCard', () => {
  it('尚未触发决策时给出占位提示', () => {
    render(<ProgressCard task={null} hasTaskId={false} />)
    expect(screen.getByText(/尚未触发决策/)).toBeInTheDocument()
  })

  it('已有 task_id 但任务对象未就绪时显示加载态', () => {
    render(<ProgressCard task={null} hasTaskId={true} />)
    expect(screen.getByText('加载中')).toBeInTheDocument()
    expect(screen.getByText(/正在获取任务状态/)).toBeInTheDocument()
  })

  // Req 6.2：进度更新时刷新进度显示与状态消息。
  it('运行中任务反映进度百分比与进度消息', () => {
    const task = makeTask({ status: 'running', progress: 45, message: '正在进行 CNN 推理' })
    const { container } = render(<ProgressCard task={task} hasTaskId={true} />)

    expect(screen.getByText('执行中')).toBeInTheDocument()
    expect(screen.getByText('正在进行 CNN 推理')).toBeInTheDocument()
    // antd Progress 将百分比渲染为 45%
    const progress = container.querySelector('.ant-progress')
    expect(progress).toBeTruthy()
    expect(container.textContent).toContain('45%')
  })

  // Req 6.2：完成态进度联动到 100% / 成功。
  it('完成态任务显示已完成与 100% 进度', () => {
    const task = makeTask({ status: 'completed', progress: 100, message: '决策完成' })
    const { container } = render(<ProgressCard task={task} hasTaskId={true} />)

    expect(screen.getByText('已完成')).toBeInTheDocument()
    // antd Progress 在 success 状态下渲染对勾图标而非百分比文本，断言成功态类名。
    expect(container.querySelector('.ant-progress-status-success')).toBeTruthy()
  })

  // Req 6.7：任务失败时展示错误消息。
  it('失败态任务展示错误消息', () => {
    const task = makeTask({
      status: 'failed',
      progress: 30,
      message: '决策日 2026-06-08 行情缺失',
    })
    render(<ProgressCard task={task} hasTaskId={true} />)

    expect(screen.getByText('失败')).toBeInTheDocument()
    expect(screen.getByText('决策任务失败')).toBeInTheDocument()
    expect(screen.getByText('决策日 2026-06-08 行情缺失')).toBeInTheDocument()
  })

  it('失败态但无消息时回退到默认错误文案', () => {
    const task = makeTask({ status: 'failed', message: '' })
    render(<ProgressCard task={task} hasTaskId={true} />)
    expect(screen.getByText(/任务执行失败/)).toBeInTheDocument()
  })
})

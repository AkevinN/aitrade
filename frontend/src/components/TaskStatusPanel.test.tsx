// TaskStatusPanel 示例测试：耗时展示、失败任务 traceback 折叠面板（TSO Wave 4 / R6.5）。
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

import TaskStatusPanel from './TaskStatusPanel'
import type { Task } from '../types/alpha'

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    task_id: 'task-1',
    type: 'live_decision',
    title: '今日决策',
    entity_type: 'live',
    entity_name: '000001.SZSE',
    status: 'completed',
    progress: 100,
    message: '',
    result: null,
    created_at: '2026-06-12T14:30:00',
    updated_at: '2026-06-12T14:31:00',
    ...overrides,
  }
}

describe('TaskStatusPanel', () => {
  it('task 为 null 时不渲染', () => {
    const { container } = render(<TaskStatusPanel task={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('展示 duration_ms 耗时（毫秒转换为秒，1 位小数）', () => {
    render(<TaskStatusPanel task={makeTask({ duration_ms: 2345 })} />)
    // 2345ms → 2.3 秒
    expect(screen.getByText('耗时 2.3 秒')).toBeInTheDocument()
  })

  it('duration_ms 为 null 时不显示耗时', () => {
    render(<TaskStatusPanel task={makeTask({ duration_ms: null })} />)
    expect(screen.queryByText(/耗时/)).not.toBeInTheDocument()
  })

  it('duration_ms 未定义时不显示耗时', () => {
    render(<TaskStatusPanel task={makeTask()} />)
    expect(screen.queryByText(/耗时/)).not.toBeInTheDocument()
  })

  it('失败任务有 error_traceback 时展示折叠面板「错误堆栈」', () => {
    render(
      <TaskStatusPanel
        task={makeTask({
          status: 'failed',
          error_traceback: 'Traceback (most recent call last):\n  File "foo.py", line 1, in bar\nValueError: bad',
        })}
      />,
    )
    expect(screen.getByText('错误堆栈')).toBeInTheDocument()
  })

  it('点击折叠面板展开后展示堆栈内容', () => {
    render(
      <TaskStatusPanel
        task={makeTask({
          status: 'failed',
          error_traceback: 'ValueError: something went wrong',
        })}
      />,
    )
    // 点击展开
    fireEvent.click(screen.getByText('错误堆栈'))
    expect(screen.getByText('ValueError: something went wrong')).toBeInTheDocument()
  })

  it('失败任务 error_traceback 为空时不展示折叠面板', () => {
    render(<TaskStatusPanel task={makeTask({ status: 'failed', error_traceback: '' })} />)
    expect(screen.queryByText('错误堆栈')).not.toBeInTheDocument()
  })

  it('完成态任务无 traceback 时不展示折叠面板', () => {
    render(
      <TaskStatusPanel
        task={makeTask({ status: 'completed', result: { ok: true } })}
      />,
    )
    expect(screen.queryByText('错误堆栈')).not.toBeInTheDocument()
  })
})

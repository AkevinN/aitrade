// RiskDetailPanel 示例测试：覆盖风控明细逐项展示（Req 6.4）。
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import RiskDetailPanel from './RiskDetailPanel'
import { makeTask, makeResult, makeRiskDetail } from './testFixtures'

describe('RiskDetailPanel', () => {
  it('任务未完成时给出占位', () => {
    render(<RiskDetailPanel task={null} />)
    expect(screen.getByText(/决策完成后将在此展示/)).toBeInTheDocument()
  })

  // Req 6.4：逐项展示 5 个风控检查项与通过/拦截结果。
  it('完成态展示全部 5 项风控检查', () => {
    const task = makeTask({ status: 'completed', result: makeResult() })
    render(<RiskDetailPanel task={task} />)

    expect(screen.getByText('Kill-switch / 熔断')).toBeInTheDocument()
    expect(screen.getByText('黑名单')).toBeInTheDocument()
    expect(screen.getByText('停牌 / 涨跌停封死')).toBeInTheDocument()
    expect(screen.getByText('总仓位上限')).toBeInTheDocument()
    expect(screen.getByText('单票仓位上限')).toBeInTheDocument()
    // 全部通过
    expect(screen.getAllByText('通过')).toHaveLength(5)
  })

  it('存在拦截项时展示「拦截」标签与明细', () => {
    const riskDetail = makeRiskDetail([
      { check: 'kill_switch_or_circuit', passed: true, detail: '未触发' },
      { check: 'blacklist', passed: false, detail: '命中黑名单 000001.SZSE' },
    ])
    const task = makeTask({
      status: 'completed',
      result: makeResult({ risk_detail: riskDetail }),
    })
    render(<RiskDetailPanel task={task} />)

    expect(screen.getByText('拦截')).toBeInTheDocument()
    expect(screen.getByText('命中黑名单 000001.SZSE')).toBeInTheDocument()
  })

  it('幂等命中且无明细时给出友好占位', () => {
    const task = makeTask({
      status: 'completed',
      result: makeResult({ idempotent_hit: true, risk_detail: [] }),
    })
    render(<RiskDetailPanel task={task} />)
    expect(screen.getByText(/幂等命中/)).toBeInTheDocument()
  })
})

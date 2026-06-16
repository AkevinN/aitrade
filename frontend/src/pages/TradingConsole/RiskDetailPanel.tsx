import React from 'react'
import { Empty, Table, Tag, Typography } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

import type { Task } from '../../types/alpha'
import type { LiveDecisionResult, RiskDetailItem } from '../../types/live'

const { Text } = Typography

/**
 * {@link RiskDetailPanel} 组件 props。
 */
interface RiskDetailPanelProps {
  /** 当前决策任务（来自 useTask 订阅）。任务完成后读取 task.result.risk_detail 渲染。 */
  task?: Task | null
}

/** 风控检查项 key → 可读中文标签。与后端 RiskInspector 记录的 check 名称对齐。 */
const CHECK_LABELS: Record<string, string> = {
  kill_switch_or_circuit: 'Kill-switch / 熔断',
  blacklist: '黑名单',
  halted: '停牌 / 涨跌停封死',
  max_total_position: '总仓位上限',
  max_single_position: '单票仓位上限',
}

/**
 * 风控明细面板（任务 9.5）。
 *
 * 任务完成后读取 task.result.risk_detail（RiskDetailItem[]），逐项展示：
 * 检查项名称（已知 5 项映射为中文标签）、通过/拦截指示（绿勾/红叉 Tag）、明细文本（Req 6.4）。
 * 幂等命中时 risk_detail 为空数组，给出友好占位。
 */
const RiskDetailPanel: React.FC<RiskDetailPanelProps> = ({ task }) => {
  // task.result 在 Task 类型中为宽松的 Record，按本特性结果结构断言（与 DecisionResultCard 同约定）。
  const result = task?.result as LiveDecisionResult | undefined
  const riskDetail: RiskDetailItem[] = result?.risk_detail ?? []

  // 任务尚未完成或无结果：占位。
  if (task?.status !== 'completed' || !result) {
    return (
      <Empty
        description={
          task?.status === 'failed'
            ? '决策任务失败，未产出风控明细'
            : '决策完成后将在此展示各项风控检查与结果'
        }
      />
    )
  }

  // 幂等命中：未重新走风控，risk_detail 为空。
  if (riskDetail.length === 0) {
    return (
      <Empty
        description={
          result.idempotent_hit
            ? '幂等命中：返回首次决策，本次未重新执行风控检查'
            : '本次决策无风控明细'
        }
      />
    )
  }

  const columns: ColumnsType<RiskDetailItem> = [
    {
      title: '检查项',
      dataIndex: 'check',
      key: 'check',
      width: 160,
      render: (check: string) => CHECK_LABELS[check] ?? check,
    },
    {
      title: '结果',
      dataIndex: 'passed',
      key: 'passed',
      width: 100,
      render: (passed: boolean) =>
        passed ? (
          <Tag color="green" icon={<CheckCircleOutlined />}>
            通过
          </Tag>
        ) : (
          <Tag color="red" icon={<CloseCircleOutlined />}>
            拦截
          </Tag>
        ),
    },
    {
      title: '明细',
      dataIndex: 'detail',
      key: 'detail',
      render: (detail: string) => <Text type="secondary">{detail || '—'}</Text>,
    },
  ]

  return (
    <Table<RiskDetailItem>
      size="small"
      rowKey="check"
      columns={columns}
      dataSource={riskDetail}
      pagination={false}
    />
  )
}

export default RiskDetailPanel

import React from 'react'
import { Badge, Descriptions, Spin } from 'antd'

import type { SchedulerStatus } from '../../types/live'

interface SchedulerStatusCardProps {
  status?: SchedulerStatus | null
  loading?: boolean
}

/**
 * 调度器状态卡片（任务 11）：展示运行状态、轮询周期、启用计划数。
 */
const SchedulerStatusCard: React.FC<SchedulerStatusCardProps> = ({ status, loading }) => {
  if (loading && !status) {
    return <Spin />
  }
  if (!status) {
    return <Badge status="default" text="调度器状态未知" />
  }
  return (
    <Descriptions size="small" column={1}>
      <Descriptions.Item label="运行状态">
        {status.running ? (
          <Badge status="processing" text="运行中" />
        ) : (
          <Badge status="default" text="未运行" />
        )}
      </Descriptions.Item>
      <Descriptions.Item label="轮询周期">{status.tick_seconds} 秒</Descriptions.Item>
      <Descriptions.Item label="启用计划数">{status.enabled_plan_count}</Descriptions.Item>
    </Descriptions>
  )
}

export default SchedulerStatusCard

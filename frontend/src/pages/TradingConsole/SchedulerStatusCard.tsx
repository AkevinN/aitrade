import React from 'react'
import { Badge, Descriptions, Spin, Typography } from 'antd'

import type { SchedulerStatus } from '../../types/live'

const { Text } = Typography

/** 调度器状态卡片的 Props。 */
interface SchedulerStatusCardProps {
  /** 调度器运行状态；undefined/null 时显示加载态或「未知」 */
  status?: SchedulerStatus | null
  /** 是否正在加载（首次拉取时显示 Spin） */
  loading?: boolean
}

/**
 * 调度器状态卡片（任务 11 / TSO Wave 4）。
 *
 * 展示运行状态、轮询周期、启用计划数，以及各计划的上次触发日期（R6.3）。
 * PlanManager 以 5 秒 refetchInterval 轮询，此组件无需内部定时器。
 *
 * @param status - 调度器状态快照
 * @param loading - 首次加载标志
 */
const SchedulerStatusCard: React.FC<SchedulerStatusCardProps> = ({ status, loading }) => {
  if (loading && !status) {
    return <Spin />
  }
  if (!status) {
    return <Badge status="default" text="调度器状态未知" />
  }

  const lastTriggeredEntries = Object.entries(status.last_triggered ?? {})

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
      <Descriptions.Item label="上次触发">
        {lastTriggeredEntries.length === 0 ? (
          <Text type="secondary">暂无触发记录</Text>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {lastTriggeredEntries.map(([planId, date]) => (
              <div key={planId} style={{ display: 'flex', gap: 8 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>{planId}</Text>
                <Text style={{ fontSize: 12 }}>{date}</Text>
              </div>
            ))}
          </div>
        )}
      </Descriptions.Item>
    </Descriptions>
  )
}

export default SchedulerStatusCard

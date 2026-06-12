import React, { useState } from 'react'
import { Button, Card, Select, Space, Table, Tag, Typography } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import type { ColumnsType } from 'antd/es/table'

import { liveService } from '../../api/liveApi'
import type { SchedulerRunEvent } from '../../types/live'
import type { TradingPlanSummary } from '../../types/live'

const { Text } = Typography

// Skip_Reason 中文映射（R6.4）
const SKIP_REASON_LABEL: Record<string, string> = {
  not_trading_day: '非交易日',
  schedule_gate: '未到调度日',
  already_done: '今日已触发',
  degraded: '降级暂停',
  data_lag: '行情滞后',
  disabled: '计划停用',
}

/**
 * 将 ISO 时间戳格式化为 HH:mm:ss 字符串（仅展示时分秒）。
 *
 * @param ts - ISO 8601 时间字符串
 * @returns "HH:mm:ss" 格式的本地时间；解析失败时返回原始字符串
 */
function formatTs(ts: string): string {
  try {
    const d = new Date(ts)
    const hh = String(d.getHours()).padStart(2, '0')
    const mm = String(d.getMinutes()).padStart(2, '0')
    const ss = String(d.getSeconds()).padStart(2, '0')
    return `${hh}:${mm}:${ss}`
  } catch {
    return ts
  }
}

/**
 * 根据事件类型返回 Ant Design Tag 颜色。
 *
 * @param event - "trigger" | "skip" | "error" 或其他
 * @returns Ant Design color 字符串
 */
function eventTagColor(event: string): string {
  if (event === 'trigger') return 'green'
  if (event === 'skip') return 'orange'
  if (event === 'error') return 'red'
  return 'default'
}

/**
 * 将调度事件行拼合为可读的原因说明文本。
 *
 * - skip：Skip_Reason 中文映射 + detail
 * - trigger：调度时点（slot）+ detail
 * - error：error 字段
 *
 * @param row - 调度运行事件对象
 * @returns 拼合后的说明文本；无内容时返回空字符串
 */
function reasonDescription(row: SchedulerRunEvent): string {
  if (row.event === 'skip') {
    const label = row.reason ? (SKIP_REASON_LABEL[row.reason] ?? row.reason) : ''
    return [label, row.detail].filter(Boolean).join(' · ')
  }
  if (row.event === 'trigger') {
    return [row.slot ? `时点 ${row.slot}` : '', row.detail].filter(Boolean).join(' · ')
  }
  if (row.event === 'error') {
    return row.error ?? ''
  }
  return ''
}

interface SchedulerRunsCardProps {
  /** 可供过滤的计划列表（来自 PlanManager 的 plansQuery）。 */
  plans?: TradingPlanSummary[]
}

/**
 * 调度日志卡片（TSO Wave 4 / R6.4）：
 * 展示当日调度运行事件（触发/跳过/错误），支持按计划过滤，手动刷新。
 */
const SchedulerRunsCard: React.FC<SchedulerRunsCardProps> = ({ plans = [] }) => {
  const [selectedPlanId, setSelectedPlanId] = useState<string | undefined>(undefined)

  const runsQuery = useQuery({
    queryKey: ['scheduler-runs', selectedPlanId],
    queryFn: () => liveService.getSchedulerRuns({ plan_id: selectedPlanId }),
  })

  const columns: ColumnsType<SchedulerRunEvent> = [
    {
      title: '时刻',
      key: 'ts',
      width: 80,
      render: (_, row) => (
        <Text style={{ fontSize: 12, fontFamily: 'monospace' }}>{formatTs(row.ts)}</Text>
      ),
    },
    {
      title: '计划',
      dataIndex: 'plan_id',
      key: 'plan_id',
      width: 120,
      render: (planId: string) => {
        const plan = plans.find((p) => p.plan_id === planId)
        return (
          <Text style={{ fontSize: 12 }} title={planId}>
            {plan ? plan.name : planId}
          </Text>
        )
      },
    },
    {
      title: '事件',
      dataIndex: 'event',
      key: 'event',
      width: 70,
      render: (event: string) => (
        <Tag color={eventTagColor(event)} style={{ fontSize: 11 }}>
          {event}
        </Tag>
      ),
    },
    {
      title: '原因说明',
      key: 'reason',
      render: (_, row) => {
        const desc = reasonDescription(row)
        return desc ? (
          <Text type="secondary" style={{ fontSize: 12 }}>
            {desc}
          </Text>
        ) : null
      },
    },
  ]

  const planOptions = [
    { value: '', label: '全部计划' },
    ...plans.map((p) => ({ value: p.plan_id, label: p.name })),
  ]

  return (
    <Card
      title="调度日志"
      size="small"
      extra={
        <Space>
          <Select
            size="small"
            style={{ width: 140 }}
            value={selectedPlanId ?? ''}
            options={planOptions}
            onChange={(v: string) => setSelectedPlanId(v === '' ? undefined : v)}
          />
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={runsQuery.isFetching}
            onClick={() => void runsQuery.refetch()}
          >
            刷新
          </Button>
        </Space>
      }
    >
      <Table<SchedulerRunEvent>
        size="small"
        rowKey={(r) => `${r.ts}-${r.event}-${r.plan_id}`}
        columns={columns}
        dataSource={runsQuery.data ?? []}
        loading={runsQuery.isLoading}
        pagination={{ pageSize: 20, size: 'small', hideOnSinglePage: true }}
        locale={{ emptyText: '暂无调度日志' }}
      />
    </Card>
  )
}

export default SchedulerRunsCard

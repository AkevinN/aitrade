import React from 'react'
import { Button, Empty, Popconfirm, Space, Switch, Table, Tag, Typography } from 'antd'
import { EditOutlined, DeleteOutlined, ThunderboltOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

import type { TradingPlanSummary } from '../../types/live'
import { isIntradayBarFreq } from '../../utils/barFreq'

const { Text } = Typography

interface PlanListProps {
  plans: TradingPlanSummary[]
  loading?: boolean
  /** 启停切换中的 plan_id（行内 Switch loading）。 */
  togglingId?: string | null
  /** 触发中的 plan_id（行内触发按钮 loading）。 */
  runningId?: string | null
  onToggle: (planId: string, enabled: boolean) => void
  onRun: (planId: string) => void
  onEdit: (planId: string) => void
  onDelete: (planId: string) => void
}

/**
 * 交易计划列表（任务 11）：展示计划摘要，支持启停、立即触发、编辑、删除。
 */
const PlanList: React.FC<PlanListProps> = ({
  plans,
  loading,
  togglingId,
  runningId,
  onToggle,
  onRun,
  onEdit,
  onDelete,
}) => {
  const columns: ColumnsType<TradingPlanSummary> = [
    {
      title: '计划',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, row) => (
        <Space direction="vertical" size={0}>
          <Text strong>{name}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {row.vt_symbol} · {row.scheme}
          </Text>
        </Space>
      ),
    },
    {
      title: '调度',
      key: 'trigger_times',
      render: (_, row) => {
        // 日内计划走盘中监控模式（按 bar 网格自动调度），无固定唤醒时刻。
        if (isIntradayBarFreq(row.bar_freq)) {
          return <Tag color="purple">盘中监控 · {row.bar_freq}</Tag>
        }
        const times = row.trigger_times ?? []
        return (
          <Space size={4} wrap>
            <Tag color="geekblue">{row.bar_freq}</Tag>
            {times.map((t) => (
              <Tag color="blue" key={t}>
                {t}
              </Tag>
            ))}
          </Space>
        )
      },
    },
    {
      title: '最近触发',
      dataIndex: 'last_triggered',
      key: 'last_triggered',
      render: (d?: string | null) => (d ? <Text>{d}</Text> : <Text type="secondary">—</Text>),
    },
    {
      title: '启用',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, row) => (
        <Switch
          size="small"
          checked={enabled}
          loading={togglingId === row.plan_id}
          onChange={(checked) => onToggle(row.plan_id, checked)}
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, row) => (
        <Space>
          <Button
            size="small"
            icon={<ThunderboltOutlined />}
            loading={runningId === row.plan_id}
            onClick={() => onRun(row.plan_id)}
          >
            立即触发
          </Button>
          <Button size="small" icon={<EditOutlined />} onClick={() => onEdit(row.plan_id)}>
            编辑
          </Button>
          <Popconfirm title="确认删除该计划？" onConfirm={() => onDelete(row.plan_id)} okText="删除" cancelText="取消">
            <Button size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  if (!loading && plans.length === 0) {
    return <Empty description="暂无交易计划，点击右上角新建" />
  }

  return (
    <Table<TradingPlanSummary>
      rowKey="plan_id"
      size="small"
      loading={loading}
      columns={columns}
      dataSource={plans}
      pagination={false}
    />
  )
}

export default PlanList

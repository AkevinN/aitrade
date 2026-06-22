import React, { useState } from 'react'
import { Alert, Empty, Segmented, Space, Table, Tag, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'

import type { Task } from '../../types/alpha'
import { useRunHistory, type RunHistoryCategory } from '../../hooks/useRunHistory'
import RunDetailDrawer from './RunDetailDrawer'
import { runCategory, runCategoryLabel } from './runTypes'

const { Title, Text } = Typography

/** 状态 → Tag 颜色。 */
const statusColor: Record<string, string> = {
  completed: 'green',
  failed: 'red',
  running: 'blue',
  pending: 'default',
}

/** 毫秒 → 可读耗时。 */
function formatDuration(ms?: number | null): string {
  if (ms == null) return '—'
  return `${(ms / 1000).toFixed(1)} 秒`
}

/** ISO 串 → 本地时间。 */
function formatTime(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('zh-CN')
}

/**
 * 运行历史页：列出过去的回测 / CNN 选股运行，点开回看完整结果。
 *
 * 数据源是既有任务历史（回测无专门库、选股 persist 默认关闭，二者都靠任务历史持久化）。
 * 纯只读：仅 GET 任务历史，不触发任何重跑/训练/写库。点击行打开右侧详情抽屉。
 */
const RunHistory: React.FC = () => {
  const [category, setCategory] = useState<RunHistoryCategory>('all')
  const [selected, setSelected] = useState<Task | null>(null)

  const { data: runs, isLoading, isError } = useRunHistory(category)

  const columns: ColumnsType<Task> = [
    {
      title: '运行时间',
      dataIndex: 'created_at',
      width: 180,
      render: (v: string) => formatTime(v),
    },
    {
      title: '类别',
      key: 'category',
      width: 90,
      render: (_, t) => <Tag>{runCategoryLabel(runCategory(t.type))}</Tag>,
    },
    {
      title: '名称',
      key: 'name',
      render: (_, t) => <Text>{t.title || t.entity_name || t.task_id}</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      width: 110,
      render: (s: string) => <Tag color={statusColor[s] ?? 'default'}>{s}</Tag>,
    },
    {
      title: '耗时',
      dataIndex: 'duration_ms',
      width: 100,
      render: (v?: number | null) => formatDuration(v),
    },
    {
      title: '操作',
      key: 'action',
      width: 90,
      render: (_, t) => (
        <a
          onClick={(e) => {
            e.stopPropagation()
            setSelected(t)
          }}
        >
          查看
        </a>
      ),
    },
  ]

  return (
    <section className="panel">
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Title level={4} style={{ margin: 0 }}>
          运行历史
        </Title>
        <Text type="secondary">
          回看过去的回测与 CNN 选股运行（来自任务历史，永久保存）。点击任意一行查看完整结果。
        </Text>

        <Segmented
          value={category}
          onChange={(v) => setCategory(v as RunHistoryCategory)}
          options={[
            { label: '全部', value: 'all' },
            { label: '回测', value: 'backtest' },
            { label: '选股', value: 'screening' },
          ]}
        />

        {isError ? (
          <Alert type="error" showIcon message="加载运行历史失败，请稍后重试" />
        ) : null}

        <Table<Task>
          rowKey="task_id"
          size="small"
          loading={isLoading}
          columns={columns}
          dataSource={runs ?? []}
          onRow={(t) => ({ onClick: () => setSelected(t), style: { cursor: 'pointer' } })}
          locale={{
            emptyText: <Empty description="暂无运行历史" />,
          }}
          pagination={{ pageSize: 20, hideOnSinglePage: true }}
        />
      </Space>

      <RunDetailDrawer task={selected} onClose={() => setSelected(null)} />
    </section>
  )
}

export default RunHistory

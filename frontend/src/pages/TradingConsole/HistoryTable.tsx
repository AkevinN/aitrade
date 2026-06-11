import React, { useEffect, useState } from 'react'
import {
  Alert,
  App,
  Button,
  Descriptions,
  Empty,
  Modal,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import { DeleteOutlined, ReloadOutlined } from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { useMutation, useQuery } from '@tanstack/react-query'

import { liveService } from '../../api/liveApi'
import type { Task } from '../../types/alpha'
import type { Decision } from '../../types/live'
import DecisionTracePanel from './DecisionTracePanel'

const { Text } = Typography

interface HistoryTableProps {
  /**
   * 当前决策任务（来自 useTask 订阅）。当任务完成时刷新历史列表，
   * 以便新落盘的决策即时出现在表格中（设计中的可选刷新）。
   */
  task?: Task | null
}

/** 决策动作 → 标签颜色 / 文案。与 DecisionResultCard 约定一致。 */
const ACTION_META: Record<string, { color: string; label: string }> = {
  buy: { color: 'red', label: '买入' },
  sell: { color: 'green', label: '卖出' },
  hold: { color: 'default', label: '观望' },
}

/** 列表行模型：仅含 signal_id。 */
interface HistoryRow {
  signal_id: string
}

/**
 * 历史决策表（任务 9.6）。
 *
 * - useQuery 拉 GET /api/live/decisions，得到 signal_id 集合并渲染为 antd Table（Req 6.5）。
 * - 点击行经第二个 useQuery（enabled on selection）调 getDecision 拉取完整 Decision，
 *   在 Modal 中复用与 DecisionResultCard 类似的字段展示。
 * - 任务完成后自动刷新列表，使新决策即时出现。
 */
const HistoryTable: React.FC<HistoryTableProps> = ({ task }) => {
  const { message } = App.useApp()
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // 列表查询：signal_id 集合。
  const {
    data: signalIds,
    isLoading: listLoading,
    isError: listError,
    refetch: refetchList,
    isFetching: listFetching,
  } = useQuery({
    queryKey: ['live-decisions'],
    queryFn: () => liveService.listDecisions(),
  })

  // 任务完成 → 刷新列表（新落盘的决策应出现在历史中）。
  useEffect(() => {
    if (task?.status === 'completed') {
      void refetchList()
    }
  }, [task?.status, refetchList])

  // 详情查询：仅在选中某行时启用（懒加载）。
  const {
    data: detail,
    isLoading: detailLoading,
    isError: detailError,
  } = useQuery({
    queryKey: ['live-decision', selectedId],
    queryFn: () => liveService.getDecision(selectedId as string),
    enabled: selectedId !== null,
  })

  // 归档式删除：决策 + trace 整体移入 archive/，解除幂等占位（同一 bar 可重新决策与提醒）。
  const deleteMutation = useMutation({
    mutationFn: (signalId: string) => liveService.deleteDecision(signalId),
    onSuccess: (_data, signalId) => {
      message.success('决策已删除（归档），同一 bar 可重新产出决策')
      if (selectedId === signalId) setSelectedId(null)
      void refetchList()
    },
    onError: (e: unknown) =>
      message.error(e instanceof Error ? e.message : '删除失败，请重试'),
  })

  const rows: HistoryRow[] = (signalIds ?? []).map((id) => ({ signal_id: id }))

  const columns: ColumnsType<HistoryRow> = [
    {
      title: 'signal_id',
      dataIndex: 'signal_id',
      key: 'signal_id',
      render: (id: string) => (
        <Text code style={{ fontSize: 12 }}>
          {id}
        </Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 72,
      align: 'center',
      render: (_, record) => (
        // 冒泡屏障：Popconfirm 弹层经 React portal 沿组件树冒泡，需在单元格层
        // 拦截 click，避免删除操作触发行点击（打开详情弹窗）。
        <span onClick={(e) => e.stopPropagation()}>
          <Popconfirm
            title="删除该决策？"
            description="决策与过程档案将移入归档；同一 bar 之后可重新产出决策与提醒。"
            okText="确认删除"
            cancelText="取消"
            okButtonProps={{ danger: true }}
            onConfirm={() => deleteMutation.mutate(record.signal_id)}
          >
            <Button
              size="small"
              type="text"
              danger
              aria-label={`删除决策 ${record.signal_id}`}
              icon={<DeleteOutlined />}
              loading={
                deleteMutation.isPending &&
                deleteMutation.variables === record.signal_id
              }
            />
          </Popconfirm>
        </span>
      ),
    },
  ]

  if (listError) {
    return (
      <Alert
        type="error"
        showIcon
        message="历史决策加载失败"
        description="无法从 /api/live/decisions 获取决策列表，请稍后重试。"
        action={
          <Button size="small" onClick={() => void refetchList()}>
            重试
          </Button>
        }
      />
    )
  }

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={listFetching}
          onClick={() => void refetchList()}
        >
          刷新
        </Button>
      </Space>

      <Table<HistoryRow>
        size="small"
        rowKey="signal_id"
        loading={listLoading}
        columns={columns}
        dataSource={rows}
        pagination={{ pageSize: 8, hideOnSinglePage: true }}
        locale={{
          emptyText: <Empty description="暂无历史决策记录" />,
        }}
        onRow={(record) => ({
          onClick: () => setSelectedId(record.signal_id),
          style: { cursor: 'pointer' },
        })}
      />

      <Modal
        open={selectedId !== null}
        title="决策详情"
        footer={null}
        onCancel={() => setSelectedId(null)}
        width={640}
      >
        {detailError ? (
          <Alert
            type="error"
            showIcon
            message="决策详情加载失败"
            description={`未能加载 ${selectedId} 的决策详情。`}
          />
        ) : (
          <DecisionDetail decision={detail} loading={detailLoading} />
        )}
      </Modal>
    </Space>
  )
}

/** 数字字段安全格式化：null/undefined → 占位符。 */
function fmtNumber(value: number | null | undefined, suffix = ''): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value}${suffix}`
}

interface DecisionDetailProps {
  decision?: Decision
  loading: boolean
}

/** Modal 内的决策字段展示，复用 DecisionResultCard 的字段口径。 */
const DecisionDetail: React.FC<DecisionDetailProps> = ({ decision, loading }) => {
  if (loading || !decision) {
    return <Empty description={loading ? '加载中…' : '无决策数据'} />
  }

  const actionMeta = ACTION_META[decision.action] || {
    color: 'default',
    label: decision.action,
  }

  const signalPercent =
    decision.signal === null || decision.signal === undefined
      ? null
      : decision.signal * 100

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Descriptions size="small" column={1} bordered>
        <Descriptions.Item label="决策动作">
          <Tag color={actionMeta.color}>{actionMeta.label}</Tag>
        </Descriptions.Item>
        <Descriptions.Item label="目标标的">{decision.vt_symbol || '—'}</Descriptions.Item>
        <Descriptions.Item label="建议手数">{fmtNumber(decision.volume, ' 股')}</Descriptions.Item>
        <Descriptions.Item label="建议价位">
          {decision.price === null || decision.price === undefined
            ? '—'
            : decision.price.toFixed(2)}
        </Descriptions.Item>
        <Descriptions.Item label="信号概率">
          {signalPercent === null ? '—' : `${signalPercent.toFixed(2)}%`}
        </Descriptions.Item>
        <Descriptions.Item label="reason">{decision.reason || '—'}</Descriptions.Item>
        <Descriptions.Item label="方案">{decision.scheme}</Descriptions.Item>
        <Descriptions.Item label="决策时刻">{decision.as_of}</Descriptions.Item>
        <Descriptions.Item label="创建时间">{decision.created_at}</Descriptions.Item>
        <Descriptions.Item label="signal_id">
          <Text copyable code style={{ fontSize: 12 }}>
            {decision.signal_id}
          </Text>
        </Descriptions.Item>
      </Descriptions>

      {/* 历史决策详情中的过程档案（任务 20.2）：懒加载该 signal_id 的六段 trace。 */}
      <DecisionTracePanel signalId={decision.signal_id} />
    </Space>
  )
}
export default HistoryTable

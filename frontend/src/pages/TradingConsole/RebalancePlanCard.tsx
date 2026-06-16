// 规则调仓结果卡片——展示 RebalanceDecision 的调仓清单、风控摘要与确认操作。
// 与 DecisionResultCard 并列，不扩展它（各自契约独立，由测试锁定）。
import React, { useState } from 'react'
import {
  Alert,
  App,
  Button,
  Descriptions,
  Empty,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons'

import type { Task } from '../../types/alpha'
import type { RebalanceDecision, RebalanceItem, RebalanceResult } from '../../types/live'
import { liveService } from '../../api/liveApi'

const { Text } = Typography

interface RebalancePlanCardProps {
  /** 当前调仓任务（来自 useTask 订阅）。任务完成后读取 task.result 渲染。 */
  task?: Task | null
}

/** 调仓清单表格列定义。 */
const REBALANCE_COLUMNS: ColumnsType<RebalanceItem> = [
  {
    title: '标的',
    dataIndex: 'vt_symbol',
    key: 'vt_symbol',
    render: (v: string) => <Text code>{v}</Text>,
  },
  {
    title: '方向',
    dataIndex: 'action',
    key: 'action',
    render: (action: string) =>
      action === 'buy' ? (
        <Tag color="red">买入</Tag>
      ) : (
        <Tag color="green">卖出</Tag>
      ),
  },
  {
    title: '数量（股）',
    dataIndex: 'volume',
    key: 'volume',
    align: 'right',
    render: (v: number) => v.toLocaleString(),
  },
  {
    title: '参考价',
    dataIndex: 'price',
    key: 'price',
    align: 'right',
    render: (v: number) => (v !== null && v !== undefined ? v.toFixed(2) : '—'),
  },
  {
    title: '信号值',
    dataIndex: 'signal',
    key: 'signal',
    align: 'right',
    render: (v: number | null | undefined) =>
      v !== null && v !== undefined ? v.toFixed(4) : '—',
  },
  {
    title: '原因',
    dataIndex: 'reason',
    key: 'reason',
    ellipsis: true,
  },
]

/**
 * 规则调仓结果卡片（任务 3.8）。
 *
 * 任务完成后读取 task.result（RebalanceResult），展示：
 * - 调仓清单（Table：标的/方向/数量/价格/信号/原因）
 * - 风控摘要（risk_summary 逐条，passed=false 红色警示）
 * - 幂等命中 Tag
 * - skipped_reason 空态
 * - 确认执行按钮（status==='proposed' → Popconfirm → confirmRebalance → 回填提示）
 * - 底部 Alert「仅提醒，不自动下单」
 */
const RebalancePlanCard: React.FC<RebalancePlanCardProps> = ({ task }) => {
  const { message } = App.useApp()
  const [confirming, setConfirming] = useState(false)
  const [confirmedDecision, setConfirmedDecision] = useState<RebalanceDecision | null>(null)

  const result = task?.result as RebalanceResult | undefined

  const notice = (
    <Alert
      type="info"
      showIcon
      message="仅提醒，不自动下单"
      description="本调仓清单仅为人工执行建议，系统不会向任何券商网关提交真实订单。确认后将回填账本持仓记录。"
    />
  )

  if (task?.status !== 'completed' || !result) {
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Empty
          description={
            task?.status === 'failed'
              ? '调仓任务失败，未产出决策结果'
              : '调仓完成后将在此展示调仓清单'
          }
        />
        {notice}
      </Space>
    )
  }

  // skipped 状态：无决策
  if (!result.decision && result.skipped_reason) {
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Empty description={`已跳过：${result.skipped_reason}`} />
        {notice}
      </Space>
    )
  }

  if (!result.decision) {
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Empty description="本次未产出调仓决策" />
        {notice}
      </Space>
    )
  }

  const decision = confirmedDecision ?? result.decision

  /**
   * 确认调仓执行并回填账本。
   *
   * 调用 `liveService.confirmRebalance`，成功后将返回的最新决策存入 confirmedDecision 以刷新 UI。
   * 409 表示已确认或卖超（重复确认），404 表示决策不存在；两者分别给出明确提示。
   */
  const handleConfirm = async () => {
    setConfirming(true)
    try {
      const res = await liveService.confirmRebalance(decision.signal_id)
      setConfirmedDecision(res.decision)
      void message.success('已确认执行，账本持仓已回填。')
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string }; status?: number }; message?: string }
      const detail = err?.response?.data?.detail
      const status = err?.response?.status
      if (status === 409) {
        void message.error(`确认失败（409）：${detail ?? '已确认或卖超，无法重复确认'}`)
      } else if (status === 404) {
        void message.error('调仓决策不存在（404）')
      } else {
        void message.error(detail ?? err?.message ?? '确认失败')
      }
    } finally {
      setConfirming(false)
    }
  }

  const currentStatus = decision.status
  const isConfirmed = currentStatus === 'confirmed'

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {/* 幂等命中标记 */}
      {result.idempotent_hit && (
        <Tag color="gold" style={{ marginBottom: 4 }}>幂等命中（本次返回首次调仓决策，未重新计算）</Tag>
      )}

      {/* 决策元信息 */}
      <Descriptions size="small" column={2} bordered>
        <Descriptions.Item label="signal_id">
          <Text copyable code style={{ fontSize: 12 }}>
            {decision.signal_id}
          </Text>
        </Descriptions.Item>
        <Descriptions.Item label="组合 ID">
          <Text code>{decision.portfolio_id}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="决策时刻">{decision.as_of}</Descriptions.Item>
        <Descriptions.Item label="状态">
          {isConfirmed ? (
            <Tag color="blue" icon={<CheckCircleOutlined />}>已确认</Tag>
          ) : (
            <Tag color="orange">待确认</Tag>
          )}
        </Descriptions.Item>
      </Descriptions>

      {/* 调仓清单 */}
      <Table<RebalanceItem>
        size="small"
        rowKey="vt_symbol"
        columns={REBALANCE_COLUMNS}
        dataSource={decision.items}
        pagination={false}
        locale={{ emptyText: '暂无调仓指令' }}
      />

      {/* 风控摘要 */}
      {decision.risk_summary && decision.risk_summary.length > 0 && (
        <Space direction="vertical" size={4} style={{ width: '100%' }}>
          <Text strong>风控摘要</Text>
          {decision.risk_summary.map((item, idx) => (
            <Space key={idx} size={8}>
              {item.passed ? (
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              ) : (
                <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
              )}
              <Text type={item.passed ? undefined : 'danger'}>
                {item.check}：{item.detail}
              </Text>
            </Space>
          ))}
        </Space>
      )}

      {/* 确认执行按钮 */}
      {isConfirmed ? (
        <Alert
          type="success"
          showIcon
          message="已确认执行"
          description="账本持仓已按调仓清单回填。"
        />
      ) : (
        <Popconfirm
          title="确认已按清单人工执行完毕？"
          description="确认后将回填账本持仓记录，不可撤销。"
          onConfirm={() => void handleConfirm()}
          okText="确认回填"
          cancelText="取消"
        >
          <Button type="primary" loading={confirming}>
            确认执行
          </Button>
        </Popconfirm>
      )}

      {notice}
    </Space>
  )
}

export default RebalancePlanCard

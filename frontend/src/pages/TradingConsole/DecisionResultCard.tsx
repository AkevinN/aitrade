import React from 'react'
import { Alert, Descriptions, Empty, Space, Statistic, Tag, Typography } from 'antd'
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons'

import type { Task } from '../../types/alpha'
import type { Decision, LiveDecisionResult } from '../../types/live'
import DecisionTracePanel from './DecisionTracePanel'

const { Text } = Typography

interface DecisionResultCardProps {
  /** 当前决策任务（来自 useTask 订阅）。任务完成后读取 task.result.decision 渲染。 */
  task?: Task | null
}

/** 决策动作 → 标签颜色 / 文案 / 图标。 */
const ACTION_META: Record<
  string,
  { color: string; label: string; icon: React.ReactNode }
> = {
  buy: { color: 'red', label: '买入', icon: <ArrowUpOutlined /> },
  sell: { color: 'green', label: '卖出', icon: <ArrowDownOutlined /> },
  hold: { color: 'default', label: '观望', icon: <PauseCircleOutlined /> },
}

/**
 * 安全格式化数字字段，`null`/`undefined`/`NaN` 时返回「—」占位符。
 *
 * @param value - 待格式化的数值。
 * @param suffix - 可选后缀（如 `'手'`、`'%'`）。
 * @returns 格式化后的字符串，如 `"1000手"` 或 `"—"`。
 */
function fmtNumber(value: number | null | undefined, suffix = ''): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${value}${suffix}`
}

/**
 * 决策结果卡片（任务 9.4）。
 *
 * 任务完成后读取 task.result.decision（LiveDecisionResult），展示：
 * action（buy/sell/hold 带颜色）、建议手数 volume、价位 price、信号概率 signal、reason（Req 6.3）。
 * 始终展示「仅提醒，不自动下单」提示（Req 7.4）。
 */
const DecisionResultCard: React.FC<DecisionResultCardProps> = ({ task }) => {
  // task.result 在 Task 类型中为宽松的 Record，按本特性结果结构断言。
  const result = task?.result as LiveDecisionResult | undefined
  const decision: Decision | undefined = result?.decision

  // 始终展示的安全提示（Req 7.4），无论是否已有决策结果。
  const notice = (
    <Alert
      type="info"
      showIcon
      message="仅提醒，不自动下单"
      description="本决策仅为建议与提醒，系统不会向任何券商网关提交真实订单。"
    />
  )

  // 任务尚未完成或无结果：占位 + 安全提示。
  if (task?.status !== 'completed' || !decision) {
    return (
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Empty
          description={
            task?.status === 'failed'
              ? '决策任务失败，未产出决策结果'
              : '决策完成后将在此展示 action / 手数 / 价位 / 概率 / reason'
          }
        />
        {notice}
      </Space>
    )
  }

  const actionMeta = ACTION_META[decision.action] || {
    color: 'default',
    label: decision.action,
    icon: null,
  }

  // 信号概率以百分比展示（signal ∈ [0,1]）。
  const signalPercent =
    decision.signal === null || decision.signal === undefined
      ? null
      : decision.signal * 100

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Space align="center" size={16} wrap>
        <Statistic
          title="决策动作"
          valueRender={() => (
            <Tag color={actionMeta.color} icon={actionMeta.icon} style={{ fontSize: 16, padding: '4px 12px' }}>
              {actionMeta.label}
            </Tag>
          )}
        />
        <Statistic
          title="信号概率"
          value={signalPercent === null ? '—' : signalPercent}
          precision={signalPercent === null ? undefined : 2}
          suffix={signalPercent === null ? undefined : '%'}
        />
        <Statistic title="建议手数" value={fmtNumber(decision.volume)} suffix="股" />
        <Statistic
          title="建议价位"
          value={decision.price === null || decision.price === undefined ? '—' : decision.price}
          precision={decision.price === null || decision.price === undefined ? undefined : 2}
        />
      </Space>

      <Descriptions size="small" column={1} bordered>
        <Descriptions.Item label="reason">
          <Text>{decision.reason || '—'}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="目标标的">{decision.vt_symbol || '—'}</Descriptions.Item>
        <Descriptions.Item label="方案">{decision.scheme}</Descriptions.Item>
        <Descriptions.Item label="决策时刻">{decision.as_of}</Descriptions.Item>
        <Descriptions.Item label="signal_id">
          <Text copyable code style={{ fontSize: 12 }}>
            {decision.signal_id}
          </Text>
        </Descriptions.Item>
        {result?.idempotent_hit ? (
          <Descriptions.Item label="幂等">
            <Tag color="gold">幂等命中（返回首次决策，未重新触发提醒）</Tag>
          </Descriptions.Item>
        ) : null}
      </Descriptions>

      {/* 决策过程档案（任务 20.2）：懒加载，展开分组时才拉取该 signal_id 的六段 trace。 */}
      <DecisionTracePanel signalId={decision.signal_id} />

      {notice}
    </Space>
  )
}

export default DecisionResultCard

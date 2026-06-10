import React from 'react'
import { Alert, Empty, Progress, Space, Tag, Typography } from 'antd'

import type { Task } from '../../types/alpha'

const { Text } = Typography

interface ProgressCardProps {
  /** 当前决策任务（来自 useTask 订阅，复用既有 task/WS 机制）。 */
  task?: Task | null
  /** 是否已发起过任务（已有 task_id），用于在任务对象尚未返回时给出占位提示。 */
  hasTaskId: boolean
}

/** 任务状态 → 标签颜色。 */
const STATUS_TAG_COLOR: Record<string, string> = {
  pending: 'default',
  running: 'processing',
  completed: 'success',
  failed: 'error',
}

/** 任务状态 → 中文文案。 */
const STATUS_LABEL: Record<string, string> = {
  pending: '排队中',
  running: '执行中',
  completed: '已完成',
  failed: '失败',
}

/**
 * 任务进度卡片（任务 9.3）。
 *
 * 复用既有 task 机制（useTask 轮询 + WS task_update 写入 taskStore 的同一 Task 结构）：
 *  - 订阅页面提升的该 task_id 对应的 Task，进度更新时刷新进度条与状态消息（Req 6.2）。
 *  - 任务 FAILED 时展示错误消息（Req 6.7）。
 */
const ProgressCard: React.FC<ProgressCardProps> = ({ task, hasTaskId }) => {
  // 尚未触发任何决策：提示先在左侧表单触发。
  if (!hasTaskId) {
    return <Empty description="尚未触发决策，请在上方配置后点击「触发决策」" />
  }

  // 已有 task_id 但任务对象还在首帧加载：给一个加载态占位。
  if (!task) {
    return (
      <Space direction="vertical" size={8} style={{ width: '100%' }}>
        <Tag color="processing">加载中</Tag>
        <Progress percent={0} status="active" />
        <Text type="secondary">正在获取任务状态...</Text>
      </Space>
    )
  }

  const isFailed = task.status === 'failed'
  const progressStatus =
    isFailed ? 'exception' : task.status === 'completed' ? 'success' : 'active'

  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Space align="center" wrap>
        <Tag color={STATUS_TAG_COLOR[task.status] || 'default'}>
          {STATUS_LABEL[task.status] || task.status}
        </Tag>
        {task.entity_name ? <Text type="secondary">{task.entity_name}</Text> : null}
        {task.updated_at ? (
          <Text type="secondary">{new Date(task.updated_at).toLocaleString()}</Text>
        ) : null}
      </Space>

      <Progress percent={Math.round(task.progress ?? 0)} status={progressStatus} />

      {/* 任务失败：展示错误消息（Req 6.7）。 */}
      {isFailed ? (
        <Alert
          type="error"
          showIcon
          message="决策任务失败"
          description={task.message || '任务执行失败，请检查模型与行情数据后重试。'}
        />
      ) : task.message ? (
        // 非失败态：展示当前进度消息（Req 6.2）。
        <Text>{task.message}</Text>
      ) : null}
    </Space>
  )
}

export default ProgressCard

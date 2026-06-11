import React, { useMemo, useState } from 'react'
import { Alert, Button, Card, Modal, Progress, Space, Tag, Typography } from 'antd'

import type { Task } from '../types/alpha'

const { Text } = Typography

interface TaskStatusPanelProps {
  task?: Task | null
  title?: string
}

const TaskStatusPanel: React.FC<TaskStatusPanelProps> = ({ task, title }) => {
  const [resultOpen, setResultOpen] = useState(false)
  const resultText = useMemo(
    () => (task?.result ? JSON.stringify(task.result, null, 2) : ''),
    [task?.result],
  )

  if (!task) return null

  const tagColor: Record<string, string> = {
    pending: 'default',
    running: 'processing',
    completed: 'success',
    failed: 'error',
  }

  // 部分失败：任务整体完成，但结果中存在失败明细（如下载逐合约失败），需显著提示。
  const result = (task.result || {}) as {
    failed?: number
    failed_symbols?: string[]
    success?: number
    total?: number
  }
  const failedSymbols = Array.isArray(result.failed_symbols) ? result.failed_symbols : []
  const hasPartialFailure = task.status === 'completed' && (result.failed ?? failedSymbols.length) > 0

  return (
    <Card
      size="small"
      title={title || task.title || '任务状态'}
      styles={{ body: { display: 'flex', flexDirection: 'column', gap: 12 } }}
    >
      <Space align="center" wrap>
        <Tag color={tagColor[task.status] || 'default'}>{task.status}</Tag>
        {task.entity_name ? <Text type="secondary">{task.entity_name}</Text> : null}
        <Text type="secondary">{new Date(task.updated_at).toLocaleString()}</Text>
      </Space>
      <Progress
        percent={Math.round(task.progress)}
        status={task.status === 'failed' ? 'exception' : task.status === 'completed' ? 'success' : 'active'}
      />
      {task.message ? (
        task.status === 'failed'
          ? <Alert type="error" showIcon message={task.message} />
          : <Text>{task.message}</Text>
      ) : null}
      {hasPartialFailure ? (
        <Alert
          type="warning"
          showIcon
          message={`部分失败：成功 ${result.success ?? '-'}/${result.total ?? '-'}，失败 ${result.failed ?? failedSymbols.length} 个`}
          description={failedSymbols.length ? (
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {failedSymbols.map((item) => (
                <li key={item}><Text type="secondary" style={{ fontSize: 12 }}>{item}</Text></li>
              ))}
            </ul>
          ) : undefined}
        />
      ) : null}
      {task.status === 'completed' && task.result ? (
        <>
          <Alert
            type={hasPartialFailure ? 'info' : 'success'}
            showIcon
            message="任务完成"
            description="训练结果已收起，可点击按钮查看完整内容。"
            action={<Button size="small" type="primary" onClick={() => setResultOpen(true)}>查看结果</Button>}
          />
          <Modal
            open={resultOpen}
            title="任务结果"
            footer={null}
            width={900}
            onCancel={() => setResultOpen(false)}
          >
            <pre
              style={{
                maxHeight: '65vh',
                margin: 0,
                overflow: 'auto',
                padding: 12,
                background: 'rgba(0, 0, 0, 0.04)',
                borderRadius: 6,
                fontSize: 12,
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {resultText}
            </pre>
          </Modal>
        </>
      ) : null}
    </Card>
  )
}

export default TaskStatusPanel

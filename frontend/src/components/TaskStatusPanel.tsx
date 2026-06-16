import React, { useMemo, useState } from 'react'
import { Alert, Button, Card, Collapse, Modal, Progress, Space, Tag, Typography } from 'antd'

import type { Task } from '../types/alpha'

const { Text } = Typography

/** 任务状态面板的 Props。 */
interface TaskStatusPanelProps {
  /** 要展示的任务对象；null/undefined 时整个组件不渲染 */
  task?: Task | null
  /** 卡片标题；缺省使用 task.title 或 "任务状态" */
  title?: string
}

/**
 * 通用任务状态面板（Task Status Panel）。
 *
 * 将异步任务的生命周期以卡片形式展示：
 * - 状态 Tag + 实体名 + 更新时刻 + 耗时（duration_ms，R1 新增）
 * - 进度条
 * - 失败时：error Alert + 可折叠错误堆栈（error_traceback，R1 新增）
 * - 部分失败：warning Alert（failed_symbols 列表）
 * - 完成时：可弹窗查看完整 result JSON
 *
 * @param task - 要展示的任务对象
 * @param title - 卡片标题覆盖
 */
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
        {task.duration_ms != null ? (
          <Text type="secondary">耗时 {(task.duration_ms / 1000).toFixed(1)} 秒</Text>
        ) : null}
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
      {task.status === 'failed' && task.error_traceback ? (
        <Collapse
          size="small"
          items={[
            {
              key: 'traceback',
              label: '错误堆栈',
              children: (
                <pre
                  style={{
                    maxHeight: 240,
                    margin: 0,
                    overflow: 'auto',
                    padding: 8,
                    background: 'rgba(0, 0, 0, 0.04)',
                    borderRadius: 4,
                    fontSize: 11,
                    lineHeight: 1.5,
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                    fontFamily: 'monospace',
                  }}
                >
                  {task.error_traceback}
                </pre>
              ),
            },
          ]}
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

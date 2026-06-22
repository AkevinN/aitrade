import React from 'react'
import { Alert, Collapse, Descriptions, Drawer, Empty, Tag, Typography } from 'antd'

import type { BacktestStatistics, Task } from '../../types/alpha'
import type { ScreeningResult } from '../../types/screening'
import BacktestResults from '../Backtest/BacktestResults'
import ScreeningRunResult from './ScreeningRunResult'
import { runCategory, runCategoryLabel } from './runTypes'

const { Text, Paragraph } = Typography

/** 状态 → Tag 颜色。 */
const statusColor: Record<string, string> = {
  completed: 'green',
  failed: 'red',
  running: 'blue',
  pending: 'default',
}

/** 毫秒 → 可读耗时（与 TaskStatusPanel 一致：秒，保留 1 位）。 */
function formatDuration(ms?: number | null): string {
  if (ms == null) return '—'
  return `${(ms / 1000).toFixed(1)} 秒`
}

/** ISO 串 → 本地时间；空值返回占位。 */
function formatTime(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('zh-CN')
}

/**
 * 运行详情抽屉：回看一次历史回测/选股运行的完整结果。
 *
 * 顶部展示元信息（类型/名称/状态/时间/耗时）；失败运行展示错误堆栈、不渲染结果；
 * 成功运行按类别路由——回测复用既有 `BacktestResults`（统计指标），选股用只读榜单
 * `ScreeningRunResult`。纯只读，数据全部来自传入的任务记录（任务历史），无副作用。
 */
interface RunDetailDrawerProps {
  /** 选中的运行任务；null 时抽屉关闭 */
  task: Task | null
  /** 关闭抽屉回调 */
  onClose: () => void
}

const RunDetailDrawer: React.FC<RunDetailDrawerProps> = ({ task, onClose }) => {
  const category = task ? runCategory(task.type) : 'other'
  const result = (task?.result ?? null) as Record<string, unknown> | null

  /** 按类别与状态渲染结果主体。 */
  const renderBody = () => {
    if (!task) return null
    if (task.status === 'failed') {
      return (
        <>
          <Alert
            type="error"
            showIcon
            message="该次运行失败"
            description={task.message || '无错误消息'}
            style={{ marginBottom: 12 }}
          />
          {task.error_traceback ? (
            <Collapse
              size="small"
              items={[
                {
                  key: 'tb',
                  label: '错误堆栈',
                  children: (
                    <Paragraph>
                      <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 12 }}>
                        {task.error_traceback}
                      </pre>
                    </Paragraph>
                  ),
                },
              ]}
            />
          ) : null}
        </>
      )
    }
    if (task.status !== 'completed') {
      return <Empty description="该运行尚未完成，暂无结果" />
    }
    if (category === 'screening') {
      return <ScreeningRunResult result={result as unknown as ScreeningResult} />
    }
    if (category === 'backtest') {
      const statistics = result?.statistics as BacktestStatistics | undefined
      const capital =
        (result?.capital as number | undefined) ??
        ((task.params?.capital as number | undefined) || 1_000_000)
      return <BacktestResults statistics={statistics} capital={capital} />
    }
    return <Empty description="暂不支持该类型运行的详情展示" />
  }

  return (
    <Drawer
      open={task != null}
      onClose={onClose}
      width={760}
      destroyOnClose
      title={task ? `${runCategoryLabel(category)}运行详情` : '运行详情'}
    >
      {task ? (
        <>
          <Descriptions size="small" column={2} bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="类别">{runCategoryLabel(category)}</Descriptions.Item>
            <Descriptions.Item label="状态">
              <Tag color={statusColor[task.status] ?? 'default'}>{task.status}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="名称" span={2}>
              <Text>{task.title || task.entity_name || task.task_id}</Text>
            </Descriptions.Item>
            <Descriptions.Item label="创建时刻">{formatTime(task.created_at)}</Descriptions.Item>
            <Descriptions.Item label="完成时刻">
              {formatTime(task.finished_at ?? task.updated_at)}
            </Descriptions.Item>
            <Descriptions.Item label="耗时">{formatDuration(task.duration_ms)}</Descriptions.Item>
            <Descriptions.Item label="任务 ID">
              <Text code>{task.task_id}</Text>
            </Descriptions.Item>
          </Descriptions>
          {task.params && Object.keys(task.params).length > 0 ? (
            <Collapse
              size="small"
              style={{ marginBottom: 16 }}
              items={[
                {
                  key: 'params',
                  label: '运行参数',
                  children: (
                    <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 12 }}>
                      {JSON.stringify(task.params, null, 2)}
                    </pre>
                  ),
                },
              ]}
            />
          ) : null}
          {renderBody()}
        </>
      ) : null}
    </Drawer>
  )
}

export default RunDetailDrawer

import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  App,
  Button,
  Form,
  Progress,
  Select,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useQueryClient } from '@tanstack/react-query'
import type { AxiosError } from 'axios'

import { alphaService } from '../../api/alpha'
import { useTask } from '../../hooks/useTask'
import type {
  CsvImportMode,
  ParquetFilePreview,
  ParquetStageResult,
} from '../../types/alpha'

const { Text } = Typography

/** K 线导入可选周期（与页面主表保持一致的中文标签）。 */
const BAR_INTERVAL_OPTIONS = [
  { label: '日线', value: 'd' },
  { label: '1分钟', value: '1m' },
  { label: '5分钟', value: '5m' },
  { label: '15分钟', value: '15m' },
  { label: '30分钟', value: '30m' },
  { label: '60分钟', value: '60m' },
]

/** 导入模式可选项。 */
const IMPORT_MODE_OPTIONS = [
  { label: '合并（保留已有数据）', value: 'merge' },
  { label: '覆盖（替换已有数据）', value: 'replace' },
]

/**
 * 从异常中提取可展示的错误消息：优先后端 detail，其次 Error.message，最后兜底。
 *
 * @param error - 捕获到的异常，按 AxiosError 形态尝试读取后端 detail
 * @param fallback - 兜底文案
 * @returns 用于展示的错误消息字符串
 */
function getErrorMessage(error: unknown, fallback: string): string {
  const axiosError = error as AxiosError<{ detail?: string }>
  return axiosError.response?.data?.detail || axiosError.message || fallback
}

/**
 * ParquetUploadPanel 的属性。
 */
export interface ParquetUploadPanelProps {
  /** 用户为本面板收集到的待暂存 Parquet 文件；为空数组时面板不渲染内容。 */
  files: File[]
  /** 数据类型：`bar` 走 K线导入（含周期选择）；`tick` 走逐笔导入（无周期）。 */
  kind: 'bar' | 'tick'
  /** 全部文件清空后回调（如取消会话或导入成功），父组件据此重置已收集文件态。 */
  onClear: () => void
}

/**
 * Parquet 批量上传面板：暂存所选文件、展示逐文件预览汇总，并驱动正式导入与进度。
 *
 * 渲染流程：files 变化时自动调用 {@link alphaService.stageParquet} 暂存并拿到逐文件
 * {@link ParquetFilePreview} 列表与会话 ID；表格按文件渲染识别代码/行数/时间范围/缺列/
 * 可导入标记，不可导入行以红色文本与警告标签标注；底部汇总「N 文件 / M 可导入 / 不可导入数」。
 * bar 模式额外提供周期与导入模式选择，tick 无周期。「确认导入」在无可导入文件时禁用，点击后
 * 调用 {@link alphaService.importParquet} 并用 {@link useTask} 轮询进度；任务完成时刷新
 * 资源列表、提示成功并清空已暂存文件态。
 */
const ParquetUploadPanel: React.FC<ParquetUploadPanelProps> = ({ files, kind, onClear }) => {
  const { message } = App.useApp()
  const queryClient = useQueryClient()
  const [stage, setStage] = useState<ParquetStageResult | null>(null)
  const [staging, setStaging] = useState(false)
  const [importing, setImporting] = useState(false)
  const [taskId, setTaskId] = useState<string | null>(null)
  const [interval, setIntervalValue] = useState<string>('d')
  const [importMode, setImportMode] = useState<CsvImportMode>('merge')

  const task = useTask(taskId)

  // files 变化时重新暂存（每次选择新的一批文件即重置会话与预览）。
  useEffect(() => {
    let cancelled = false
    if (files.length === 0) {
      setStage(null)
      return
    }
    setStaging(true)
    setStage(null)
    alphaService
      .stageParquet(files, kind)
      .then((result) => {
        if (!cancelled) {
          setStage(result)
        }
      })
      .catch((error) => {
        if (!cancelled) {
          message.error(getErrorMessage(error, 'Parquet 暂存失败'))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setStaging(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [files, kind, message])

  // 任务完成后刷新资源列表、提示并清空已暂存文件态（仅触发一次）。
  useEffect(() => {
    if (task.data?.status === 'completed') {
      const result = task.data.result as
        | { total?: number; success?: number; failed?: number }
        | null
        | undefined
      const success = result?.success ?? 0
      const failed = result?.failed ?? 0
      message.success(`Parquet 导入完成：成功 ${success} 个${failed ? `，失败 ${failed} 个` : ''}`)
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
      setStage(null)
      setTaskId(null)
      onClear()
    } else if (task.data?.status === 'failed') {
      message.error(task.data.message || 'Parquet 导入失败')
      setTaskId(null)
    }
    // onClear 在父级为稳定回调；仅依赖任务状态变化即可。
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task.data?.status])

  const importableCount = useMemo(
    () => stage?.files.filter((f) => f.importable).length ?? 0,
    [stage],
  )
  const notImportableCount = (stage?.files.length ?? 0) - importableCount

  /**
   * 提交正式导入：以当前会话、周期与模式调用 importParquet，记录返回的 task_id 驱动进度轮询。
   */
  const handleImport = async () => {
    if (!stage) return
    setImporting(true)
    try {
      const result = await alphaService.importParquet({
        session_id: stage.session_id,
        data_kind: kind,
        interval,
        import_mode: importMode,
      })
      setTaskId(result.task_id)
      message.success('Parquet 导入任务已启动')
    } catch (error) {
      message.error(getErrorMessage(error, 'Parquet 导入失败'))
    } finally {
      setImporting(false)
    }
  }

  const columns: ColumnsType<ParquetFilePreview> = [
    {
      title: '文件',
      dataIndex: 'file_name',
      key: 'file_name',
      render: (value: string, record) => (
        <Text type={record.importable ? undefined : 'danger'}>{value}</Text>
      ),
    },
    {
      title: '识别代码',
      key: 'vt_symbol',
      render: (_, record) =>
        record.vt_symbol ? (
          record.vt_symbol
        ) : (
          <Text type={record.importable ? 'secondary' : 'danger'}>{record.reason || '未识别'}</Text>
        ),
    },
    {
      title: '行数',
      dataIndex: 'row_count',
      key: 'row_count',
    },
    {
      title: '时间范围',
      key: 'date_range',
      render: (_, record) =>
        record.date_range[0] ? `${record.date_range[0]} ~ ${record.date_range[1]}` : '—',
    },
    {
      title: '缺列',
      key: 'missing_required',
      render: (_, record) =>
        record.missing_required.length ? (
          <Text type="danger">{record.missing_required.join(', ')}</Text>
        ) : (
          '—'
        ),
    },
    {
      title: '可导入',
      key: 'importable',
      render: (_, record) =>
        record.importable ? (
          <Tag color="success">可导入</Tag>
        ) : (
          <Tag color="error">不可导入</Tag>
        ),
    },
  ]

  if (files.length === 0) {
    return null
  }

  const running = task.data?.status === 'running' || task.data?.status === 'pending'

  return (
    <Space direction="vertical" style={{ width: '100%' }} size={12}>
      {staging ? <Alert type="info" showIcon message="正在解析 Parquet 文件..." /> : null}
      {stage ? (
        <>
          <Table<ParquetFilePreview>
            rowKey="file_name"
            size="small"
            pagination={false}
            columns={columns}
            dataSource={stage.files}
          />
          <Text>
            {`共 ${stage.files.length} 文件 / 可导入 ${importableCount} / 不可导入 ${notImportableCount}`}
          </Text>
          {kind === 'bar' ? (
            <Form layout="vertical">
              <Form.Item label="导入周期" style={{ marginBottom: 12 }}>
                <Select
                  value={interval}
                  onChange={setIntervalValue}
                  options={BAR_INTERVAL_OPTIONS}
                />
              </Form.Item>
              <Form.Item label="导入模式" style={{ marginBottom: 0 }}>
                <Select
                  value={importMode}
                  onChange={(value) => setImportMode(value as CsvImportMode)}
                  options={IMPORT_MODE_OPTIONS}
                />
              </Form.Item>
            </Form>
          ) : (
            <Form layout="vertical">
              <Form.Item label="导入模式" style={{ marginBottom: 0 }}>
                <Select
                  value={importMode}
                  onChange={(value) => setImportMode(value as CsvImportMode)}
                  options={IMPORT_MODE_OPTIONS}
                />
              </Form.Item>
            </Form>
          )}
          {taskId && task.data ? (
            <Alert
              type={task.data.status === 'failed' ? 'error' : 'info'}
              showIcon
              message={task.data.message || '导入进行中...'}
              description={<Progress percent={task.data.progress} size="small" />}
            />
          ) : null}
          <Space>
            <Button
              type="primary"
              onClick={() => void handleImport()}
              loading={importing || running}
              disabled={importableCount === 0}
            >
              确认导入
            </Button>
            <Button onClick={onClear} disabled={importing || running}>
              取消
            </Button>
          </Space>
        </>
      ) : null}
    </Space>
  )
}

export default ParquetUploadPanel

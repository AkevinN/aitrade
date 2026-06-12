import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  App,
  Button,
  Card,
  Checkbox,
  Col,
  Descriptions,
  Empty,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
  Upload,
} from 'antd'
import {
  DeleteOutlined,
  EyeOutlined,
  InboxOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import type { UploadFile, UploadProps } from 'antd'
import type { AxiosError } from 'axios'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import dayjs from 'dayjs'

import { alphaService } from '../../api/alpha'
import { statusService } from '../../api/status'
import AggregationWorkspace from './AggregationWorkspace'
import DateRangeSelector from '../../components/DateRangeSelector'
import TaskStatusPanel from '../../components/TaskStatusPanel'
import { useTask } from '../../hooks/useTask'
import type {
  CsvImportMode,
  CsvInterval,
  CsvPreviewResult,
  DataResourceSummary,
  DataResourceMergePreview,
} from '../../types/alpha'

const { Text, Title } = Typography
const { TextArea } = Input
const { Dragger } = Upload

const BAR_INTERVAL_OPTIONS = [
  { label: '日线', value: 'd' },
  { label: '1分钟', value: '1m' },
  { label: '5分钟', value: '5m' },
  { label: '15分钟', value: '15m' },
  { label: '30分钟', value: '30m' },
  { label: '60分钟', value: '60m' },
]

const INTERVAL_LABELS: Record<string, string> = {
  d: '日线',
  tick: 'Tick',
  '1m': '1分钟',
  '5m': '5分钟',
  '10m': '10分钟',
  '15m': '15分钟',
  '30m': '30分钟',
  '60m': '60分钟',
}

const INTERVAL_PRIORITY: Record<string, number> = {
  tick: 0,
  '1m': 1,
  '5m': 2,
  '10m': 3,
  '15m': 4,
  '30m': 5,
  '60m': 6,
  d: 99,
}

const RESOURCE_COLUMN_LABELS: Record<string, string> = {
  datetime: '时间',
  trade_datetime: '交易时间',
  timestamp: '时间戳',
  symbol: '证券代码',
  vt_symbol: '合约代码',
  open: '开盘价',
  high: '最高价',
  low: '最低价',
  close: '收盘价',
  volume: '成交量',
  turnover: '成交额',
  open_interest: '持仓量',
  last_price: '最新价',
  bid_price_1: '买一价',
  ask_price_1: '卖一价',
  bid_volume_1: '买一量',
  ask_volume_1: '卖一量',
}

/**
 * 将原始列名映射为中文可读标签；无映射时返回原始列名。
 *
 * @param column - 数据列名，如 "close" / "vt_symbol"
 * @returns 中文标签，如 "收盘价" / "合约代码"
 */
const formatResourceColumnLabel = (column: string) => RESOURCE_COLUMN_LABELS[column] || column

/**
 * 将多行或逗号分隔的标的代码文本解析为字符串数组。
 *
 * 支持换行符与英文逗号两种分隔方式，自动去除首尾空白和空项。
 *
 * @param raw - 原始文本，如 "000001.SZSE\n600000.SSE" 或 "000001.SZSE,600000.SSE"
 * @returns 合约代码数组，如 ["000001.SZSE", "600000.SSE"]
 */
const parseSymbols = (raw: string) => (
  raw
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean)
)

/**
 * 去除合约代码末尾多余的小数点（如 "000001." → "000001"）。
 *
 * @param value - 原始合约代码字符串
 * @returns 去除末尾点的字符串
 */
const formatResourceSymbol = (value: string) => value.replace(/\.$/, '')

/**
 * 将行情周期代码映射为中文标签；无映射时返回原始值，null/undefined 时返回 "-"。
 *
 * @param value - 周期代码，如 "d" / "1m" / "tick"
 * @returns 中文标签，如 "日线" / "1分钟"
 */
const formatIntervalLabel = (value?: string | null) => {
  if (!value) {
    return '-'
  }
  return INTERVAL_LABELS[value] || value
}

/**
 * 将 ISO 时间字符串格式化为 "YYYY-MM-DD HH:mm:ss"；无效时返回原值或 fallback。
 *
 * @param value - ISO 时间字符串；null/undefined 时返回 fallback
 * @param fallback - 缺省值，默认 "-"
 * @returns 格式化后的时间字符串
 */
const formatDateTime = (value?: string | null, fallback = '-') => {
  if (!value) {
    return fallback
  }
  const parsed = dayjs(value)
  if (!parsed.isValid()) {
    return value
  }
  return parsed.format(value.includes(':') && value.split(':').length >= 3 ? 'YYYY-MM-DD HH:mm:ss' : 'YYYY-MM-DD HH:mm')
}

const formatResourceDate = (value: string | null | undefined, interval?: string | null) => {
  if (!value) {
    return '-'
  }
  const parsed = dayjs(value)
  if (!parsed.isValid()) {
    return value
  }
  if (interval === 'd') {
    return parsed.format('YYYY-MM-DD')
  }
  if (interval === 'tick') {
    return parsed.format('YYYY-MM-DD HH:mm:ss')
  }
  return parsed.format('YYYY-MM-DD HH:mm')
}

const formatDateRange = (start?: string | null, end?: string | null, interval?: string | null) => (
  `${formatResourceDate(start, interval)} ~ ${formatResourceDate(end, interval)}`
)

const getTimestamp = (value?: string | null) => {
  if (!value) {
    return 0
  }
  const parsed = dayjs(value)
  return parsed.isValid() ? parsed.valueOf() : 0
}

const formatResourceSource = (resource: DataResourceSummary) => {
  if (resource.kind === 'raw_bar_batch' || resource.kind === 'raw_tick_batch') {
    const name = resource.file_name ? ` · ${resource.file_name}` : ''
    return `上传批次${name}`
  }
  if (resource.kind === 'raw_tick') {
    return '历史 Tick 原始数据'
  }
  if (resource.kind === 'derived_bar') {
    return resource.source_kind === 'tick'
      ? `由 Tick 聚合为 ${resource.target_interval}`
      : `由 ${resource.source_interval} 聚合为 ${resource.target_interval}`
  }
  return resource.interval === 'd' ? '原始日线数据' : `原始 ${resource.interval} 数据`
}

const formatResourceKindLabel = (kind: DataResourceSummary['kind']) => {
  if (kind === 'raw_tick' || kind === 'raw_tick_batch') {
    return 'Tick'
  }
  if (kind === 'derived_bar') {
    return '派生K线'
  }
  if (kind === 'raw_bar_batch') {
    return 'K线批次'
  }
  return '原始K线'
}

const formatResourceKindColor = (kind: DataResourceSummary['kind']) => {
  if (kind === 'raw_tick' || kind === 'raw_tick_batch') {
    return 'gold'
  }
  if (kind === 'derived_bar') {
    return 'purple'
  }
  if (kind === 'raw_bar_batch') {
    return 'cyan'
  }
  return 'blue'
}

const sortResources = (items: DataResourceSummary[]) => (
  [...items].sort((left, right) => {
    const createdDiff = getTimestamp(right.created_at) - getTimestamp(left.created_at)
    if (createdDiff !== 0) {
      return createdDiff
    }

    const intervalDiff = (INTERVAL_PRIORITY[left.target_interval || left.interval] ?? 50)
      - (INTERVAL_PRIORITY[right.target_interval || right.interval] ?? 50)
    if (intervalDiff !== 0) {
      return intervalDiff
    }

    const endDiff = getTimestamp(right.end) - getTimestamp(left.end)
    if (endDiff !== 0) {
      return endDiff
    }

    if (right.row_count !== left.row_count) {
      return right.row_count - left.row_count
    }

    return formatResourceSymbol(left.vt_symbol).localeCompare(formatResourceSymbol(right.vt_symbol), 'zh-CN')
  })
)

const getErrorMessage = (error: unknown, fallback: string) => {
  const axiosError = error as AxiosError<{ detail?: string }>
  return axiosError.response?.data?.detail || axiosError.message || fallback
}

const ResourceTable: React.FC<{
  data: DataResourceSummary[]
  emptyText: string
  onPreview: (resource: DataResourceSummary) => void
  onDelete: (resource: DataResourceSummary) => void
  onTrain: (resource: DataResourceSummary) => void
  selectedRowKeys?: React.Key[]
  onSelectionChange?: (keys: React.Key[]) => void
}> = ({ data, emptyText, onPreview, onDelete, onTrain, selectedRowKeys, onSelectionChange }) => (
  <Table<DataResourceSummary>
    size="small"
    rowKey="key"
    dataSource={data}
    pagination={{ pageSize: 6, showSizeChanger: false }}
    locale={{ emptyText }}
    rowSelection={onSelectionChange ? {
      selectedRowKeys,
      onChange: (keys) => onSelectionChange(keys),
      // 已合并批次保留为历史记录，但不能再次作为待合并输入。
      getCheckboxProps: (record) => ({ disabled: record.status === 'merged' }),
    } : undefined}
    columns={[
      {
        title: '资源信息',
        key: 'resource',
        width: 280,
        render: (_value, record) => (
          <Space direction="vertical" size={2}>
            <Button type="link" style={{ padding: 0 }} onClick={() => onPreview(record)}>
              {formatResourceSymbol(record.vt_symbol)}
            </Button>
            <Space size={4} wrap>
              <Tag color={formatResourceKindColor(record.kind)}>
                {formatResourceKindLabel(record.kind)}
              </Tag>
              <Tag>{formatIntervalLabel(record.interval)}</Tag>
              {record.status ? <Tag color={record.status === 'merged' ? 'green' : 'orange'}>{record.status === 'merged' ? '已合并' : '待合并'}</Tag> : null}
            </Space>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {formatResourceSource(record)}
            </Text>
          </Space>
        ),
      },
      {
        title: '区间',
        key: 'range',
        width: 220,
        render: (_value, record) => (
          <Space direction="vertical" size={0}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {formatDateRange(record.start, record.end, record.interval)}
            </Text>
            {record.created_at ? (
              <Text type="secondary" style={{ fontSize: 12 }}>
                生成于 {formatDateTime(record.created_at)}
              </Text>
            ) : null}
          </Space>
        ),
      },
      {
        title: '数据量',
        key: 'size',
        width: 150,
        render: (_value, record) => (
          <Space direction="vertical" size={0}>
            <Text style={{ fontVariantNumeric: 'tabular-nums' }}>
              {new Intl.NumberFormat('zh-CN').format(record.row_count)} 行
            </Text>
            <Text type="secondary" style={{ fontSize: 12, fontVariantNumeric: 'tabular-nums' }}>
              {record.file_size_kb.toFixed(1)} KB
            </Text>
          </Space>
        ),
      },
      {
        title: '操作',
        key: 'actions',
        width: 180,
        render: (_value, record) => (
          <Space size="small">
            <Button size="small" icon={<EyeOutlined />} onClick={() => onPreview(record)}>
              预览
            </Button>
            {record.kind !== 'raw_tick' && record.kind !== 'raw_bar_batch' && record.kind !== 'raw_tick_batch' ? (
              <Button size="small" onClick={() => onTrain(record)}>
                训练CNN
              </Button>
            ) : null}
            <Popconfirm title="确认删除这个数据资源？" onConfirm={() => onDelete(record)}>
              <Button size="small" danger icon={<DeleteOutlined />} />
            </Popconfirm>
          </Space>
        ),
      },
    ]}
    scroll={{ x: 860 }}
  />
)

const DataPrepare: React.FC = () => {
  const { message, modal } = App.useApp()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [downloadForm] = Form.useForm()
  const [barImportForm] = Form.useForm()
  const watchedProvider = Form.useWatch('provider', downloadForm)
  const watchedSourceInterval = Form.useWatch('source_interval', downloadForm)
  // AKShare 的 1 分钟线仅提供近 5 个交易日，提前提示避免用户选长区间后被截断/报错。
  const showAkshare1mWarning = watchedProvider === 'akshare' && watchedSourceInterval === '1m'

  const [taskId, setTaskId] = useState<string | null>(null)
  const [selectedResource, setSelectedResource] = useState<DataResourceSummary | null>(null)
  const [barCsvFile, setBarCsvFile] = useState<UploadFile[]>([])
  const [tickCsvFile, setTickCsvFile] = useState<UploadFile[]>([])
  const [barPreview, setBarPreview] = useState<CsvPreviewResult | null>(null)
  const [tickPreview, setTickPreview] = useState<CsvPreviewResult | null>(null)
  const [previewLoading, setPreviewLoading] = useState<'bar' | 'tick' | null>(null)
  const [importingKind, setImportingKind] = useState<'bar' | 'tick' | null>(null)
  const [intervalDraft, setIntervalDraft] = useState<CsvInterval>('d')
  const [relocatingInterval, setRelocatingInterval] = useState(false)
  const [showMergedBatches, setShowMergedBatches] = useState(false)
  const [selectedBarBatchKeys, setSelectedBarBatchKeys] = useState<React.Key[]>([])
  const [selectedTickBatchKeys, setSelectedTickBatchKeys] = useState<React.Key[]>([])
  const [mergePreview, setMergePreview] = useState<DataResourceMergePreview | null>(null)
  const [mergingKind, setMergingKind] = useState<'raw_bar' | 'raw_tick' | null>(null)

  const task = useTask(taskId)

  // 任务面板在页面顶部，而聚合按钮在页面底部；提交后将面板滚动进视野，
  // 避免“提示已启动却看不到任何反应”的体验问题。
  const taskPanelRef = useRef<HTMLDivElement>(null)
  const handleTaskStarted = (id: string) => {
    setTaskId(id)
    requestAnimationFrame(() => {
      taskPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }

  const {
    data: resources,
    isLoading: resourcesLoading,
    error: resourcesError,
  } = useQuery({
    queryKey: ['alpha-data-resources'],
    queryFn: () => alphaService.getDataResources(),
  })

  const { data: resourceDetail, isLoading: detailLoading } = useQuery({
    queryKey: ['alpha-data-resource-detail', selectedResource?.kind, selectedResource?.key],
    queryFn: () => alphaService.getDataResourceDetail(selectedResource!.kind, selectedResource!.key, { limit: 60 }),
    enabled: !!selectedResource,
  })

  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: () => statusService.getStatus(),
  })

  // 数据源选项：排除 mock（仅兜底，不用于真实下载），按优先级排序，附带状态标注。
  const providerOptions = useMemo(() => {
    const providers = (systemStatus?.providers || [])
      .filter((p) => p.name !== 'mock')
      .sort((a, b) => a.priority - b.priority)
    const statusLabel = (status: string) => (status === 'available' ? '可用' : '不可用')
    return [
      { label: '自动（按优先级）', value: 'auto' },
      ...providers.map((p) => ({
        label: `${p.name}（${statusLabel(p.status)}）`,
        value: p.name,
        disabled: p.status !== 'available',
      })),
    ]
  }, [systemStatus])

  useEffect(() => {
    downloadForm.setFieldsValue({
      range: [dayjs().subtract(1, 'year'), dayjs()],
      source_interval: 'd',
      provider: 'auto',
      asset_class: 'stock',
    })
    barImportForm.setFieldsValue({
      interval: 'd',
      import_mode: 'merge',
    })
  }, [barImportForm, downloadForm])

  useEffect(() => {
    if (resourceDetail?.kind === 'raw_bar' && resourceDetail.interval) {
      setIntervalDraft(resourceDetail.interval as CsvInterval)
    }
  }, [resourceDetail])

  useEffect(() => {
    if (task.data?.status === 'completed') {
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    }
  }, [queryClient, task.data?.status])



  const totalRows = useMemo(() => {
    const rawBars = resources?.raw_bars.reduce((sum, item) => sum + item.row_count, 0) || 0
    const rawTicks = resources?.raw_ticks.reduce((sum, item) => sum + item.row_count, 0) || 0
    const derived = resources?.derived_bars.reduce((sum, item) => sum + item.row_count, 0) || 0
    return rawBars + rawTicks + derived
  }, [resources])

  const sortedRawBars = useMemo(() => sortResources(resources?.raw_bars || []), [resources?.raw_bars])
  const sortedRawTicks = useMemo(() => sortResources(resources?.raw_ticks || []), [resources?.raw_ticks])
  const sortedDerivedBars = useMemo(() => sortResources(resources?.derived_bars || []), [resources?.derived_bars])
  const rawBarBatches = resources?.raw_bar_batches || []
  const rawTickBatches = resources?.raw_tick_batches || []
  const visibleRawBarBatches = useMemo(
    () => sortResources(rawBarBatches.filter((item) => showMergedBatches || item.status !== 'merged')),
    [rawBarBatches, showMergedBatches],
  )
  const visibleRawTickBatches = useMemo(
    () => sortResources(rawTickBatches.filter((item) => showMergedBatches || item.status !== 'merged')),
    [rawTickBatches, showMergedBatches],
  )
  const pendingBatchCount = useMemo(
    () => [...rawBarBatches, ...rawTickBatches].filter((item) => item.status !== 'merged').length,
    [rawBarBatches, rawTickBatches],
  )

  const handleDownload = async () => {
    try {
      const values = await downloadForm.validateFields()
      const symbols = parseSymbols(values.symbols)
      const result = await alphaService.downloadData({
        vt_symbols: symbols,
        data_kind: 'bar',
        source_interval: values.source_interval,
        start: values.range[0].format('YYYY-MM-DD'),
        end: values.range[1].format('YYYY-MM-DD'),
        provider: values.provider && values.provider !== 'auto' ? values.provider : undefined,
        asset_class: (values.asset_class as 'stock' | 'etf' | 'cbond') || 'stock',
      })
      setTaskId(result.task_id)
      message.success('原始K线下载任务已启动')
    } catch (error) {
      if (error instanceof Error) {
        message.error(error.message)
      }
    }
  }

  const previewCsv = async (kind: 'bar' | 'tick', file: File) => {
    setPreviewLoading(kind)
    try {
      const result = kind === 'bar'
        ? await alphaService.previewCsvImport(file)
        : await alphaService.previewTickCsvImport(file)
      if (kind === 'bar') {
        setBarPreview(result)
      } else {
        setTickPreview(result)
      }
      message.success(`${kind === 'bar' ? 'K线' : 'Tick'} CSV 预览完成`)
    } catch (error) {
      message.error(getErrorMessage(error, `${kind === 'bar' ? 'K线' : 'Tick'} CSV 预览失败`))
    } finally {
      setPreviewLoading(null)
    }
  }

  const buildUploadProps = (kind: 'bar' | 'tick'): UploadProps => ({
    beforeUpload: (file) => {
      if (!file.name.toLowerCase().endsWith('.csv')) {
        message.error('仅支持 CSV 文件')
        return Upload.LIST_IGNORE
      }
      void previewCsv(kind, file)
      return false
    },
    fileList: kind === 'bar' ? barCsvFile : tickCsvFile,
    onChange: ({ fileList }) => {
      if (kind === 'bar') {
        setBarCsvFile(fileList)
      } else {
        setTickCsvFile(fileList)
      }
    },
    onRemove: () => {
      if (kind === 'bar') {
        setBarCsvFile([])
        setBarPreview(null)
      } else {
        setTickCsvFile([])
        setTickPreview(null)
      }
    },
    accept: '.csv',
    maxCount: 1,
  })

  const handleImport = async (kind: 'bar' | 'tick') => {
    const fileList = kind === 'bar' ? barCsvFile : tickCsvFile
    const preview = kind === 'bar' ? barPreview : tickPreview
    if (!fileList.length || !preview) {
      message.warning('请先上传并预览 CSV 文件')
      return
    }
    if (preview.missing_required.length > 0) {
      message.warning('CSV 缺少必填字段，无法导入')
      return
    }

    setImportingKind(kind)
    try {
      const file = fileList[0].originFileObj as File
      const importValues = kind === 'bar'
        ? await barImportForm.validateFields()
        : { import_mode: 'merge' }
      const importMode = (importValues.import_mode || 'merge') as CsvImportMode
      const result = kind === 'bar'
        ? await alphaService.importCsvData(
          file,
          importValues.interval,
          importMode,
          'batch',
        )
        : await alphaService.importTickCsvData(
          file,
          importMode,
          'batch',
        )
      if (result.success) {
        message.success(result.message)
        queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
        if (kind === 'bar') {
          setBarCsvFile([])
          setBarPreview(null)
        } else {
          setTickCsvFile([])
          setTickPreview(null)
        }
      } else {
        message.error(result.message)
      }
    } catch (error) {
      message.error(getErrorMessage(error, `${kind === 'bar' ? 'K线' : 'Tick'} CSV 导入失败`))
    } finally {
      setImportingKind(null)
    }
  }

  const handleDeleteResource = async (resource: DataResourceSummary) => {
    try {
      await alphaService.deleteDataResource(resource.kind, resource.key)
      message.success('数据资源已删除')
      if (selectedResource?.key === resource.key) {
        setSelectedResource(null)
      }
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
    } catch {
      message.error('删除失败')
    }
  }

  const handleRelocateInterval = async () => {
    if (!selectedResource || selectedResource.kind !== 'raw_bar' || !resourceDetail) {
      return
    }
    if (intervalDraft === resourceDetail.interval) {
      message.info('周期未变化')
      return
    }

    // 二次确认：relocate 仅更正周期"标签"并移动文件，不会重新采样/转换 K 线内容。
    const confirmed = await new Promise<boolean>((resolve) => {
      Modal.confirm({
        title: '确认更正周期标签？',
        content: `将把 ${selectedResource.vt_symbol} 的周期标签从 ${resourceDetail.interval} 改为 ${intervalDraft}。` +
          '此操作仅更正标签并移动文件，不会对 K 线数据做重采样或聚合转换。' +
          '请仅在原数据本身就是该周期、只是标签写错时使用。',
        okText: '确认更正',
        cancelText: '取消',
        onOk: () => resolve(true),
        onCancel: () => resolve(false),
      })
    })
    if (!confirmed) {
      return
    }

    setRelocatingInterval(true)
    try {
      const result = await alphaService.relocateRawBarInterval(selectedResource.key, intervalDraft)
      message.success(result.message)
      const updatedResource: DataResourceSummary = {
        ...selectedResource,
        key: result.key,
        interval: result.interval,
        source_interval: result.interval,
        target_interval: result.interval,
      }
      setSelectedResource(updatedResource)
      queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
      queryClient.invalidateQueries({
        queryKey: ['alpha-data-resource-detail', 'raw_bar', result.key],
      })
    } catch (error) {
      message.error(getErrorMessage(error, '周期更正失败'))
    } finally {
      setRelocatingInterval(false)
    }
  }

  const handleMergeBatches = async (kind: 'raw_bar' | 'raw_tick') => {
    const selectedKeys = kind === 'raw_bar' ? selectedBarBatchKeys : selectedTickBatchKeys
    const keys = selectedKeys.map(String)
    if (keys.length < 1) {
      message.warning('请至少选择一个待合并批次')
      return
    }

    setMergingKind(kind)
    setMergePreview(null)
    try {
      // 所有合并规则由后端 preview 统一判断，前端只负责展示原因和二次确认。
      const preview = await alphaService.previewDataResourceMerge({ kind, keys })
      setMergePreview(preview)
      if (!preview.can_merge) {
        message.warning(preview.reason || '所选批次不可合并')
        return
      }

      modal.confirm({
        title: '确认合并到正式 K 线？',
        content: (
          <Space direction="vertical" size={4}>
            <Text>{preview.vt_symbol} / {formatIntervalLabel(preview.interval)}</Text>
            <Text type="secondary">
              {preview.has_official ? '将与现有正式数据合并' : '将新建正式资源'}，预计 {preview.estimated_rows || 0} 行。
            </Text>
            <Text type="secondary">
              已校验：重叠且重叠区数据一致{preview.conflict_count ? `（冲突 ${preview.conflict_count} 个）` : ''}、分钟线连续。
            </Text>
          </Space>
        ),
        okText: '合并',
        cancelText: '取消',
        onOk: async () => {
          // merge 接口会重新执行同一套校验，避免 preview 后批次状态变化造成误合并。
          const result = await alphaService.mergeDataResourceBatches({ kind, keys })
          message.success(result.message || '批次已合并')
          if (kind === 'raw_bar') {
            setSelectedBarBatchKeys([])
          } else {
            setSelectedTickBatchKeys([])
          }
          setMergePreview(null)
          queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })
        },
      })
    } catch (error) {
      message.error(getErrorMessage(error, '批次合并失败'))
    } finally {
      setMergingKind(null)
    }
  }

  const handleTrainWithResource = (resource: DataResourceSummary) => {
    navigate('/cnn-train', {
      state: {
        preset: {
          target_symbol: resource.vt_symbol,
          input_data_kind: resource.kind === 'raw_tick' ? 'tick' : 'bar',
          input_interval: resource.target_interval || resource.interval,
          symbols: [resource.vt_symbol],
        },
      },
    })
  }


  return (
    <div className="page-enter">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>数据准备工作台</Title>
          <Text type="secondary">
            准备原始数据和资源，可在线下载K线、导入CSV文件，或将已有数据聚合为派生周期。
          </Text>
        </div>

        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="原始K线" value={resources?.raw_bars.length || 0} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="历史Tick" value={resources?.raw_ticks.length || 0} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="派生周期" value={resources?.derived_bars.length || 0} /></Card>
          </Col>
          <Col xs={24} sm={12} xl={6}>
            <Card><Statistic title="批次 / 总行数" value={pendingBatchCount} suffix={`/ ${totalRows.toLocaleString('zh-CN')} 行`} /></Card>
          </Col>
        </Row>

        <div ref={taskPanelRef}>
          <TaskStatusPanel task={task.data || null} title="当前任务" />
        </div>

        {resourcesError ? (
          <Alert
            type="error"
            showIcon
            message="数据资源加载失败"
            description={getErrorMessage(resourcesError, '请确认后端服务已启动，并检查接口返回错误')}
          />
        ) : null}

        <Row gutter={[16, 16]} align="top">
          <Col xs={24} xl={8} xxl={7}>
            <Card title="原始数据获取">
              <Tabs
                items={[
                  {
                    key: 'download',
                    label: 'K线下载',
                    children: (
                      <Form form={downloadForm} layout="vertical">
                        <Alert
                          type="info"
                          showIcon
                          style={{ marginBottom: 12 }}
                          message="下载数据将存为待合并批次"
                          description="下载结果不会直接写入正式 K 线，需在「K线批次」中合并（自动校验重叠/一致/连续）后并入正式资源。"
                        />
                        <Form.Item
                          label="合约列表"
                          name="symbols"
                          rules={[{ required: true, message: '请输入至少一个证券代码' }]}
                        >
                          <TextArea rows={4} placeholder="000001.SZSE&#10;399300.SZSE" />
                        </Form.Item>
                        <Form.Item
                          label="品种"
                          name="asset_class"
                        >
                          <Select
                            options={[
                              { label: 'A股股票', value: 'stock' },
                              { label: 'ETF', value: 'etf' },
                              { label: '可转债', value: 'cbond' },
                            ]}
                          />
                        </Form.Item>
                        <Form.Item
                          label="数据源"
                          name="provider"
                          tooltip="选择行情数据源用于下载或补充；自动模式按优先级 tushare → akshare 选择。AKShare 仅支持 A 股股票，ETF 请用 Tushare；1 分钟数据仅近 5 个交易日。"
                        >
                          <Select options={providerOptions} />
                        </Form.Item>
                        <Form.Item
                          label="下载周期"
                          name="source_interval"
                          rules={[{ required: true, message: '请选择原始K线周期' }]}
                        >
                          <Select options={BAR_INTERVAL_OPTIONS} />
                        </Form.Item>
                        <Form.Item
                          label="时间范围"
                          name="range"
                          rules={[{ required: true, message: '请选择时间范围' }]}
                        >
                          <DateRangeSelector />
                        </Form.Item>
                        {showAkshare1mWarning ? (
                          <Alert
                            type="warning"
                            showIcon
                            style={{ marginBottom: 12 }}
                            message="AKShare 1 分钟线仅支持近 5 个交易日"
                            description="所选范围过长将被截断或被服务端拒绝。请缩短到最近 5 个交易日内，或改用 5m 及以上周期 / Tushare 数据源。"
                          />
                        ) : null}
                        <Button
                          type="primary"
                          icon={<PlayCircleOutlined />}
                          loading={task.data?.status === 'running'}
                          onClick={() => void handleDownload()}
                          block
                        >
                          启动下载
                        </Button>
                      </Form>
                    ),
                  },
                  {
                    key: 'bar-import',
                    label: 'K线CSV',
                    children: (
                      <Space direction="vertical" style={{ width: '100%' }} size={12}>
                        <Dragger {...buildUploadProps('bar')}>
                          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                          <p className="ant-upload-text">上传 K线 CSV 文件</p>
                          <p className="ant-upload-hint">支持 d / 1m / 5m / 15m / 30m / 60m</p>
                        </Dragger>
                        <Form form={barImportForm} layout="vertical">
                          <Form.Item
                            label="导入周期"
                            name="interval"
                            rules={[{ required: true, message: '请选择导入周期' }]}
                          >
                            <Select options={BAR_INTERVAL_OPTIONS} />
                          </Form.Item>
                        </Form>
                        <Alert
                          type="info"
                          showIcon
                          message="CSV 将保存为待合并批次，不会直接覆盖正式行情。"
                        />
                        {previewLoading === 'bar' ? <Alert type="info" showIcon message="正在解析 K线 CSV..." /> : null}
                        {barPreview ? (
                          <Descriptions size="small" bordered column={1}>
                            <Descriptions.Item label="识别证券">{barPreview.symbols.join(', ') || '无'}</Descriptions.Item>
                            <Descriptions.Item label="日期范围">{barPreview.date_range[0]} ~ {barPreview.date_range[1]}</Descriptions.Item>
                            <Descriptions.Item label="缺失字段">{barPreview.missing_required.join(', ') || '无'}</Descriptions.Item>
                          </Descriptions>
                        ) : null}
                        <Button
                          type="primary"
                          onClick={() => void handleImport('bar')}
                          loading={importingKind === 'bar'}
                          disabled={!barPreview}
                          block
                        >
                          保存为待合并批次
                        </Button>
                      </Space>
                    ),
                  },
                  {
                    key: 'tick-import',
                    label: 'Tick导入',
                    children: (
                      <Space direction="vertical" style={{ width: '100%' }} size={12}>
                        <Dragger {...buildUploadProps('tick')}>
                          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
                          <p className="ant-upload-text">上传历史 Tick CSV 文件</p>
                          <p className="ant-upload-hint">导入后可本地聚合出任意分钟周期</p>
                        </Dragger>
                        {previewLoading === 'tick' ? <Alert type="info" showIcon message="正在解析 Tick CSV..." /> : null}
                        {tickPreview ? (
                          <>
                            <Descriptions size="small" bordered column={1}>
                              <Descriptions.Item label="识别证券">{tickPreview.symbols.join(', ') || '无'}</Descriptions.Item>
                              <Descriptions.Item label="日期范围">{tickPreview.date_range[0]} ~ {tickPreview.date_range[1]}</Descriptions.Item>
                              <Descriptions.Item label="缺失字段">{tickPreview.missing_required.join(', ') || '无'}</Descriptions.Item>
                            </Descriptions>
                            {tickPreview.missing_required.length > 0 ? (
                              <Alert
                                type="warning"
                                showIcon
                                message={`缺少字段: ${tickPreview.missing_required.join(', ')}`}
                              />
                            ) : null}
                          </>
                        ) : null}
                        <Button
                          type="primary"
                          onClick={() => void handleImport('tick')}
                          loading={importingKind === 'tick'}
                          disabled={!tickPreview}
                          block
                        >
                          保存为待合并批次
                        </Button>
                      </Space>
                    ),
                  },
                ]}
              />
            </Card>
          </Col>

          <Col xs={24} xl={16} xxl={17}>
            <Card
              title="数据工作区"
              extra={
                <Button
                  type="text"
                  icon={<ReloadOutlined />}
                  onClick={() => queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })}
                >
                  刷新
                </Button>
              }
            >
              <Tabs
                items={[
                  {
                    key: 'raw-bar',
                    label: `原始K线 (${resources?.raw_bars.length || 0})`,
                    children: (
                      <ResourceTable
                        data={sortedRawBars}
                        emptyText={resourcesError ? '资源接口异常，请查看上方错误提示' : '还没有原始K线'}
                        onPreview={setSelectedResource}
                        onDelete={(resource) => void handleDeleteResource(resource)}
                        onTrain={handleTrainWithResource}
                      />
                    ),
                  },
                  {
                    key: 'raw-bar-batches',
                    label: `K线批次 (${visibleRawBarBatches.length})`,
                    children: (
                      <Space direction="vertical" size={12} style={{ width: '100%' }}>
                        <Space wrap style={{ width: '100%', justifyContent: 'space-between' }}>
                          <Space wrap>
                            <Button
                              type="primary"
                              loading={mergingKind === 'raw_bar'}
                              disabled={selectedBarBatchKeys.length < 1}
                              onClick={() => void handleMergeBatches('raw_bar')}
                            >
                              合并到正式K线
                            </Button>
                            <Text type="secondary">已选 {selectedBarBatchKeys.length} 个批次（含现有正式数据作基底）</Text>
                          </Space>
                          <Checkbox
                            checked={showMergedBatches}
                            onChange={(event) => setShowMergedBatches(event.target.checked)}
                          >
                            显示已合并
                          </Checkbox>
                        </Space>
                        {mergePreview && mergePreview.kind === 'raw_bar' ? (
                          <Alert
                            type={mergePreview.can_merge ? 'success' : 'warning'}
                            showIcon
                            message={mergePreview.can_merge ? (mergePreview.has_official ? '可合并到现有正式K线' : '可新建正式K线') : mergePreview.reason}
                            description={mergePreview.can_merge
                              ? `交集 ${formatResourceDate(mergePreview.intersection_start, mergePreview.interval)} ~ ${formatResourceDate(mergePreview.intersection_end, mergePreview.interval)}，预计 ${mergePreview.estimated_rows || 0} 行，冲突 ${mergePreview.conflict_count || 0} 个。`
                              : mergePreview.errors?.join('；')}
                          />
                        ) : null}
                        <ResourceTable
                          data={visibleRawBarBatches}
                          emptyText="还没有待合并 K线批次"
                          onPreview={setSelectedResource}
                          onDelete={(resource) => void handleDeleteResource(resource)}
                          onTrain={handleTrainWithResource}
                          selectedRowKeys={selectedBarBatchKeys}
                          onSelectionChange={setSelectedBarBatchKeys}
                        />
                      </Space>
                    ),
                  },
                  {
                    key: 'raw-tick',
                    label: `原始Tick (${resources?.raw_ticks.length || 0})`,
                    children: (
                      <ResourceTable
                        data={sortedRawTicks}
                        emptyText={resourcesError ? '资源接口异常，请查看上方错误提示' : '还没有历史 Tick'}
                        onPreview={setSelectedResource}
                        onDelete={(resource) => void handleDeleteResource(resource)}
                        onTrain={handleTrainWithResource}
                      />
                    ),
                  },
                  {
                    key: 'raw-tick-batches',
                    label: `Tick批次 (${visibleRawTickBatches.length})`,
                    children: (
                      <Space direction="vertical" size={12} style={{ width: '100%' }}>
                        <Space wrap style={{ width: '100%', justifyContent: 'space-between' }}>
                          <Space wrap>
                            <Button
                              type="primary"
                              loading={mergingKind === 'raw_tick'}
                              disabled={selectedTickBatchKeys.length < 1}
                              onClick={() => void handleMergeBatches('raw_tick')}
                            >
                              合并到正式Tick
                            </Button>
                            <Text type="secondary">已选 {selectedTickBatchKeys.length} 个批次（含现有正式数据作基底）</Text>
                          </Space>
                          <Checkbox
                            checked={showMergedBatches}
                            onChange={(event) => setShowMergedBatches(event.target.checked)}
                          >
                            显示已合并
                          </Checkbox>
                        </Space>
                        {mergePreview && mergePreview.kind === 'raw_tick' ? (
                          <Alert
                            type={mergePreview.can_merge ? 'success' : 'warning'}
                            showIcon
                            message={mergePreview.can_merge ? (mergePreview.has_official ? '可合并到现有正式Tick' : '可新建正式Tick') : mergePreview.reason}
                            description={mergePreview.can_merge
                              ? `交集 ${formatResourceDate(mergePreview.intersection_start, 'tick')} ~ ${formatResourceDate(mergePreview.intersection_end, 'tick')}，预计 ${mergePreview.estimated_rows || 0} 行，冲突 ${mergePreview.conflict_count || 0} 个。`
                              : mergePreview.errors?.join('；')}
                          />
                        ) : null}
                        <ResourceTable
                          data={visibleRawTickBatches}
                          emptyText="还没有待合并 Tick 批次"
                          onPreview={setSelectedResource}
                          onDelete={(resource) => void handleDeleteResource(resource)}
                          onTrain={handleTrainWithResource}
                          selectedRowKeys={selectedTickBatchKeys}
                          onSelectionChange={setSelectedTickBatchKeys}
                        />
                      </Space>
                    ),
                  },
                  {
                    key: 'derived',
                    label: `派生周期 (${resources?.derived_bars.length || 0})`,
                    children: (
                      <ResourceTable
                        data={sortedDerivedBars}
                        emptyText={resourcesError ? '资源接口异常，请查看上方错误提示' : '还没有派生周期'}
                        onPreview={setSelectedResource}
                        onDelete={(resource) => void handleDeleteResource(resource)}
                        onTrain={handleTrainWithResource}
                      />
                    ),
                  },
                  {
                    key: 'aggregate',
                    label: '本地聚合',
                    children: (
                      <AggregationWorkspace
                        embedded
                        resources={resources}
                        isLoading={resourcesLoading}
                        error={resourcesError}
                        onTaskStarted={handleTaskStarted}
                        onRetry={() => queryClient.invalidateQueries({ queryKey: ['alpha-data-resources'] })}
                      />
                    ),
                  },
                ]}
              />
              {resourcesLoading ? <Alert style={{ marginTop: 12 }} type="info" showIcon message="正在读取数据资源..." /> : null}
            </Card>
          </Col>
        </Row>

        <Modal
          open={!!selectedResource}
          onCancel={() => setSelectedResource(null)}
          footer={null}
          width={960}
          destroyOnClose
          title={
            selectedResource ? (
              <Space>
                <span>资源预览 · {formatResourceSymbol(selectedResource.vt_symbol)}</span>
                <Tag>{formatIntervalLabel(selectedResource.kind === 'raw_tick' ? 'tick' : selectedResource.interval)}</Tag>
                <Tag color={formatResourceKindColor(selectedResource.kind)}>
                  {formatResourceKindLabel(selectedResource.kind)}
                </Tag>
              </Space>
            ) : '资源预览'
          }
        >
          {detailLoading ? (
            <div style={{ textAlign: 'center', padding: '60px 0' }}>
              <Spin tip="加载数据预览中..." />
            </div>
          ) : resourceDetail ? (
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Descriptions size="small" column={3} bordered>
                <Descriptions.Item label="行数">
                  {new Intl.NumberFormat('zh-CN').format(resourceDetail.row_count)}
                </Descriptions.Item>
                <Descriptions.Item label="文件大小">{resourceDetail.file_size_kb.toFixed(1)} KB</Descriptions.Item>
                <Descriptions.Item label="当前周期">{formatIntervalLabel(resourceDetail.interval)}</Descriptions.Item>
                <Descriptions.Item label="起始">{formatResourceDate(resourceDetail.start, resourceDetail.interval)}</Descriptions.Item>
                <Descriptions.Item label="结束">{formatResourceDate(resourceDetail.end, resourceDetail.interval)}</Descriptions.Item>
                <Descriptions.Item label="来源说明" span={3}>{formatResourceSource(resourceDetail)}</Descriptions.Item>
              </Descriptions>
              {selectedResource?.kind === 'raw_bar' ? (
                <Card size="small" title="修改周期标签">
                  <Space direction="vertical" style={{ width: '100%' }} size={12}>
                    <Space wrap>
                      <Select
                        style={{ width: 140 }}
                        value={intervalDraft}
                        options={BAR_INTERVAL_OPTIONS}
                        onChange={(value) => setIntervalDraft(value)}
                      />
                      <Button
                        type="primary"
                        loading={relocatingInterval}
                        disabled={intervalDraft === resourceDetail.interval}
                        onClick={() => void handleRelocateInterval()}
                      >
                        保存
                      </Button>
                    </Space>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      修改标签会将数据文件移动到对应周期目录，不会修改 K 线内容。
                    </Text>
                  </Space>
                </Card>
              ) : null}
              <Table
                size="small"
                pagination={false}
                scroll={{ x: 'max-content', y: 400 }}
                rowKey={(row) => String(row.datetime || row.trade_datetime || row.timestamp || JSON.stringify(row))}
                columns={resourceDetail.columns.map((column) => ({
                  title: formatResourceColumnLabel(column),
                  dataIndex: column,
                  key: column,
                  width: 130,
                  render: (value: unknown) => {
                    if (value === null || value === undefined) {
                      return '-'
                    }
                    if (column === 'datetime') {
                      return formatResourceDate(String(value), resourceDetail.interval)
                    }
                    return String(value)
                  },
                }))}
                dataSource={resourceDetail.preview}
              />
            </Space>
          ) : (
            <Empty description="加载失败或无数据，请重试" />
          )}
        </Modal>
      </Space>
    </div>
  )
}

export default DataPrepare

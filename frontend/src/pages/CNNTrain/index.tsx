import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Divider,
  Empty,
  Form,
  Input,
  InputNumber,
  List,
  Popconfirm,
  Row,
  Segmented,
  Select,
  Space,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import {
  DeleteOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useLocation } from 'react-router-dom'
import dayjs, { type Dayjs } from 'dayjs'

import { alphaService } from '../../api/alpha'
import { cnnService } from '../../api/cnn'
import DateRangeSelector from '../../components/DateRangeSelector'
import TaskStatusPanel from '../../components/TaskStatusPanel'
import { useAvailableSymbols } from '../../hooks/useAvailableSymbols'
import { useTask } from '../../hooks/useTask'
import type { CNNArchitecture, CNNHistoryItem, CNNModelDetail, CNNModelInfo } from '../../types/cnn'
import type { ConfidenceLevel, ObservationGroup, SymbolProfileResponse } from '../../types/alpha'
import { confidenceStyle } from '../../utils/profiling'
import ProfilingPanel from './ProfilingPanel'

const { Title, Text } = Typography
const GROUP_ROLE_OPTIONS = [
  { label: '目标证券', value: 'target' },
  { label: '大盘', value: 'market' },
  { label: '板块', value: 'sector' },
  { label: '龙头', value: 'leaders' },
  { label: '自定义', value: 'custom' },
]

const LABEL_MODE_OPTIONS = [
  { label: '下一个周期', value: 'next_bar' },
  { label: 'N 个周期后', value: 'horizon_bars' },
  { label: '当日收盘', value: 'session_close' },
  { label: '次日收盘', value: 'next_session_close' },
  { label: 'OCO 止盈止损（路径依赖）', value: 'oco' },
]

const BAR_INPUT_OPTIONS = [
  { label: 'K线', value: 'bar' },
  { label: 'Tick', value: 'tick' },
]

const NEUTRAL_POLICY_OPTIONS = [
  { label: '丢弃噪声样本（推荐）', value: 'drop' },
  { label: '并入下跌类', value: 'negative' },
]

const PRICE_REF_OPTIONS = [
  { label: '收盘→收盘（旧/研究口径）', value: 'close' },
  { label: '次开盘→次开盘（对齐T+1开盘成交，推荐）', value: 'next_open' },
  { label: '次收盘→次收盘（对齐T+1收盘价MOC成交）', value: 'next_close' },
  { label: '次日均价→次日均价（对齐T+1全天VWAP成交）', value: 'next_vwap' },
]

// A 股每个交易日 240 分钟（9:30–11:30、13:00–15:00），各周期每交易日的 bar 数固定。
// 用于「按时间窗口」自动推算 lookback(T) = 观测交易日数 × 每日 bar 数。
// 派生/自定义周期不在表内（返回 undefined），此时回退到手填 bar 数。
const BARS_PER_TRADING_DAY: Record<string, number> = {
  d: 1,
  '60m': 4,
  '30m': 8,
  '15m': 16,
  '10m': 24,
  '5m': 48,
  '1m': 240,
}

// 时间维(T)硬上限：防止「分钟周期 × 多交易日」误配出巨型张量（显存/训练时长爆炸）。
const LOOKBACK_MAX = 1024

const LOOKBACK_MODE_OPTIONS = [
  { label: '按时间窗口（推荐）', value: 'window' },
  { label: '按 bar 数手填', value: 'manual' },
]

const LOSS_WEIGHTING_OPTIONS = [
  { label: '普通 BCE（不加权）', value: 'none' },
  { label: '按收益幅度加权（推荐）', value: 'magnitude' },
]

const OBJECTIVE_OPTIONS = [
  { label: '方向分类（输出上涨概率）', value: 'classification' },
  { label: '收益回归（直接预测涨跌幅）', value: 'regression' },
  { label: '路径形态分类（四类剧本概率）', value: 'path_class' },
]

/**
 * 把比率（小数）格式化为保留一位小数的百分比字符串。
 *
 * @param value - 比率，单位为小数（0.123 表示 12.3%）；null/undefined 视为缺值
 * @returns 形如 "12.3%" 的字符串；缺值返回 "-"
 */
const pct = (value?: number) => (value === undefined || value === null ? '-' : `${(value * 100).toFixed(1)}%`)

/**
 * 把数值格式化为保留三位小数的字符串，常用于 IC / AUC / F1 等指标。
 *
 * @param value - 待格式化的指标值；null/undefined 视为缺值
 * @returns 保留三位小数的字符串；缺值返回 "-"
 */
const num3 = (value?: number | null) => (value === undefined || value === null ? '-' : value.toFixed(3))

/**
 * 训练历史 Epoch 明细表。
 *
 * 根据 `objective` 自动切换列配置：
 * - `regression`：IC / RankIC / MAE / 方向准确率
 * - `path_class`：tp_auc / sl_auc / macro_f1（四分类专属指标）
 * - 其余（`classification`）：val_acc / AUC / F1
 *
 * 空值（null / undefined）统一渲染为 '-'。
 *
 * @param history - 后端 history 数组，元素为 {@link CNNHistoryItem}。
 * @param objective - 训练目标，决定列配置分支。
 */
const LossTable: React.FC<{ history: CNNHistoryItem[]; objective?: string }> = ({ history, objective }) => {
  const isReg = objective === 'regression'
  const isPathClass = objective === 'path_class'
  const columns = isReg
    ? [
        { title: 'Epoch', dataIndex: 'epoch', width: 70, fixed: 'left' as const },
        { title: 'Val Loss', dataIndex: 'val_loss', width: 90 },
        { title: 'IC', dataIndex: 'val_ic', width: 80, render: num3 },
        { title: 'RankIC', dataIndex: 'val_rank_ic', width: 90, render: num3 },
        { title: 'MAE', dataIndex: 'val_mae', width: 90, render: (v?: number) => (v === undefined ? '-' : v.toFixed(4)) },
        { title: '方向准确率', dataIndex: 'val_dir_acc', width: 100, render: (v?: number) => pct(v) },
        { title: '基线', dataIndex: 'val_baseline_acc', width: 80, render: (v?: number) => pct(v) },
      ]
    : isPathClass
      ? [
          { title: 'Epoch', dataIndex: 'epoch', width: 70, fixed: 'left' as const },
          { title: 'Val Loss', dataIndex: 'val_loss', width: 90 },
          { title: 'TP AUC', dataIndex: 'val_tp_auc', width: 90, render: num3 },
          { title: 'SL AUC', dataIndex: 'val_sl_auc', width: 90, render: num3 },
          { title: 'Macro F1', dataIndex: 'val_macro_f1', width: 100, render: num3 },
          { title: 'LR', dataIndex: 'lr', width: 80, render: (v?: number) => (v === undefined || v === null ? '—' : v.toExponential(2)) },
        ]
      : [
          { title: 'Epoch', dataIndex: 'epoch', width: 70, fixed: 'left' as const },
          { title: 'Val Loss', dataIndex: 'val_loss', width: 90 },
          { title: 'Val Acc', dataIndex: 'val_acc', width: 90, render: (value: number) => pct(value) },
          { title: '基线', dataIndex: 'val_baseline_acc', width: 80, render: (value?: number) => pct(value) },
          {
            title: '超额',
            dataIndex: 'val_excess_acc',
            width: 90,
            render: (value?: number) =>
              value === undefined || value === null ? (
                '-'
              ) : (
                <span style={{ color: value > 0 ? '#49aa19' : '#dc4446' }}>
                  {value > 0 ? '+' : ''}{(value * 100).toFixed(1)}%
                </span>
              ),
          },
          { title: 'AUC', dataIndex: 'val_auc', width: 80, render: num3 },
          { title: 'F1', dataIndex: 'val_f1', width: 80, render: (value?: number) => (value === undefined ? '-' : value.toFixed(3)) },
        ]
  return (
    <Table
      size="small"
      rowKey="epoch"
      pagination={false}
      scroll={{ y: 260, x: 720 }}
      dataSource={history}
      columns={columns}
    />
  )
}

/**
 * 把用户输入的多证券文本拆成去重前的代码列表。
 *
 * 以换行或逗号为分隔符切分，逐项去除首尾空白并丢弃空串。
 *
 * @param raw - 原始文本，证券代码以换行或逗号分隔
 * @returns 非空、已 trim 的证券代码数组；无有效内容时返回空数组
 */
const parseSymbols = (raw: string) => (
  raw
    .split(/[\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean)
)

/**
 * 把张量形状数组渲染成可读字符串。
 *
 * @param shape - 各维大小组成的数组；null/undefined 表示形状不可用
 * @returns 形如 "[6, 30, 4]" 的字符串；非数组时返回 "-"
 */
const shapeText = (shape?: number[] | null) =>
  Array.isArray(shape) ? `[${shape.join(', ')}]` : '-'

/**
 * 把画像置信度等级渲染成带悬浮说明的彩色标签。
 *
 * 颜色、文案、描述均取自 {@link confidenceStyle}，描述以 Tooltip 形式悬浮展示。
 *
 * @param confidence - 画像整体置信度等级
 * @returns 渲染好的置信度 Tag（含 Tooltip）
 */
function profilingConfidenceTag(confidence: ConfidenceLevel) {
  const style = confidenceStyle(confidence)
  return (
    <Tooltip title={style.description}>
      <Tag color={style.color} style={{ cursor: 'help' }}>
        {style.text}
      </Tag>
    </Tooltip>
  )
}

const profilingSummaryStyle: React.CSSProperties = {
  marginTop: 8,
  padding: 12,
  border: '1px solid rgba(145, 202, 255, 0.16)',
  borderRadius: 6,
  background: '#111b26',
}

const profilingSummaryLabelStyle: React.CSSProperties = {
  color: '#8c8c8c',
  fontSize: 12,
}

const profilingSummaryValueStyle: React.CSSProperties = {
  color: '#d6e4ff',
  fontWeight: 600,
}

/**
 * 最近画像摘要条：在训练表单内联展示最近一次标的画像的关键字段。
 *
 * 展示整体置信度、来源（历史画像/本次评估）、可用性、标的与周期、有效 bar
 * 数及 artifact_id；若画像携带方案建议，提示可在详情中回填训练表单。
 * 点击「查看详情」回调由父组件打开完整画像面板。
 */
const ProfilingResultSummary: React.FC<{
  /** 最近一次画像评估结果 */
  result: SymbolProfileResponse
  /** true=展示的是历史缓存画像，false=本次刚评估的画像 */
  historical: boolean
  /** 点击「查看详情」时触发，用于打开完整画像面板 */
  onOpenDetail: () => void
}> = ({ result, historical, onOpenDetail }) => {
  return (
    <section style={profilingSummaryStyle} aria-label="最近画像摘要">
      <Space direction="vertical" size={10} style={{ width: '100%' }}>
        <Space wrap style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space wrap>
            <Text strong style={{ color: '#f0f0f0' }}>最近画像</Text>
            {profilingConfidenceTag(result.overall_confidence)}
            {historical ? <Tag color="blue">历史画像</Tag> : <Tag color="green">本次评估</Tag>}
            {result.available ? <Tag color="green">可用</Tag> : <Tag color="orange">数据不可用</Tag>}
          </Space>
          <Button size="small" type="link" onClick={onOpenDetail}>
            查看详情
          </Button>
        </Space>
        <Space wrap size={[16, 8]}>
          <span>
            <span style={profilingSummaryLabelStyle}>标的 </span>
            <span style={profilingSummaryValueStyle}>{result.input.vt_symbol}</span>
          </span>
          <span>
            <span style={profilingSummaryLabelStyle}>周期 </span>
            <span style={profilingSummaryValueStyle}>{result.input.interval}</span>
          </span>
          <span>
            <span style={profilingSummaryLabelStyle}>有效 bar </span>
            <span style={profilingSummaryValueStyle}>{result.input.effective_bar_count}</span>
          </span>
          <span>
            <span style={profilingSummaryLabelStyle}>实际右边界 </span>
            <span style={profilingSummaryValueStyle}>{result.input.effective_right_bound || '-'}</span>
          </span>
          {result.artifact_id ? (
            <span>
              <span style={profilingSummaryLabelStyle}>artifact_id </span>
              <Text code copyable>
                {result.artifact_id}
              </Text>
            </span>
          ) : null}
        </Space>
        {result.suggestion ? (
          <Text type="secondary" style={{ fontSize: 12 }}>
            已生成 {result.suggestion.items.length} 条方案建议，可在详情中查看并回填训练表单。
          </Text>
        ) : null}
      </Space>
    </section>
  )
}

/**
 * 真实网络结构卡片：模块树 + 逐层形状 + 参数量。
 *
 * 数据全部来自后端加载权重后探查到的真实实例：展示输入/输出形状、可训练参数量、
 * 逐层输出形状与参数量，以及 PyTorch 原生模块树。当结构与权重不完全匹配
 * （`arch.verified === false`）或逐层形状探查失败（`arch.forward_error`）时给出告警。
 * `loading` 为 true 时显示加载态；`arch` 为 null 时显示「无法探查」空态
 * （通常因 PyTorch 未安装或模型读取失败）。
 */
const ArchitectureCard: React.FC<{
  /** 后端探查到的真实网络结构；null 表示探查失败或尚未加载 */
  arch: CNNArchitecture | null
  /** 是否正在加载并探查结构，true 时展示 Spin 加载态 */
  loading: boolean
}> = ({ arch, loading }) => {
  const layerColumns = [
    { title: '#', width: 48, render: (_: unknown, __: unknown, idx: number) => idx + 1 },
    { title: '层', dataIndex: 'name', width: 150, render: (v: string) => <Text code>{v}</Text> },
    { title: '类型', dataIndex: 'type', width: 130 },
    {
      title: '输出形状',
      dataIndex: 'output_shape',
      width: 170,
      render: (v: number[] | null) => <Text type={v ? undefined : 'secondary'}>{shapeText(v)}</Text>,
    },
    {
      title: '参数量',
      dataIndex: 'params_h',
      width: 100,
      align: 'right' as const,
      render: (v: string, row: CNNArchitecture['layers'][number]) =>
        row.params > 0 ? v : <Text type="secondary">0</Text>,
    },
  ]

  return (
    <Card
      size="small"
      title="网络结构（真实模型）"
      style={{ marginBottom: 16 }}
      extra={
        arch ? (
          <Space wrap size={4}>
            <Tag color={arch.verified ? 'green' : 'red'}>
              {arch.verified ? '结构已校验' : '结构未匹配'}
            </Tag>
            <Tag color="blue">{arch.total_params_h} 参数</Tag>
            <Tag>{arch.param_dtype}</Tag>
          </Space>
        ) : null
      }
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '24px 0' }}>
          <Spin tip="正在加载真实模型并探查结构..." />
        </div>
      ) : !arch ? (
        <Empty description="无法探查模型结构（PyTorch 未安装或模型读取失败）" />
      ) : (
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          {!arch.verified && arch.verify_message ? (
            <Alert type="warning" showIcon message="权重与结构不完全一致" description={arch.verify_message} />
          ) : (
            <Alert
              type="success"
              showIcon
              message="所见即真实：已用训练时保存的权重加载进重建的模型，结构与权重严格匹配"
            />
          )}
          {arch.forward_error ? (
            <Alert type="warning" showIcon message="逐层形状不可用" description={arch.forward_error} />
          ) : null}

          <Descriptions size="small" bordered column={2}>
            <Descriptions.Item label="输入 x 形状">{shapeText(arch.input_shapes?.x)}</Descriptions.Item>
            <Descriptions.Item label="掩码 group_mask">{shapeText(arch.input_shapes?.group_mask)}</Descriptions.Item>
            <Descriptions.Item label="输出形状">{shapeText(arch.output_shape)}</Descriptions.Item>
            <Descriptions.Item label="可训练参数">{arch.trainable_params_h}</Descriptions.Item>
          </Descriptions>

          <Table
            size="small"
            rowKey="name"
            pagination={false}
            scroll={{ y: 320, x: 560 }}
            dataSource={arch.layers}
            columns={layerColumns}
          />

          <details>
            <summary style={{ cursor: 'pointer', color: '#1677ff' }}>PyTorch 原生模块树</summary>
            <pre
              style={{
                marginTop: 8,
                padding: 12,
                background: 'rgba(0,0,0,0.04)',
                borderRadius: 6,
                fontSize: 12,
                lineHeight: 1.5,
                overflowX: 'auto',
                whiteSpace: 'pre',
              }}
            >
              {arch.module_repr}
            </pre>
          </details>
        </Space>
      )}
    </Card>
  )
}

/**
 * CNN 训练工作流页面：从选输入源到启动训练、查看已保存模型详情的完整闭环。
 *
 * 左栏四步表单：选输入源（标的/周期/时间范围）、配目标证券与语义观测组、
 * 定义标签（含 OCO 三重障碍）、设训练参数；支持「按时间窗口」自动换算回看
 * bar 数 T 并对超上限（{@link LOOKBACK_MAX}）做拦截，path_class 目标会锁定
 * OCO 标签。右栏展示已保存模型列表与选中模型的真实训练历史、评估指标、
 * 观测组与网络结构。还集成只读画像评估（{@link ProfilingPanel}），可把
 * 画像建议回填表单。
 *
 * 通过路由 `location.state` 接收预设（preset，预填标的/周期等）和
 * focusModelName（进入即打开指定模型详情）。
 */
const CNNTrain: React.FC = () => {
  const { message } = App.useApp()
  const location = useLocation()
  const queryClient = useQueryClient()
  const [form] = Form.useForm()
  const [groupForm] = Form.useForm()
  const [observationGroups, setObservationGroups] = useState<ObservationGroup[]>([])
  const [taskId, setTaskId] = useState<string | null>(null)
  const [viewDetail, setViewDetail] = useState<CNNModelDetail | null>(null)
  const [architecture, setArchitecture] = useState<CNNArchitecture | null>(null)
  const [archLoading, setArchLoading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [profilingOpen, setProfilingOpen] = useState(false)
  const [latestProfile, setLatestProfile] = useState<SymbolProfileResponse | null>(null)
  const [latestProfileHistorical, setLatestProfileHistorical] = useState(false)

  const task = useTask(taskId)
  const preset = (location.state as {
    preset?: {
      target_symbol?: string
      input_data_kind?: 'bar' | 'tick'
      input_interval?: string
      symbols?: string[]
    }
    modelName?: string
  } | null)?.preset
  const focusModelName = (location.state as { modelName?: string } | null)?.modelName

  const { data: resources } = useQuery({
    queryKey: ['alpha-data-resources'],
    queryFn: () => alphaService.getDataResources(),
  })

  const { data: models, refetch: refetchModels } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const barIntervals = useMemo(() => {
    const values = new Set<string>([
      ...(resources?.raw_bar_intervals || []),
      ...(resources?.derived_intervals || []),
      'd',
      '1m',
      '5m',
      '10m',
      '15m',
      '30m',
      '60m',
    ])
    return [...values].sort((a, b) => a.localeCompare(b))
  }, [resources])

  // 复用共享 Hook 归并“每合约可用性映射”，消除与数据准备页的重复逻辑。
  // 再从映射派生出本组件其余部分消费的 options 数组（与原 availableSymbols 等价）。
  const availabilityMap = useAvailableSymbols(resources)
  const availableSymbols = useMemo(
    () =>
      [...availabilityMap.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([sym, meta]) => ({
          value: sym,
          label: sym,
          intervals: [...meta.intervals],
          dateRange: `${meta.start.slice(0, 10)} ~ ${meta.end.slice(0, 10)}`,
          intervalRanges: meta.intervalRanges,
        })),
    [availabilityMap],
  )

  const inputDataKind = Form.useWatch('input_data_kind', form) || 'bar'
  const lookback = Form.useWatch('lookback', form) || 30
  const targetSymbol = Form.useWatch('target_symbol', form) || ''
  const labelMode = Form.useWatch('label_mode', form) || 'next_bar'
  const inputInterval = Form.useWatch('input_interval', form) || 'd'
  const objective = Form.useWatch('objective', form) || 'classification'
  const trainRange = Form.useWatch('range', form) as [Dayjs, Dayjs] | undefined

  // 回看窗口配置：window=按观测交易日数自动推算；manual=直接手填 bar 数
  const lookbackMode = Form.useWatch('lookback_mode', form) || 'window'
  const observationDays = Form.useWatch('observation_days', form) || 30
  // 当前周期每交易日 bar 数；派生/自定义周期不可推算 → null
  const barsPerDay = BARS_PER_TRADING_DAY[inputInterval] ?? null
  // 由「观测交易日数 × 每日 bar 数」推算出的 lookback(T)
  const derivedLookback = useMemo(() => {
    if (!barsPerDay) return null
    return Math.max(1, Math.round(observationDays * barsPerDay))
  }, [barsPerDay, observationDays])
  // window 模式下把推算值实时回写到真正提交的 lookback 字段（manual 模式不干预手填值）
  useEffect(() => {
    if (lookbackMode === 'window' && derivedLookback != null) {
      form.setFieldsValue({ lookback: derivedLookback })
    }
  }, [lookbackMode, derivedLookback, form])
  const lookbackOverLimit = lookback > LOOKBACK_MAX

  const targetLocalRange = useMemo(() => {
    if (!targetSymbol) {
      return null
    }
    const targetKey = targetSymbol.replace(/\.$/, '').toLowerCase()
    const meta = availableSymbols.find(
      (item) => item.value.replace(/\.$/, '').toLowerCase() === targetKey,
    )
    if (!meta) {
      return null
    }
    if (inputDataKind === 'tick') {
      return meta.intervalRanges.tick || null
    }
    return meta.intervalRanges[inputInterval] || null
  }, [availableSymbols, inputDataKind, inputInterval, targetSymbol])

  const tensorEstimate = useMemo(() => {
    const groupCount = observationGroups.length + 1
    const maxGroupWidth = Math.max(
      1,
      ...observationGroups.map((group) => group.symbols.length),
    )
    return {
      channels: 6,
      time: lookback,
      width: maxGroupWidth,
      groups: groupCount,
    }
  }, [lookback, observationGroups])

  useEffect(() => {
    form.setFieldsValue({
      name: '',
      target_symbol: '',
      input_data_kind: 'bar',
      input_interval: 'd',
      range: [dayjs().subtract(3, 'year'), dayjs()],
      label_mode: 'next_bar',
      label_horizon: 3,
      oco_take_profit_pct: 3,
      oco_stop_loss_pct: 2,
      oco_max_hold: 10,
      objective: 'classification',
      label_threshold_pct: 0.5,
      neutral_policy: 'drop',
      price_ref: 'next_open',
      loss_weighting: 'magnitude',
      epochs: 50,
      batch_size: 32,
      learning_rate: 0.001,
      lookback_mode: 'window',
      observation_days: 30,
      lookback: 30,
      dropout: 0.4,
      train_ratio: 0.7,
    })
    groupForm.setFieldsValue({
      role: 'market',
      name: '',
      symbols: [],
    })
  }, [form, groupForm])

  useEffect(() => {
    if (preset) {
      form.setFieldsValue({
        target_symbol: preset.target_symbol,
        input_data_kind: preset.input_data_kind || 'bar',
        input_interval: preset.input_interval || 'd',
      })
    }
  }, [form, preset])

  useEffect(() => {
    if (task.data?.status === 'completed') {
      refetchModels()
      queryClient.invalidateQueries({ queryKey: ['cnn-models'] })
      const taskResultName = String(task.data.result?.name || form.getFieldValue('name') || '')
      if (taskResultName) {
        void cnnService.getModel(taskResultName).then(setViewDetail).catch(() => undefined)
      }
    }
  }, [form, queryClient, refetchModels, task.data?.result, task.data?.status])

  useEffect(() => {
    if (focusModelName) {
      void cnnService.getModel(focusModelName).then(setViewDetail).catch(() => undefined)
    }
  }, [focusModelName])

  // path_class 目标依赖 OCO 三重障碍标签；选中时自动锁定 label_mode 为 'oco'。
  // 解除 path_class 时不强制重置，让用户自行选择。
  useEffect(() => {
    if (objective === 'path_class') {
      form.setFieldsValue({ label_mode: 'oco' })
    }
  }, [objective, form])

  // 当查看的模型变化时，拉取真实网络结构
  useEffect(() => {
    const name = viewDetail?.name
    if (!name) {
      setArchitecture(null)
      return
    }
    let cancelled = false
    setArchLoading(true)
    setArchitecture(null)
    cnnService
      .getModelArchitecture(name)
      .then((arch) => {
        if (!cancelled) setArchitecture(arch)
      })
      .catch(() => {
        if (!cancelled) setArchitecture(null)
      })
      .finally(() => {
        if (!cancelled) setArchLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [viewDetail?.name])

  /**
   * 校验观测组子表单并把一条新的语义观测组追加到列表。
   *
   * 证券列表支持数组（tags 模式）或多行/逗号文本，统一 trim 去空后入组。
   * 角色为 target 时拒绝添加（目标证券在上方单独配置），证券为空时拒绝添加，
   * 两种情况均以 message.warning 提示并 return。成功后重置子表单为默认值。
   * 表单校验失败由 antd 表单自身提示，这里静默吞掉异常。
   */
  const addGroup = async () => {
    try {
      const values = await groupForm.validateFields()
      const symbols: string[] = Array.isArray(values.symbols)
        ? values.symbols.map((s: string) => s.trim()).filter(Boolean)
        : parseSymbols(values.symbols || '')
      const nextGroup: ObservationGroup = {
        role: values.role,
        name: values.name,
        symbols,
      }
      if (nextGroup.role === 'target') {
        message.warning('目标证券在上方单独配置，无需重复添加 target 组')
        return
      }
      if (nextGroup.symbols.length === 0) {
        message.warning('请至少选择一个证券')
        return
      }
      setObservationGroups((current) => [...current, nextGroup])
      groupForm.resetFields()
      groupForm.setFieldsValue({ role: 'market', name: '', symbols: [] })
    } catch {
      // handled by form
    }
  }

  /**
   * 按下标删除一条观测组。
   *
   * @param index - 待删除观测组在 observationGroups 中的下标
   */
  const removeGroup = (index: number) => {
    setObservationGroups((current) => current.filter((_item, itemIndex) => itemIndex !== index))
  }

  /**
   * 按当前关键配置自动生成并回填模型名称。
   *
   * 命名格式为 `cnn_{标的}_{周期}_{标签}_{目标}_T{回看}_{计价口径}_{时间戳}`：
   * 标的取目标证券去除非字母数字字符（缺失时用 'sym'）；周期对 tick 输入加 'tk' 前缀；
   * 标签按 label_mode 缩写（nb/h{N}/sc/nsc/oco_tp{x}_sl{y}_h{z}）；目标 cls/reg；
   * 计价口径 cl/no/nc/vw。末尾 MMDDHHmm 时间戳保证唯一、避免重名覆盖。
   * 结果直接写入表单 name 字段，用户仍可手动修改。
   */
  const autoFillModelName = () => {
    const sym = (targetSymbol || '').replace(/\.$/, '').replace(/[^0-9A-Za-z]/g, '') || 'sym'
    const kindInterval = `${inputDataKind === 'tick' ? 'tk' : ''}${inputInterval}`

    const horizon = form.getFieldValue('label_horizon') || 3
    const ocoTp = form.getFieldValue('oco_take_profit_pct') || 0
    const ocoSl = form.getFieldValue('oco_stop_loss_pct') || 0
    const ocoHold = form.getFieldValue('oco_max_hold') || 0
    const labelTagMap: Record<string, string> = {
      next_bar: 'nb',
      horizon_bars: `h${horizon}`,
      session_close: 'sc',
      next_session_close: 'nsc',
      oco: `oco_tp${ocoTp}_sl${ocoSl}_h${ocoHold}`,
    }
    const labelTag = labelTagMap[labelMode] || labelMode

    const objTag = objective === 'regression' ? 'reg' : 'cls'

    const priceRef = form.getFieldValue('price_ref') || 'next_open'
    const priceTagMap: Record<string, string> = {
      close: 'cl',
      next_open: 'no',
      next_close: 'nc',
      next_vwap: 'vw',
    }
    const priceTag = priceTagMap[priceRef] || priceRef

    const stamp = dayjs().format('MMDDHHmm')
    const name = `cnn_${sym}_${kindInterval}_${labelTag}_${objTag}_T${lookback}_${priceTag}_${stamp}`
    form.setFieldsValue({ name })
  }

  /**
   * 校验主表单并提交 CNN 训练任务，记下返回的 task_id 以便轮询进度。
   *
   * 提交前做两道前端拦截：目标证券为空、回看窗口 T 超过 {@link LOOKBACK_MAX}，
   * 均以 message.warning 提示并 return。阈值/止盈/止损按百分比转小数（除以 100
   * 并取非负）；label_horizon 仅在 horizon_bars 模式下传，take_profit/stop_loss/
   * max_hold 仅在 oco 模式下传；regression 与 path_class 目标强制 loss_weighting 为 'none'。
   * 启动成功后写入 taskId 并提示；表单校验或网络异常时以 message.error 提示。
   * 全程通过 submitting 控制按钮 loading 态。
   */
  const handleTrain = async () => {
    try {
      const values = await form.validateFields()
      if (!targetSymbol) {
        message.warning('请先填写目标证券')
        return
      }
      if (values.lookback > LOOKBACK_MAX) {
        message.warning(
          `回看窗口 T=${values.lookback} 超过上限 ${LOOKBACK_MAX}，` +
            '请减少观测交易日数或改用更大周期（如 5m/15m/d）后再训练。',
        )
        return
      }
      setSubmitting(true)
      const taskStart = await cnnService.train({
        name: values.name,
        start: values.range[0].format('YYYY-MM-DD'),
        end: values.range[1].format('YYYY-MM-DD'),
        target_symbol: values.target_symbol,
        input_data_kind: values.input_data_kind,
        input_interval: values.input_interval,
        observation_groups: observationGroups,
        objective: values.objective,
        label_spec: {
          mode: values.label_mode,
          horizon: values.label_mode === 'horizon_bars' ? values.label_horizon : undefined,
          threshold: Math.max(0, (values.label_threshold_pct || 0) / 100),
          neutral_policy: values.neutral_policy,
          price_ref: values.price_ref,
          take_profit:
            values.label_mode === 'oco' ? Math.max(0, (values.oco_take_profit_pct || 0) / 100) : undefined,
          stop_loss:
            values.label_mode === 'oco' ? Math.max(0, (values.oco_stop_loss_pct || 0) / 100) : undefined,
          max_hold: values.label_mode === 'oco' ? values.oco_max_hold : undefined,
        },
        loss_weighting: (values.objective === 'regression' || values.objective === 'path_class') ? 'none' : values.loss_weighting,
        epochs: values.epochs,
        batch_size: values.batch_size,
        learning_rate: values.learning_rate,
        lookback: values.lookback,
        dropout: values.dropout,
        train_ratio: values.train_ratio,
      })
      setTaskId(taskStart.task_id)
      message.success('CNN 训练任务已启动')
    } catch (error) {
      if (error instanceof Error) {
        message.error(error.message)
      }
    } finally {
      setSubmitting(false)
    }
  }

  /**
   * 拉取指定模型的训练详情并填入右栏详情区。
   *
   * 成功后写入 viewDetail（触发网络结构探查）；失败以 message.error 提示。
   *
   * @param name - 模型名称
   */
  const handleViewModel = async (name: string) => {
    try {
      const detail = await cnnService.getModel(name)
      setViewDetail(detail)
    } catch {
      message.error('加载模型详情失败')
    }
  }

  /**
   * 删除指定 CNN 模型并刷新列表。
   *
   * 成功后提示并 refetch 模型列表；若被删模型正是当前详情区展示的模型，
   * 一并清空 viewDetail。失败以 message.error 提示。
   *
   * @param name - 待删除模型名称
   */
  const handleDeleteModel = async (name: string) => {
    try {
      await cnnService.deleteModel(name)
      message.success('CNN 模型已删除')
      refetchModels()
      if (viewDetail?.name === name) {
        setViewDetail(null)
      }
    } catch {
      message.error('删除失败')
    }
  }

  /**
   * 把画像面板给出的方案建议回填到训练表单。
   *
   * 当前为 path_class 目标时，强制把建议里的 label_mode 覆盖为 'oco'（path_class
   * 必须用 OCO 标签，避免禁用的 Select 显示非法值），并在提示中额外说明已锁定。
   * 有可填字段时按是否存在未映射建议给出不同成功提示；无可填字段时给 warning。
   *
   * @param values - 可直接回填的表单字段键值对（已由画像面板映射）
   * @param unmappedCount - 未能自动映射、需人工处理的建议条数
   */
  const handleApplyProfilingSuggestion = (values: Record<string, unknown>, unmappedCount: number) => {
    if (Object.keys(values).length > 0) {
      // path_class 必须使用 OCO 标签，若建议中携带了其他 label_mode，强制覆盖为 'oco'
      // 并额外提示用户，避免禁用的 Select 显示非法值。
      const isPathClass = objective === 'path_class'
      const overrodeLabel = isPathClass && values.label_mode !== undefined && values.label_mode !== 'oco'
      const merged = isPathClass ? { ...values, label_mode: 'oco' } : values
      form.setFieldsValue(merged)
      const baseMsg =
        unmappedCount > 0
          ? `已填充可映射建议，${unmappedCount} 条需人工处理`
          : '已填充画像建议，请确认后再训练'
      message.success(overrodeLabel ? `${baseMsg}（建议的标签模式已被锁定为 OCO）` : baseMsg)
    } else {
      message.warning('当前建议没有可直接填充的训练字段')
    }
  }

  const modelList = (
    <Table<CNNModelInfo>
      size="small"
      rowKey="name"
      dataSource={models || []}
      pagination={{ pageSize: 6, showSizeChanger: false }}
      locale={{ emptyText: '还没有 CNN 模型' }}
      columns={[
        {
          title: '模型',
          key: 'name',
          render: (_value, record) => (
            <Space direction="vertical" size={2}>
              <Button type="link" style={{ padding: 0 }} onClick={() => void handleViewModel(record.name)}>
                {record.name}
              </Button>
              <Space size={4} wrap>
                <Tag>{record.input_interval || 'd'}</Tag>
                <Tag color="purple">{record.group_count || 1} 组</Tag>
              </Space>
            </Space>
          ),
        },
        {
          title: '目标证券',
          dataIndex: 'target_symbol',
          width: 130,
        },
        {
          title: '最佳损失',
          dataIndex: 'best_val_loss',
          width: 110,
          render: (value?: number) => value?.toFixed(4) || '-',
        },
        {
          title: '操作',
          key: 'actions',
          width: 110,
          render: (_value, record) => (
            <Space size="small">
              <Button size="small" icon={<EyeOutlined />} onClick={() => void handleViewModel(record.name)} />
              <Popconfirm title="确认删除这个 CNN 模型？" onConfirm={() => void handleDeleteModel(record.name)}>
                <Button size="small" danger icon={<DeleteOutlined />} />
              </Popconfirm>
            </Space>
          ),
        },
      ]}
    />
  )

  return (
    <div className="page-enter">
      <Space direction="vertical" size={20} style={{ width: '100%' }}>
        <div>
          <Title level={3} style={{ marginBottom: 4 }}>CNN 训练工作流</Title>
          <Text type="secondary">
            先选输入源和周期，再配置目标证券与语义观测组，最后定义标签和训练参数。
          </Text>
        </div>

        <TaskStatusPanel task={task.data || null} title="当前训练任务" />

        <Row gutter={[16, 16]}>
          <Col xs={24} xl={10}>
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Card title="步骤 1 · 选择输入源">
                <Form form={form} layout="vertical">
                  <Form.Item label="模型名称" required>
                    <Space.Compact style={{ width: '100%' }}>
                      <Form.Item name="name" noStyle rules={[{ required: true, message: '请输入模型名称' }]}>
                        <Input placeholder="cnn_market_context_v1" />
                      </Form.Item>
                      <Tooltip
                        title={
                          <span>
                            按当前关键配置自动生成模型名称，格式：
                            <br />
                            cnn_标的_周期_标签_目标_T回看_计价口径_时间戳
                            <br />
                            标签：nb=下一周期 / h{'{N}'}=N周期后 / sc=当日收盘 / nsc=次日收盘
                            <br />
                            目标：cls=分类 / reg=回归
                            <br />
                            计价口径：no=次开盘 / nc=次收盘 / vw=次日均价 / cl=收盘
                            <br />
                            末尾 MMDDHHmm 时间戳保证唯一、避免重名覆盖；生成后仍可手动修改。
                          </span>
                        }
                      >
                        <Button onClick={autoFillModelName}>自动填充</Button>
                      </Tooltip>
                    </Space.Compact>
                  </Form.Item>
                  <Form.Item
                    label="输入数据类型"
                    name="input_data_kind"
                    rules={[{ required: true, message: '请选择输入类型' }]}
                  >
                    <Select options={BAR_INPUT_OPTIONS} />
                  </Form.Item>
                  <Form.Item
                    label="输入周期"
                    name="input_interval"
                    rules={[{ required: true, message: '请选择输入周期' }]}
                    extra={inputDataKind === 'tick' ? 'Tick 会先按这个周期在本地聚合，再进入 CNN。' : '可直接使用原始K线或派生周期。'}
                  >
                    <Select options={barIntervals.map((interval) => ({ label: interval, value: interval }))} />
                  </Form.Item>
                  <Form.Item
                    label="时间范围"
                    name="range"
                    rules={[{ required: true, message: '请选择时间范围' }]}
                    extra={targetLocalRange
                      ? '可先选目标证券，再点「使用本地全区间」或快捷区间。'
                      : '选择目标证券后，可一键匹配本地数据区间。'}
                  >
                    <DateRangeSelector localRange={targetLocalRange} />
                  </Form.Item>
                </Form>
              </Card>

              <Card title="步骤 2 · 目标证券与观测组">
                <Form form={form} layout="vertical">
                  <Form.Item
                    label="目标证券"
                    name="target_symbol"
                    rules={[{ required: true, message: '请选择或输入目标证券' }]}
                    extra={availableSymbols.length > 0 ? `已有 ${availableSymbols.length} 个证券的本地数据可用` : '暂无本地数据资源，请先在数据准备中下载'}
                  >
                    <Select
                      showSearch
                      allowClear
                      placeholder="选择已有数据，或输入 000415.SZSE / sz000415"
                      optionFilterProp="label"
                      notFoundContent={availableSymbols.length === 0 ? '暂无本地数据' : '未找到匹配证券'}
                      options={availableSymbols.map((sym) => ({
                        value: sym.value,
                        label: sym.label,
                      }))}
                      optionRender={(option) => {
                        const meta = availableSymbols.find((s) => s.value === option.value)
                        return (
                          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                            <Space size={4}>
                              <DatabaseOutlined style={{ color: '#52c41a' }} />
                              <span>{option.label}</span>
                            </Space>
                            {meta ? (
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                {meta.intervals.join('/')} · {meta.dateRange}
                              </Text>
                            ) : null}
                          </Space>
                        )
                      }}
                    />
                  </Form.Item>
                  <Space wrap style={{ marginTop: -8, marginBottom: 8 }}>
                    <Tooltip title={targetSymbol ? '基于当前目标证券、输入周期和观测组做只读画像评估' : '先选择目标证券后再评估'}>
                      <span>
                        <Button
                          size="small"
                          icon={<ExperimentOutlined />}
                          disabled={!targetSymbol}
                          onClick={() => setProfilingOpen(true)}
                        >
                          评估该标的
                        </Button>
                      </span>
                    </Tooltip>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      只读诊断，可按建议回填训练表单，不会启动训练。
                    </Text>
                  </Space>
                  {latestProfile ? (
                    <ProfilingResultSummary
                      result={latestProfile}
                      historical={latestProfileHistorical}
                      onOpenDetail={() => setProfilingOpen(true)}
                    />
                  ) : null}
                </Form>

                <Divider style={{ margin: '8px 0 16px' }} />

                <Form form={groupForm} layout="vertical">
                  <Form.Item label="分组角色" name="role" rules={[{ required: true, message: '请选择分组角色' }]}>
                    <Select options={GROUP_ROLE_OPTIONS.filter((item) => item.value !== 'target')} />
                  </Form.Item>
                  <Form.Item label="分组名称" name="name" rules={[{ required: true, message: '请输入分组名称' }]}>
                    <Input placeholder="沪深300 / 银行板块 / 龙头组" />
                  </Form.Item>
                  <Form.Item
                    label="证券列表"
                    name="symbols"
                    rules={[{ required: true, message: '请选择至少一个证券' }]}
                  >
                    <Select
                      mode="tags"
                      placeholder="选择已有数据，或输入 000415.SZSE / sz000415"
                      optionFilterProp="label"
                      tokenSeparators={[',', '\n', ' ']}
                      style={{ width: '100%' }}
                      options={availableSymbols.map((sym) => ({
                        value: sym.value,
                        label: sym.label,
                      }))}
                      optionRender={(option) => {
                        const meta = availableSymbols.find((s) => s.value === option.value)
                        return (
                          <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                            <Space size={4}>
                              <DatabaseOutlined style={{ color: '#52c41a' }} />
                              <span>{option.label}</span>
                            </Space>
                            {meta ? (
                              <Text type="secondary" style={{ fontSize: 12 }}>
                                {meta.intervals.join('/')}
                              </Text>
                            ) : null}
                          </Space>
                        )
                      }}
                    />
                  </Form.Item>
                  <Button onClick={() => void addGroup()} block>
                    添加观测组
                  </Button>
                </Form>

                <Divider style={{ margin: '16px 0' }} />

                {observationGroups.length > 0 ? (
                  <List
                    size="small"
                    dataSource={observationGroups}
                    renderItem={(group, index) => (
                      <List.Item
                        actions={[
                          <Button key="delete" type="link" danger onClick={() => removeGroup(index)}>删除</Button>,
                        ]}
                      >
                        <List.Item.Meta
                          title={
                            <Space wrap>
                              <Text strong>{group.name}</Text>
                              <Tag>{group.role}</Tag>
                              <Tag color="purple">{group.symbols.length} 只</Tag>
                            </Space>
                          }
                          description={group.symbols.join(', ')}
                        />
                      </List.Item>
                    )}
                  />
                ) : (
                  <Empty description="还没有观测组。可以添加大盘、板块、龙头或自定义组。" />
                )}
              </Card>

              <Card title="步骤 3 · 标签定义">
                <Form form={form} layout="vertical">
                  <Form.Item
                    label="预测目标"
                    name="objective"
                    tooltip="方向分类：输出上涨概率；收益回归：直接预测涨跌幅，分数与幅度单调对应，可按预测收益排序/定仓；路径形态分类：输出四类剧本概率（先触止盈/先触止损/到期小涨/到期小跌），需搭配 OCO 标签。"
                  >
                    <Select options={OBJECTIVE_OPTIONS} />
                  </Form.Item>
                  {objective === 'regression' ? (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginBottom: 16 }}
                      message="回归模式：标签为连续未来收益，损失用 Huber，评估看 IC/方向准确率；阈值仅用于剔除过小噪声，损失加权与噪声并类不适用。"
                    />
                  ) : null}
                  {objective === 'path_class' ? (
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginBottom: 16 }}
                      message="路径形态分类：标签模式已锁定为 OCO 三重障碍，输出四类剧本概率（先触止盈/先触止损/到期小涨/到期小跌）；损失加权不适用；回测可启用 veto_threshold 过滤高止损概率信号。"
                    />
                  ) : null}
                  <Form.Item
                    label="标签模式"
                    name="label_mode"
                    rules={[{ required: true, message: '请选择标签模式' }]}
                    extra={objective === 'path_class' ? '路径形态分类依赖 OCO 三重障碍标签，已自动锁定。' : undefined}
                  >
                    <Select options={LABEL_MODE_OPTIONS} disabled={objective === 'path_class'} />
                  </Form.Item>
                  {labelMode === 'horizon_bars' ? (
                    <Form.Item
                      label="预测跨度（bar）"
                      name="label_horizon"
                      rules={[{ required: true, message: '请填写跨度' }]}
                    >
                      <InputNumber min={1} max={120} style={{ width: '100%' }} />
                    </Form.Item>
                  ) : null}
                  {labelMode === 'oco' ? (
                    <>
                      <Alert
                        type="info"
                        showIcon
                        style={{ marginBottom: 16 }}
                        message="OCO 三重障碍标签：T+1 开盘建仓，持有期内先触止盈/止损按触发价计收益，到期未触发按时间止损（次开盘）平仓。回归=真实出场收益；分类=止盈→1、止损→0、时间止损按收益符号。同根 bar 双触发时保守假设止损先到。"
                      />
                      <Row gutter={12}>
                        <Col span={8}>
                          <Form.Item
                            label="止盈幅度 (%)"
                            name="oco_take_profit_pct"
                            rules={[{ required: true, message: '请填写止盈幅度' }]}
                            tooltip="触及该涨幅即止盈出场（如 3 表示 +3%）。"
                          >
                            <InputNumber min={0.1} max={50} step={0.1} style={{ width: '100%' }} addonAfter="%" />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item
                            label="止损幅度 (%)"
                            name="oco_stop_loss_pct"
                            rules={[{ required: true, message: '请填写止损幅度' }]}
                            tooltip="触及该跌幅即止损出场（如 2 表示 -2%）。"
                          >
                            <InputNumber min={0.1} max={50} step={0.1} style={{ width: '100%' }} addonAfter="%" />
                          </Form.Item>
                        </Col>
                        <Col span={8}>
                          <Form.Item
                            label="最大持有（bar）"
                            name="oco_max_hold"
                            rules={[{ required: true, message: '请填写最大持有 bar 数' }]}
                            tooltip="持有期内都不触发时，在第 max_hold+1 根开盘按时间止损平仓。"
                          >
                            <InputNumber min={1} max={120} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                      </Row>
                    </>
                  ) : null}
                  {labelMode === 'session_close' && inputInterval === 'd' ? (
                    <Alert
                      type="warning"
                      showIcon
                      style={{ marginBottom: 16 }}
                      message="当日收盘标签只适用于分钟级输入"
                    />
                  ) : null}
                  <Row gutter={12}>
                    <Col span={12}>
                      <Form.Item
                        label="最小波动阈值 (%)"
                        name="label_threshold_pct"
                        tooltip="|未来收益|≤该阈值视为噪声样本；0 关闭去噪。建议设为单边成本的约 2 倍。"
                      >
                        <InputNumber min={0} max={10} step={0.1} style={{ width: '100%' }} addonAfter="%" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="噪声样本处理" name="neutral_policy">
                        <Select options={NEUTRAL_POLICY_OPTIONS} disabled={objective === 'regression'} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Form.Item
                    label="收益计价口径"
                    name="price_ref"
                    tooltip="计价口径与回测撮合成交价一一对齐：next_open=T+1开盘、next_close=T+1收盘(MOC)、next_vwap=T+1全天均价(VWAP)；close为研究口径(实盘吃不到)。"
                  >
                    <Select options={PRICE_REF_OPTIONS} />
                  </Form.Item>
                </Form>
              </Card>

              <Card title="步骤 4 · 训练参数">
                <Form form={form} layout="vertical">
                  <Row gutter={12}>
                    <Col span={12}>
                      <Form.Item label="Epochs" name="epochs">
                        <InputNumber min={10} max={300} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="Batch Size" name="batch_size">
                        <InputNumber min={8} max={256} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="Learning Rate" name="learning_rate">
                        <InputNumber min={0.0001} max={0.1} step={0.0001} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={24}>
                      <Form.Item
                        label="回看窗口(T) 配置方式"
                        name="lookback_mode"
                        tooltip="按时间窗口：填观测交易日数，按输入周期自动换算回看 bar 数 T；按 bar 数：直接手填 T。"
                      >
                        <Segmented options={LOOKBACK_MODE_OPTIONS} />
                      </Form.Item>
                    </Col>
                    {lookbackMode === 'window' ? (
                      <>
                        <Col span={12}>
                          <Form.Item
                            label="观测交易日数"
                            name="observation_days"
                            tooltip="想让模型回看多少个交易日。A股每交易日 240 分钟，按输入周期换算成 bar 数 T。"
                            extra={
                              barsPerDay
                                ? `每交易日 ${barsPerDay} 根 × ${observationDays} 日 → T = ${derivedLookback}`
                                : `周期「${inputInterval}」无法自动换算，请改用「按 bar 数手填」`
                            }
                          >
                            <InputNumber min={1} max={500} style={{ width: '100%' }} addonAfter="交易日" />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item label="回看 bar 数 T（自动推算）" name="lookback">
                            <InputNumber style={{ width: '100%' }} disabled />
                          </Form.Item>
                        </Col>
                      </>
                    ) : (
                      <Col span={12}>
                        <Form.Item label="Lookback（回看 bar 数 T）" name="lookback">
                          <InputNumber min={10} max={LOOKBACK_MAX} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                    )}
                    <Col span={12}>
                      <Form.Item label="Dropout" name="dropout">
                        <InputNumber min={0} max={0.9} step={0.05} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="Train Ratio" name="train_ratio">
                        <InputNumber min={0.5} max={0.95} step={0.05} style={{ width: '100%' }} />
                      </Form.Item>
                    </Col>
                    <Col span={24}>
                      <Form.Item
                        label="损失加权"
                        name="loss_weighting"
                        tooltip="magnitude 按 |未来收益| 加权，让大波动样本主导梯度，避免对 +0.01% 和 +5% 一视同仁。回归模式下不适用。"
                      >
                        <Select options={LOSS_WEIGHTING_OPTIONS} disabled={objective === 'regression' || objective === 'path_class'} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>

                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 16 }}
                  message={`输入张量预估: ${tensorEstimate.channels} x ${tensorEstimate.time} x ${tensorEstimate.width} x ${tensorEstimate.groups}`}
                  description={`目标证券：${targetSymbol || '未填写'}；当前输入周期：${inputInterval}`}
                />

                {lookbackMode === 'window' && !barsPerDay ? (
                  <Alert
                    type="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                    message={`周期「${inputInterval}」无法按交易日自动换算 T`}
                    description="该周期不在标准换算表内（派生/自定义周期），请切换到「按 bar 数手填」直接设置回看 bar 数。"
                  />
                ) : null}

                {lookbackOverLimit ? (
                  <Alert
                    type="error"
                    showIcon
                    style={{ marginBottom: 16 }}
                    message={`回看窗口 T=${lookback} 超过上限 ${LOOKBACK_MAX}`}
                    description="分钟周期叠加多交易日会产生巨型张量（显存/训练时长爆炸）。请减少观测交易日数，或改用更大周期（如 5m/15m/d）。"
                  />
                ) : null}

                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  block
                  disabled={lookbackOverLimit || (lookbackMode === 'window' && !barsPerDay)}
                  loading={submitting || task.data?.status === 'running'}
                  onClick={() => void handleTrain()}
                >
                  启动 CNN 训练
                </Button>
              </Card>
            </Space>
          </Col>

          <Col xs={24} xl={14}>
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Card
                title="已保存 CNN 模型"
                extra={<Button type="text" icon={<ReloadOutlined />} onClick={() => void refetchModels()} />}
              >
                {modelList}
              </Card>

              {viewDetail ? (
                <Card
                  title={`训练详情 · ${viewDetail.name}`}
                  extra={
                    <Space wrap>
                      <Tag color="purple">{String(viewDetail.train_config.input_interval || viewDetail.input_interval || 'd')}</Tag>
                      <Tag color="gold">最佳 Epoch {viewDetail.best_epoch || 0}</Tag>
                    </Space>
                  }
                >
                  <Descriptions size="small" bordered column={2} style={{ marginBottom: 16 }}>
                    <Descriptions.Item label="目标证券">{String(viewDetail.train_config.target_symbol || '-')}</Descriptions.Item>
                    <Descriptions.Item label="输入">{String(viewDetail.train_config.input_data_kind || 'bar')} / {String(viewDetail.train_config.input_interval || 'd')}</Descriptions.Item>
                    <Descriptions.Item label="预测目标">
                      {(() => {
                        const obj = String(viewDetail.train_config.objective || 'classification')
                        if (obj === 'regression') return '收益回归'
                        if (obj === 'path_class') return '路径形态分类（四类）'
                        return '方向分类'
                      })()}
                    </Descriptions.Item>
                    <Descriptions.Item label="标签">{String((viewDetail.train_config.label_spec as { mode?: string } | undefined)?.mode || '-')}</Descriptions.Item>
                    <Descriptions.Item label="最佳验证损失">{viewDetail.best_val_loss?.toFixed(4) || '-'}</Descriptions.Item>
                    <Descriptions.Item label="标签阈值">
                      {(() => {
                        const t = (viewDetail.train_config.label_spec as { threshold?: number } | undefined)?.threshold
                        return t ? `${(t * 100).toFixed(2)}%` : '关闭'
                      })()}
                    </Descriptions.Item>
                    <Descriptions.Item label="计价口径">
                      {String((viewDetail.train_config.label_spec as { price_ref?: string } | undefined)?.price_ref || 'close')}
                    </Descriptions.Item>
                    <Descriptions.Item label="损失加权">
                      {String(viewDetail.train_config.loss_weighting || 'none')}
                    </Descriptions.Item>
                    <Descriptions.Item label="观测组数">{String((viewDetail.model_config.group_count as number | undefined) || 1)}</Descriptions.Item>
                  </Descriptions>

                  {(() => {
                    const best = viewDetail.history?.find((item) => item.epoch === viewDetail.best_epoch)
                    if (!best) return null
                    const detailObjective = String(viewDetail.train_config.objective || 'classification')
                    const excess = best.val_excess_acc
                    const hasExcess = excess !== undefined && excess !== null
                    const beats = hasExcess && excess > 0
                    const valPosRatio = (viewDetail.dataset_info as Record<string, number> | undefined)?.val_pos_ratio
                    if (detailObjective === 'regression') {
                      return (
                        <Card size="small" title="模型评估（最佳 Epoch · 回归）" style={{ marginBottom: 16 }}>
                          <Descriptions size="small" bordered column={2}>
                            <Descriptions.Item label="IC">{num3(best.val_ic)}</Descriptions.Item>
                            <Descriptions.Item label="RankIC">{num3(best.val_rank_ic)}</Descriptions.Item>
                            <Descriptions.Item label="MAE">{best.val_mae === undefined ? '-' : best.val_mae.toFixed(4)}</Descriptions.Item>
                            <Descriptions.Item label="RMSE">{best.val_rmse === undefined ? '-' : best.val_rmse.toFixed(4)}</Descriptions.Item>
                            <Descriptions.Item label="方向准确率">
                              <Space size={6}>
                                {pct(best.val_dir_acc)}
                                <Tag color={beats ? 'green' : 'red'}>{beats ? '跑赢基线' : '未跑赢基线'}</Tag>
                              </Space>
                            </Descriptions.Item>
                            <Descriptions.Item label="多数类基线">{pct(best.val_baseline_acc)}</Descriptions.Item>
                          </Descriptions>
                        </Card>
                      )
                    }
                    if (detailObjective === 'path_class') {
                      // class_distribution 来自训练 result，通过 dataset_info 携带
                      const classDist = (viewDetail.dataset_info as Record<string, unknown> | undefined)?.class_distribution as
                        | { tp_first?: number; sl_first?: number; time_up?: number; time_down?: number }
                        | undefined
                      // path_class 的"跑赢"判据：TP AUC > 0.5（先触止盈识别力超随机）。
                      // 后端 epoch_row 不写 val_excess_acc，不能复用通用 beats/excess 变量。
                      const tpAuc = best.val_tp_auc
                      const tpBeats = tpAuc !== undefined && tpAuc !== null && tpAuc > 0.5
                      return (
                        <Card size="small" title="模型评估（最佳 Epoch · 路径形态分类）" style={{ marginBottom: 16 }}>
                          <Descriptions size="small" bordered column={2}>
                            <Descriptions.Item label="TP AUC（先触止盈）">{num3(best.val_tp_auc)}</Descriptions.Item>
                            <Descriptions.Item label="SL AUC（先触止损）">{num3(best.val_sl_auc)}</Descriptions.Item>
                            <Descriptions.Item label="Macro F1">{num3(best.val_macro_f1)}</Descriptions.Item>
                            <Descriptions.Item label="验证损失">{best.val_loss?.toFixed(4) ?? '-'}</Descriptions.Item>
                            <Descriptions.Item label="TP 识别力">
                              <Tag color={tpBeats ? 'green' : 'red'}>
                                {tpBeats ? 'TP 识别力超随机' : 'TP 识别力未超随机'}
                              </Tag>
                            </Descriptions.Item>
                            <Descriptions.Item label="先触止盈占比(验证)">{pct(valPosRatio)}</Descriptions.Item>
                          </Descriptions>
                          {classDist ? (
                            <Descriptions
                              size="small"
                              bordered
                              column={2}
                              style={{ marginTop: 12 }}
                              title="样本类别分布（训练集）"
                            >
                              <Descriptions.Item label="先触止盈">{classDist.tp_first ?? '-'}</Descriptions.Item>
                              <Descriptions.Item label="先触止损">{classDist.sl_first ?? '-'}</Descriptions.Item>
                              <Descriptions.Item label="到期小涨">{classDist.time_up ?? '-'}</Descriptions.Item>
                              <Descriptions.Item label="到期小跌">{classDist.time_down ?? '-'}</Descriptions.Item>
                            </Descriptions>
                          ) : null}
                        </Card>
                      )
                    }
                    return (
                      <Card size="small" title="模型评估（最佳 Epoch）" style={{ marginBottom: 16 }}>
                        <Descriptions size="small" bordered column={2}>
                          <Descriptions.Item label="验证准确率">{pct(best.val_acc)}</Descriptions.Item>
                          <Descriptions.Item label="多数类基线">{pct(best.val_baseline_acc)}</Descriptions.Item>
                          <Descriptions.Item label="超额准确率">
                            <Space size={6}>
                              <span style={{ color: beats ? '#49aa19' : '#dc4446' }}>
                                {hasExcess ? `${excess > 0 ? '+' : ''}${(excess * 100).toFixed(1)}%` : '-'}
                              </span>
                              <Tag color={beats ? 'green' : 'red'}>{beats ? '跑赢基线' : '未跑赢基线'}</Tag>
                            </Space>
                          </Descriptions.Item>
                          <Descriptions.Item label="AUC">
                            {best.val_auc === undefined || best.val_auc === null ? '-' : best.val_auc.toFixed(3)}
                          </Descriptions.Item>
                          <Descriptions.Item label="F1">
                            {best.val_f1 === undefined ? '-' : best.val_f1.toFixed(3)}
                          </Descriptions.Item>
                          <Descriptions.Item label="正样本比例(验证)">{pct(valPosRatio)}</Descriptions.Item>
                        </Descriptions>
                      </Card>
                    )
                  })()}

                  <Card size="small" title="观测组配置" style={{ marginBottom: 16 }}>
                    {Array.isArray(viewDetail.train_config.observation_groups) && viewDetail.train_config.observation_groups.length > 0 ? (
                      <List
                        size="small"
                        dataSource={viewDetail.train_config.observation_groups as Array<Record<string, unknown>>}
                        renderItem={(group) => (
                          <List.Item>
                            <List.Item.Meta
                              title={
                                <Space wrap>
                                  <Text strong>{String(group.name || '-')}</Text>
                                  <Tag>{String(group.role || 'custom')}</Tag>
                                </Space>
                              }
                              description={Array.isArray(group.symbols) ? group.symbols.join(', ') : '-'}
                            />
                          </List.Item>
                        )}
                      />
                    ) : (
                      <Empty description="模型未记录观测组信息" />
                    )}
                  </Card>

                  <ArchitectureCard arch={architecture} loading={archLoading} />

                  <Card size="small" title="训练历史">
                    <LossTable
                      history={viewDetail.history}
                      objective={String(viewDetail.train_config.objective || 'classification')}
                    />
                  </Card>
                </Card>
              ) : (
                <Card title="训练详情">
                  <Empty description="选择一个 CNN 模型后，这里会显示真实训练历史、观测组和标签配置" />
                </Card>
              )}
            </Space>
          </Col>
        </Row>
      </Space>
      <ProfilingPanel
        open={profilingOpen}
        onClose={() => setProfilingOpen(false)}
        targetSymbol={targetSymbol}
        interval={inputInterval}
        defaultAsOf={trainRange?.[0] || dayjs().subtract(3, 'year')}
        observationGroups={observationGroups}
        onApplySuggestion={handleApplyProfilingSuggestion}
        onResultChange={(profile, historical) => {
          setLatestProfile(profile)
          setLatestProfileHistorical(historical)
        }}
      />
    </div>
  )
}

export default CNNTrain

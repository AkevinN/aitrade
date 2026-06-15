import React, { useState } from 'react'
import {
  Alert,
  Button,
  Col,
  Collapse,
  DatePicker,
  Empty,
  Form,
  Input,
  InputNumber,
  Row,
  Select,
  Space,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { FilterOutlined, InfoCircleOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'

import { cnnService } from '../../api/cnn'
import { useTask } from '../../hooks/useTask'
import type { CNNScreeningRequest, LeaderboardRow, ScreeningResult } from '../../types/screening'
import { metricMeta } from '../../utils/screening'

const { Text } = Typography

/**
 * 带 Tooltip 解释的指标列头标签。
 *
 * 复用 `metricMeta` 注册表，用 InfoCircleOutlined 图标作为可悬浮的视觉提示，
 * 与 ProfilingPanel 的 tooltip 呈现风格保持一致。
 *
 * @param metricKey - 指标键名，需在 `METRIC_META` 中注册
 * @returns 带标签文字和 Tooltip 图标的行内节点
 *
 * @example
 * ```tsx
 * <MetricLabel metricKey="fitness_score" />
 * // → "CNN 适配度 (Fitness Score)" + InfoCircle tooltip
 * ```
 */
function MetricLabel({ metricKey }: { metricKey: string }) {
  const meta = metricMeta(metricKey)
  return (
    <Space size={4}>
      <span>{meta.label}</span>
      <Tooltip title={meta.tooltip}>
        <InfoCircleOutlined style={{ color: '#8c8c8c', cursor: 'help' }} />
      </Tooltip>
    </Space>
  )
}

/** 综合置信度等级到 antd Tag 颜色的映射；insufficient 用 default（灰色）。 */
const confidenceColor: Record<string, string> = {
  high: 'green',
  medium: 'blue',
  low: 'orange',
  insufficient: 'default',
}

/**
 * 将置信度字符串映射为中文显示文案。
 *
 * @param level - 后端返回的置信度等级字符串
 * @returns 中文文案；未命中则原样返回
 */
function confidenceLabel(level: string): string {
  const map: Record<string, string> = {
    high: '高置信',
    medium: '中置信',
    low: '低置信',
    insufficient: '数据不足',
  }
  return map[level] ?? level
}

/**
 * 判断一行是否需要显示「未经 WF 实证」或「低置信」警示标注。
 *
 * - 未入围 Tier-2 或 Tier-2 不可评估 → "未经 WF 实证"
 * - overall_confidence 为 low/insufficient → "低置信"
 *
 * @param row - 榜单行
 * @returns 警示文案；无警示时返回 null
 */
function warningLabel(row: LeaderboardRow): string | null {
  const confidence = row.tier1.overall_confidence
  if (!row.promoted_to_tier2 || (row.tier2 && !row.tier2.evaluable)) {
    return '未经 WF 实证'
  }
  if (confidence === 'low' || confidence === 'insufficient') {
    return '低置信'
  }
  return null
}

/**
 * CNN 选股页面：配置候选股票池与漏斗参数 → 启动异步选股任务 → 可排序 Tier-1/Tier-2 榜单。
 *
 * 流程：
 * 1. 填写选股表单（K线周期、截止日期、回看天数、交易所/标的数量过滤、top_k、是否跑 Tier-2）
 * 2. 提交后调用 `cnnService.runScreening` 获取 task_id，`useTask` 轮询进度
 * 3. 任务完成后读取 `task.data.result` 渲染可排序榜单
 * 4. 每行提供「带入训练」按钮，通过 `location.state.preset` 跳转至 CNNTrain 页并预填标的
 *
 * 低置信或未入围 Tier-2 的行会显示橙色警示 Tag，提醒用户该结论未经实证。
 * `available=false` 的行（本地无数据）在置信度列显示"不可用"，榜单末尾排列。
 */
const CNNScreening: React.FC = () => {
  const [form] = Form.useForm()
  const [taskId, setTaskId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const navigate = useNavigate()

  const task = useTask(taskId)
  const screeningResult = task.data?.status === 'completed'
    ? (task.data.result as unknown as ScreeningResult)
    : null

  const isRunning =
    task.data?.status === 'running' || task.data?.status === 'pending'

  /**
   * 校验表单并启动 CNN 批量选股任务。
   *
   * 把 `as_of` DatePicker 值格式化为 YYYY-MM-DD，`exchange` 空选时传 null，
   * `include_symbols`/`exclude_symbols` 逗号/换行分割后去空；构建
   * `CNNScreeningRequest` 并调用 `cnnService.runScreening`，把返回的 task_id
   * 写入状态以驱动 `useTask` 轮询。
   */
  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      setSubmitting(true)

      const parseSymbolList = (raw?: string): string[] => {
        if (!raw) return []
        return raw
          .split(/[\n,]+/)
          .map((s) => s.trim())
          .filter(Boolean)
      }

      const req: CNNScreeningRequest = {
        name: values.name,
        interval: values.interval,
        as_of: (values.as_of as ReturnType<typeof dayjs>).format('YYYY-MM-DD'),
        lookback_days: values.lookback_days,
        exchange: values.exchange || null,
        min_bar_count: values.min_bar_count,
        include_symbols: parseSymbolList(values.include_symbols),
        exclude_symbols: parseSymbolList(values.exclude_symbols),
        top_k: values.top_k,
        run_tier2: values.run_tier2 ?? true,
        objective: values.objective,
        persist: false,
        // Tier-2 高级覆盖参数：仅在用户填写时传入，空（undefined/null）则后端使用规则默认值
        ...(values.eval_window_days != null ? { eval_window_days: values.eval_window_days } : {}),
        ...(values.train_days != null ? { train_days: values.train_days } : {}),
        ...(values.fold_test_days != null ? { fold_test_days: values.fold_test_days } : {}),
        ...(values.n_seeds != null ? { n_seeds: values.n_seeds } : {}),
      }

      const res = await cnnService.runScreening(req)
      setTaskId(res.task_id)
      message.success('CNN 选股任务已启动')
    } catch (err) {
      if (err instanceof Error) {
        message.error(err.message)
      }
    } finally {
      setSubmitting(false)
    }
  }

  /**
   * 跳转至 CNN 训练页并预填目标标的与 K 线周期。
   *
   * 通过 `location.state.preset` 传递预设值：CNNTrain 在 `useEffect` 中读取
   * `preset.target_symbol` / `preset.input_interval` 并写入训练表单，仅预填不提交。
   *
   * @param vt_symbol - 选股榜单中该行的标的代码
   * @param interval - 本次选股使用的 K 线周期
   */
  const handleGoTrain = (vt_symbol: string, interval: string) => {
    navigate('/cnn-train', {
      state: {
        preset: {
          target_symbol: vt_symbol,
          input_interval: interval,
          input_data_kind: 'bar',
        },
      },
    })
  }

  /** 榜单表格列定义，numeric 列附带 sorter 以支持客户端排序。 */
  const leaderboardColumns = [
    {
      title: <MetricLabel metricKey="rank" />,
      dataIndex: 'rank',
      width: 70,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => a.rank - b.rank,
    },
    {
      title: <MetricLabel metricKey="vt_symbol" />,
      key: 'vt_symbol',
      width: 140,
      render: (_: unknown, row: LeaderboardRow) => (
        <Space direction="vertical" size={2}>
          <Text strong>{row.tier1.vt_symbol}</Text>
          {!row.tier1.available ? <Tag color="default">不可用</Tag> : null}
        </Space>
      ),
    },
    {
      title: <MetricLabel metricKey="fitness_score" />,
      key: 'fitness_score',
      width: 180,
      defaultSortOrder: 'descend' as const,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => {
        const sa = a.tier1.fitness_score ?? -1
        const sb = b.tier1.fitness_score ?? -1
        return sa - sb
      },
      render: (_: unknown, row: LeaderboardRow) =>
        row.tier1.fitness_score != null ? row.tier1.fitness_score.toFixed(4) : '-',
    },
    {
      title: <MetricLabel metricKey="overall_confidence" />,
      key: 'overall_confidence',
      width: 180,
      render: (_: unknown, row: LeaderboardRow) => {
        const level = row.tier1.overall_confidence
        const warn = warningLabel(row)
        return (
          <Space size={4} wrap>
            <Tag color={confidenceColor[level] ?? 'default'}>
              {confidenceLabel(level)}
            </Tag>
            {warn ? <Tag color="orange">{warn}</Tag> : null}
          </Space>
        )
      },
    },
    {
      title: <MetricLabel metricKey="promoted_to_tier2" />,
      key: 'promoted_to_tier2',
      width: 160,
      render: (_: unknown, row: LeaderboardRow) =>
        row.promoted_to_tier2 ? (
          <Tag color="blue">已入围</Tag>
        ) : (
          <Tag color="default">未入围</Tag>
        ),
    },
    {
      title: <MetricLabel metricKey="edge_ok" />,
      key: 'edge_ok',
      width: 140,
      render: (_: unknown, row: LeaderboardRow) => {
        if (!row.tier2) return <Text type="secondary">-</Text>
        if (!row.tier2.evaluable) {
          // 已入围 Tier-2 但无法评估（通常是数据不足导致跳过）
          const skipLabel =
            row.tier2.note && row.tier2.note.includes('数据不足') ? '数据不足' : '跳过'
          return (
            <Tooltip title={row.tier2.note ?? '该标的 Tier-2 评估未能完成'}>
              <Tag color="orange">{skipLabel}</Tag>
            </Tooltip>
          )
        }
        return row.tier2.edge_ok ? (
          <Tag color="green">通过</Tag>
        ) : (
          <Tag color="red">未通过</Tag>
        )
      },
    },
    {
      title: <MetricLabel metricKey="avg_score" />,
      key: 'avg_score',
      width: 160,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => {
        const sa = a.tier2?.avg_score ?? -Infinity
        const sb = b.tier2?.avg_score ?? -Infinity
        return sa - sb
      },
      render: (_: unknown, row: LeaderboardRow) =>
        row.tier2?.avg_score != null ? row.tier2.avg_score.toFixed(4) : '-',
    },
    {
      title: <MetricLabel metricKey="pos_fold_ratio" />,
      key: 'pos_fold_ratio',
      width: 165,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => {
        const sa = a.tier2?.pos_fold_ratio ?? -Infinity
        const sb = b.tier2?.pos_fold_ratio ?? -Infinity
        return sa - sb
      },
      render: (_: unknown, row: LeaderboardRow) =>
        row.tier2?.pos_fold_ratio != null
          ? `${(row.tier2.pos_fold_ratio * 100).toFixed(1)}%`
          : '-',
    },
    {
      title: '操作',
      key: 'actions',
      width: 110,
      render: (_: unknown, row: LeaderboardRow) => {
        const interval =
          screeningResult?.input?.interval != null
            ? String(screeningResult.input.interval)
            : 'd'
        return (
          <Tooltip title="在 CNN 训练页预填该标的与周期，仅预填不提交">
            <Button
              size="small"
              disabled={!row.tier1.available}
              onClick={() => handleGoTrain(row.tier1.vt_symbol, interval)}
            >
              带入训练
            </Button>
          </Tooltip>
        )
      },
    },
  ]

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="CNN 选股"
        description="配置候选股票池与漏斗参数，批量评估 CNN 适配度（Tier-1 综合打分 + 可选 Tier-2 WF/OOS 实证），产出排名榜单后可一键带入 CNN 训练。选股结论恒为草稿，不自动开训、不下真实单。"
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={8}>
          <ScreeningForm
            form={form}
            onSubmit={handleSubmit}
            submitting={submitting}
            isRunning={isRunning}
            taskMessage={task.data?.message}
            taskStatus={task.data?.status}
            taskProgress={task.data?.progress}
          />
        </Col>

        <Col xs={24} xl={16}>
          <ScreeningResultPanel
            result={screeningResult}
            leaderboardColumns={leaderboardColumns}
          />
        </Col>
      </Row>
    </Space>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

// Tier-2 默认超参（与后端 ScreeningRules 默认值对齐，用于 derived history hint 回落）
const TIER2_DEFAULTS = {
  eval_window_days: 900,
  train_days: 480,
  fold_test_days: 90,
} as const

/**
 * 根据 Tier-2 窗口参数推算所需历史天数与估计折数。
 *
 * 折数公式（近似，step ≈ fold_test_days）：
 *   folds ≈ floor((evalWindow - train - test) / test) + 1
 *
 * @param evalWindow - 评估窗总长（天），缺省 900
 * @param train - 每折训练天数，缺省 480
 * @param test - 每折测试天数，缺省 90
 * @returns `{ evalWindow, neededMin, folds }`
 */
function calcHistoryHint(
  evalWindow: number | undefined,
  train: number | undefined,
  test: number | undefined,
): { evalWindow: number; neededMin: number; folds: number } {
  const ew = evalWindow ?? TIER2_DEFAULTS.eval_window_days
  const tr = train ?? TIER2_DEFAULTS.train_days
  const te = test ?? TIER2_DEFAULTS.fold_test_days
  const neededMin = tr + te
  const folds = Math.max(1, Math.floor((ew - tr - te) / te) + 1)
  return { evalWindow: ew, neededMin, folds }
}

/**
 * 选股配置表单：K 线周期、截止日期、回看天数、交易所/数量过滤、
 * 显式候选池与排除清单、漏斗参数（top_k、run_tier2）、Tier-2 目标函数，
 * 以及可折叠的「Tier-2 高级设置」面板（eval_window_days / train_days /
 * fold_test_days / n_seeds），含实时历史需求提示。
 *
 * 提交后展示任务运行进度条（来自 `useTask` 轮询）。
 */
function ScreeningForm({
  form,
  onSubmit,
  submitting,
  isRunning,
  taskMessage,
  taskStatus,
  taskProgress,
}: {
  /** 受控的 antd FormInstance；由父组件持有以便提交时读取字段值 */
  form: any
  /** 点击「启动选股」时触发 */
  onSubmit: () => void
  /** 是否正在提交（按钮 loading 状态） */
  submitting: boolean
  /** 是否有任务正在运行（轮询未到终态） */
  isRunning: boolean
  /** 当前任务状态消息（来自 useTask） */
  taskMessage?: string
  /** 当前任务状态（completed/failed/running/pending） */
  taskStatus?: string
  /** 当前任务进度百分比（0-100，来自 Task.progress） */
  taskProgress?: number
}) {
  // 监听 run_tier2 开关与 Tier-2 高级字段，用于条件渲染和历史需求实时计算
  const runTier2 = Form.useWatch('run_tier2', form)
  const evalWindowDays = Form.useWatch('eval_window_days', form)
  const trainDays = Form.useWatch('train_days', form)
  const foldTestDays = Form.useWatch('fold_test_days', form)

  const hint = calcHistoryHint(evalWindowDays, trainDays, foldTestDays)

  return (
    <section className="panel">
      <Form
        layout="vertical"
        form={form}
        initialValues={{
          name: `cnn_screen_${dayjs().format('MMDDHHmm')}`,
          interval: 'd',
          as_of: dayjs(),
          lookback_days: 250,
          exchange: '',
          min_bar_count: 250,
          include_symbols: '',
          exclude_symbols: '',
          top_k: 15,
          run_tier2: true,
          objective: 'classification',
        }}
      >
        <Form.Item label="任务名称" name="name" rules={[{ required: true, message: '请填写任务名称' }]}>
          <Input placeholder="cnn_screen_YYYYMMDD" />
        </Form.Item>

        <Row gutter={8}>
          <Col span={12}>
            <Form.Item label="K 线周期" name="interval" rules={[{ required: true }]}>
              <Select
                options={['d', '60m', '30m', '15m', '10m', '5m', '1m'].map((v) => ({
                  value: v,
                  label: v,
                }))}
              />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="截止日期"
              name="as_of"
              rules={[{ required: true, message: '请选择截止日期' }]}
              tooltip="选股数据严格不超过此日期（时间窗口隔离红线）"
            >
              <DatePicker style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="回看天数"
          name="lookback_days"
          rules={[{ required: true }]}
          tooltip="Tier-1 画像回看的日历天数，建议 ≥ 250"
        >
          <InputNumber min={1} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          label="交易所过滤"
          name="exchange"
          tooltip="不选则不过滤交易所（全部）；选中时仅保留对应交易所标的"
        >
          <Select
            allowClear
            placeholder="全部（不过滤）"
            options={[
              { value: 'SSE', label: '上交所（SSE）' },
              { value: 'SZSE', label: '深交所（SZSE）' },
              { value: 'BSE', label: '北交所（BSE）' },
            ]}
          />
        </Form.Item>

        <Form.Item
          label="最小历史 bar 数"
          name="min_bar_count"
          tooltip="本地存档低于此值的标的将被排除出候选池"
        >
          <InputNumber min={1} style={{ width: '100%' }} />
        </Form.Item>

        <Form.Item
          label="显式候选池（可选）"
          name="include_symbols"
          tooltip="非空时以此清单为 universe（忽略交易所过滤）；逗号或换行分隔"
        >
          <Input.TextArea
            rows={3}
            placeholder="600030.SSE&#10;000001.SZSE&#10;…（可留空，留空则扫描全部本地数据）"
          />
        </Form.Item>

        <Form.Item
          label="强制排除清单（可选）"
          name="exclude_symbols"
          tooltip="从最终 universe 中剔除；逗号或换行分隔"
        >
          <Input.TextArea rows={2} placeholder="000001.SZSE&#10;…（可留空）" />
        </Form.Item>

        <Row gutter={8}>
          <Col span={12}>
            <Form.Item
              label="Top-K（入围 Tier-2）"
              name="top_k"
              tooltip="Tier-1 排名前 K 只进入 Tier-2 实证"
            >
              <InputNumber min={1} max={100} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="执行 Tier-2 WF/OOS"
              name="run_tier2"
              valuePropName="checked"
              tooltip="关闭时只产出 Tier-1 排名榜单，速度更快"
            >
              <Switch />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="Tier-2 目标函数"
          name="objective"
          tooltip="供 Tier-2 WF/OOS 构造训练请求使用；建议与后续实际训练保持一致"
        >
          <Select
            options={[
              { value: 'classification', label: '方向分类' },
              { value: 'regression', label: '收益回归' },
              { value: 'path_class', label: '路径形态分类' },
            ]}
          />
        </Form.Item>

        {/* Tier-2 高级设置：仅在 run_tier2 开启时显示，默认折叠 */}
        {runTier2 && (
          <Collapse
            size="small"
            style={{ marginBottom: 16 }}
            items={[
              {
                key: 'tier2-advanced',
                label: 'Tier-2 高级设置（不填用默认）',
                children: (
                  <Space direction="vertical" size={0} style={{ width: '100%' }}>
                    <Row gutter={8}>
                      <Col span={12}>
                        <Form.Item
                          label={<MetricLabel metricKey="eval_window_days" />}
                          name="eval_window_days"
                          tooltip={null}
                        >
                          <InputNumber
                            min={30}
                            style={{ width: '100%' }}
                            placeholder="默认 900"
                          />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item
                          label={<MetricLabel metricKey="train_days" />}
                          name="train_days"
                          tooltip={null}
                        >
                          <InputNumber
                            min={30}
                            style={{ width: '100%' }}
                            placeholder="默认 480"
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                    <Row gutter={8}>
                      <Col span={12}>
                        <Form.Item
                          label={<MetricLabel metricKey="fold_test_days" />}
                          name="fold_test_days"
                          tooltip={null}
                        >
                          <InputNumber
                            min={7}
                            style={{ width: '100%' }}
                            placeholder="默认 90"
                          />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item
                          label={<MetricLabel metricKey="n_seeds" />}
                          name="n_seeds"
                          tooltip={null}
                        >
                          <InputNumber
                            min={1}
                            max={10}
                            style={{ width: '100%' }}
                            placeholder="默认 1"
                          />
                        </Form.Item>
                      </Col>
                    </Row>
                    {/* 实时历史需求提示 */}
                    <Alert
                      type="info"
                      showIcon
                      style={{ marginTop: 4, fontSize: 12 }}
                      message={
                        `本配置约需 ${hint.evalWindow} 天历史` +
                        `（≥${hint.neededMin} 天才能跑 1 折）` +
                        ` → 约 ${hint.folds} 折。` +
                        `历史不足的标的会自动跳过 Tier-2。`
                      }
                    />
                  </Space>
                ),
              },
            ]}
          />
        )}

        <Button
          type="primary"
          block
          loading={submitting || isRunning}
          onClick={onSubmit}
        >
          {isRunning ? '选股任务运行中…' : '启动选股'}
        </Button>
      </Form>

      {taskStatus && (
        <Alert
          style={{ marginTop: 12 }}
          type={
            taskStatus === 'failed'
              ? 'error'
              : taskStatus === 'completed'
                ? 'success'
                : 'info'
          }
          showIcon
          message={
            taskStatus === 'completed'
              ? '选股任务已完成'
              : taskStatus === 'failed'
                ? '选股任务失败'
                : `运行中… ${taskProgress != null ? `${taskProgress}%` : ''}`
          }
          description={taskMessage}
        />
      )}
    </section>
  )
}

/**
 * 将贡献维度名渲染为带 Tooltip 的标签节点。
 *
 * 优先从 `METRIC_META` 读取标准标签；未注册的维度名原样显示加 InfoCircle 兜底说明。
 *
 * @param dimension - `ScoreContribution.dimension` 的值
 * @returns 带中英标签 + Tooltip 图标的 Space 节点
 */
function DimensionLabel({ dimension }: { dimension: string }) {
  return <MetricLabel metricKey={dimension} />
}

/** 置信度贡献明细表列，供展开行渲染。 */
const contributionColumns = [
  {
    title: <MetricLabel metricKey="dimension" />,
    dataIndex: 'dimension',
    width: 200,
    render: (dim: string) => <DimensionLabel dimension={dim} />,
  },
  {
    title: <MetricLabel metricKey="raw_value" />,
    dataIndex: 'raw_value',
    width: 110,
    render: (v: number | null) => (v != null ? v.toFixed(4) : '-'),
  },
  {
    title: <MetricLabel metricKey="level" />,
    dataIndex: 'level',
    width: 100,
    render: (v: string | null) => v ?? '-',
  },
  {
    title: <MetricLabel metricKey="weight" />,
    dataIndex: 'weight',
    width: 90,
    render: (v: number) => v.toFixed(2),
  },
  {
    title: <MetricLabel metricKey="contribution" />,
    dataIndex: 'contribution',
    width: 120,
    render: (v: number) => v.toFixed(4),
  },
  {
    title: <MetricLabel metricKey="confidence" />,
    dataIndex: 'confidence',
    width: 120,
  },
]

/**
 * 选股结果面板：展示运行摘要（标的池大小、排除列表）与可排序的 Tier-1/Tier-2 榜单。
 *
 * 展开行展示 `tier1.contributions` 逐维贡献明细；
 * `result=null`（任务尚未完成）时展示占位提示；
 * `available=false` 且空置信度的行提示数据不可用。
 *
 * @param result - 完成的选股结果；null 时展示占位提示
 * @param leaderboardColumns - 父组件传入的表格列定义（含 handler 闭包）
 */
function ScreeningResultPanel({
  result,
  leaderboardColumns,
}: {
  /** 选股结果；null 时展示占位提示 */
  result: ScreeningResult | null
  /** 榜单表格列定义（由父组件构建，携带 navigate 等闭包） */
  leaderboardColumns: any[]
}) {
  if (!result) {
    return (
      <section className="panel">
        <Empty
          image={<FilterOutlined style={{ fontSize: 40, color: '#444' }} />}
          description="配置左侧参数并点击「启动选股」，结果将在此显示。"
        />
      </section>
    )
  }

  const { universe_size, excluded, leaderboard } = result

  return (
    <section className="panel">
      <Space direction="vertical" size={12} style={{ width: '100%' }}>
        <Space wrap>
          <Tag color="blue">候选池 {universe_size} 只</Tag>
          <Tag color="orange">排除 {excluded.length} 只</Tag>
          <Tag color="default">草稿结论</Tag>
          {result.effective_right_bound ? (
            <Tag color="default">数据右边界 {String(result.effective_right_bound).slice(0, 10)}</Tag>
          ) : null}
        </Space>

        {excluded.length > 0 ? (
          <Alert
            type="warning"
            showIcon
            message={`${excluded.length} 只标的被排除（本地数据不足或被手动排除）`}
          />
        ) : null}

        <Table<LeaderboardRow>
          size="small"
          rowKey={(row) => row.tier1.vt_symbol}
          dataSource={leaderboard}
          columns={leaderboardColumns}
          scroll={{ x: 900 }}
          pagination={{ pageSize: 20, showSizeChanger: false }}
          expandable={{
            expandedRowRender: (row) => (
              <Space direction="vertical" size={4} style={{ width: '100%' }}>
                {row.tier1.note ? (
                  <Alert type="warning" showIcon message={row.tier1.note} style={{ marginBottom: 8 }} />
                ) : null}
                {row.tier2?.note ? (
                  <Alert type="info" showIcon message={`Tier-2：${row.tier2.note}`} style={{ marginBottom: 8 }} />
                ) : null}
                <Table
                  size="small"
                  rowKey="dimension"
                  dataSource={row.tier1.contributions}
                  columns={contributionColumns}
                  pagination={false}
                />
              </Space>
            ),
            rowExpandable: (row) => row.tier1.contributions.length > 0 || Boolean(row.tier1.note),
          }}
        />
      </Space>
    </section>
  )
}

export default CNNScreening

import React, { useState } from 'react'
import {
  Alert,
  Button,
  Col,
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
import { FilterOutlined } from '@ant-design/icons'
import dayjs from 'dayjs'
import { useNavigate } from 'react-router-dom'

import { cnnService } from '../../api/cnn'
import { useTask } from '../../hooks/useTask'
import type { CNNScreeningRequest, LeaderboardRow, ScreeningResult } from '../../types/screening'

const { Text } = Typography

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
      title: '排名',
      dataIndex: 'rank',
      width: 60,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => a.rank - b.rank,
    },
    {
      title: '标的',
      key: 'vt_symbol',
      width: 130,
      render: (_: unknown, row: LeaderboardRow) => (
        <Space direction="vertical" size={2}>
          <Text strong>{row.tier1.vt_symbol}</Text>
          {!row.tier1.available ? <Tag color="default">不可用</Tag> : null}
        </Space>
      ),
    },
    {
      title: 'Fitness Score',
      key: 'fitness_score',
      width: 130,
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
      title: '综合置信度',
      key: 'overall_confidence',
      width: 120,
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
      title: '入围 Tier-2',
      key: 'promoted_to_tier2',
      width: 100,
      render: (_: unknown, row: LeaderboardRow) =>
        row.promoted_to_tier2 ? (
          <Tag color="blue">已入围</Tag>
        ) : (
          <Tag color="default">未入围</Tag>
        ),
    },
    {
      title: 'Edge OK',
      key: 'edge_ok',
      width: 90,
      render: (_: unknown, row: LeaderboardRow) => {
        if (!row.tier2) return <Text type="secondary">-</Text>
        if (!row.tier2.evaluable) return <Tag color="default">不可评估</Tag>
        return row.tier2.edge_ok ? (
          <Tag color="green">通过</Tag>
        ) : (
          <Tag color="red">未通过</Tag>
        )
      },
    },
    {
      title: 'Avg Score',
      key: 'avg_score',
      width: 110,
      sorter: (a: LeaderboardRow, b: LeaderboardRow) => {
        const sa = a.tier2?.avg_score ?? -Infinity
        const sb = b.tier2?.avg_score ?? -Infinity
        return sa - sb
      },
      render: (_: unknown, row: LeaderboardRow) =>
        row.tier2?.avg_score != null ? row.tier2.avg_score.toFixed(4) : '-',
    },
    {
      title: '正折占比',
      key: 'pos_fold_ratio',
      width: 100,
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

/**
 * 选股配置表单：K 线周期、截止日期、回看天数、交易所/数量过滤、
 * 显式候选池与排除清单、漏斗参数（top_k、run_tier2）以及 Tier-2 目标函数。
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

/** 置信度贡献明细表列，供展开行渲染。 */
const contributionColumns = [
  { title: '维度', dataIndex: 'dimension', width: 150 },
  {
    title: '原始值',
    dataIndex: 'raw_value',
    width: 90,
    render: (v: number | null) => (v != null ? v.toFixed(4) : '-'),
  },
  { title: '等级', dataIndex: 'level', width: 80, render: (v: string | null) => v ?? '-' },
  {
    title: '权重',
    dataIndex: 'weight',
    width: 70,
    render: (v: number) => v.toFixed(2),
  },
  {
    title: '贡献量',
    dataIndex: 'contribution',
    width: 80,
    render: (v: number) => v.toFixed(4),
  },
  { title: '置信度', dataIndex: 'confidence', width: 90 },
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

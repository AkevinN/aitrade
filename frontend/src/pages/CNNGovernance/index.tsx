import React, { useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Col,
  DatePicker,
  Descriptions,
  Form,
  Input,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import { SafetyCertificateOutlined, SyncOutlined, SwapOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { governanceService } from '../../api/governance'
import { useTask } from '../../hooks/useTask'
import type {
  CNNCandidate,
  CNNCandidateTrainRequest,
  CNNGovernanceReplayRequest,
  CNNGovernanceReport,
  CNNPromotionGate,
  CNNTrainingParams,
  CNNBacktestParams,
} from '../../types/governance'

const { RangePicker } = DatePicker
const { Text } = Typography

const DEFAULT_TRAINING: CNNTrainingParams = {
  epochs: 20,
  batch_size: 32,
  learning_rate: 0.001,
  lookback: 30,
  dropout: 0.4,
  train_ratio: 0.8,
  loss_weighting: 'none',
}

const DEFAULT_BACKTEST: CNNBacktestParams = {
  buy_threshold: 0.6,
  sell_threshold: 0.4,
  commission_rate: 0.0003,
  stamp_duty: 0.001,
  slippage: 0.0005,
  price_add: 0.002,
  exit_mode: 'auto',
  hold_days: 1,
  take_profit: 0,
  stop_loss: 0,
  t_plus1: false,
}

const DEFAULT_GATE: CNNPromotionGate = {
  min_win_rate: 0.5,
  min_core_score_delta: 0,
  max_drawdown_worsen_pct: 10,
  require_positive_oos: true,
}

function baseRequest(values: Record<string, any>): CNNCandidateTrainRequest {
  const [start, end] = values.range
  return {
    name: values.name,
    target_symbol: values.target_symbol,
    input_data_kind: values.input_data_kind || 'bar',
    input_interval: values.input_interval || 'd',
    start: start.format('YYYY-MM-DD'),
    end: end.format('YYYY-MM-DD'),
    train_days: values.train_days,
    test_days: values.test_days,
    step_days: values.step_days,
    objective: values.objective || 'classification',
    label_spec: {
      mode: values.label_mode || 'next_bar',
      horizon: values.label_mode === 'horizon_bars' ? values.label_horizon : undefined,
      threshold: Math.max(0, (values.label_threshold_pct || 0) / 100),
      neutral_policy: values.neutral_policy || 'drop',
      price_ref: values.price_ref || 'next_open',
    },
    observation_groups: [],
    training_params: {
      ...DEFAULT_TRAINING,
      epochs: values.epochs,
      lookback: values.lookback,
      loss_weighting: values.objective === 'classification' ? values.loss_weighting : 'none',
    },
    backtest_params: {
      ...DEFAULT_BACKTEST,
      buy_threshold: values.objective === 'regression' ? values.reg_buy_threshold : values.buy_threshold,
      sell_threshold: values.objective === 'regression' ? values.reg_sell_threshold : values.sell_threshold,
    },
    promotion_gate: {
      ...DEFAULT_GATE,
      min_win_rate: values.min_win_rate,
      min_core_score_delta: values.min_core_score_delta,
      require_positive_oos: values.require_positive_oos,
    },
  }
}

const statusColor: Record<string, string> = {
  passed: 'green',
  failed: 'red',
  promoted: 'blue',
  rejected: 'default',
  pending: 'gold',
}

const CNNGovernance: React.FC = () => {
  const [form] = Form.useForm()
  const [replayForm] = Form.useForm()
  const [taskId, setTaskId] = useState<string | null>(null)
  const [selectedReportId, setSelectedReportId] = useState<string>('')
  const [selectedReplayId, setSelectedReplayId] = useState<string>('')
  const queryClient = useQueryClient()
  const task = useTask(taskId)

  const { data: config } = useQuery({
    queryKey: ['cnn-governance-config'],
    queryFn: governanceService.getConfig,
  })
  const { data: production } = useQuery({
    queryKey: ['cnn-governance-production'],
    queryFn: governanceService.getProduction,
  })
  const { data: candidates } = useQuery({
    queryKey: ['cnn-governance-candidates'],
    queryFn: governanceService.listCandidates,
  })
  const { data: history } = useQuery({
    queryKey: ['cnn-governance-history'],
    queryFn: governanceService.listHistory,
  })
  const { data: replays } = useQuery({
    queryKey: ['cnn-governance-replays'],
    queryFn: governanceService.listReplays,
  })
  const { data: report } = useQuery({
    queryKey: ['cnn-governance-report', selectedReportId],
    queryFn: () => governanceService.getReport(selectedReportId),
    enabled: Boolean(selectedReportId),
  })
  const { data: replay } = useQuery({
    queryKey: ['cnn-governance-replay', selectedReplayId],
    queryFn: () => governanceService.getReplay(selectedReplayId),
    enabled: Boolean(selectedReplayId),
  })

  const refreshGovernance = () => {
    void queryClient.invalidateQueries({ queryKey: ['cnn-governance-production'] })
    void queryClient.invalidateQueries({ queryKey: ['cnn-governance-candidates'] })
    void queryClient.invalidateQueries({ queryKey: ['cnn-governance-history'] })
    void queryClient.invalidateQueries({ queryKey: ['cnn-governance-replays'] })
  }

  const updateConfig = useMutation({
    mutationFn: governanceService.updateConfig,
    onSuccess: () => {
      message.success('治理配置已保存')
      void queryClient.invalidateQueries({ queryKey: ['cnn-governance-config'] })
    },
  })

  const promote = useMutation({
    mutationFn: (candidateId: string) => governanceService.promoteCandidate(candidateId, '前端人工晋级'),
    onSuccess: () => {
      message.success('候选已晋级为生产模型')
      refreshGovernance()
    },
  })

  const reject = useMutation({
    mutationFn: (candidateId: string) => governanceService.rejectCandidate(candidateId, '前端人工拒绝'),
    onSuccess: () => {
      message.success('候选已拒绝')
      refreshGovernance()
    },
  })

  const rollback = useMutation({
    mutationFn: () => governanceService.rollback('前端人工回滚'),
    onSuccess: () => {
      message.success('生产模型已回滚')
      refreshGovernance()
    },
  })

  const currentResult = task.data?.result as Record<string, unknown> | undefined
  const taskHint = taskId && task.data ? `${task.data.status} · ${task.data.message}` : '无运行中的治理任务'

  const latestReport = useMemo(() => {
    if (currentResult?.report_id) return currentResult as unknown as CNNGovernanceReport
    return report
  }, [currentResult, report])

  const startEvaluate = async () => {
    const values = await form.validateFields()
    const req = baseRequest(values)
    const res = await governanceService.evaluate(req)
    setTaskId(res.task_id)
    message.success('WF/OOS 评估已启动')
  }

  const startCandidate = async () => {
    const values = await form.validateFields()
    const req = baseRequest(values)
    const res = await governanceService.trainCandidate(req)
    setTaskId(res.task_id)
    message.success('候选模型训练已启动')
  }

  const startReplay = async () => {
    const values = await replayForm.validateFields()
    const [start, end] = values.range
    const req: CNNGovernanceReplayRequest = {
      ...baseRequest(values),
      start: start.format('YYYY-MM-DD'),
      end: end.format('YYYY-MM-DD'),
      initial_train_days: values.initial_train_days,
      evaluation_period_days: values.evaluation_period_days,
      test_period_days: values.test_period_days,
      capital: values.capital,
      baselines: values.baselines,
    }
    const res = await governanceService.runReplay(req)
    setTaskId(res.task_id)
    message.success('治理回放回测已启动')
  }

  return (
    <Space direction="vertical" size="middle" style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message="CNN 模型治理"
        description="用 WF/OOS、候选模型、人工晋级、回滚和治理回放回测管理模型更新，避免新数据无门禁地污染生产模型。"
      />

      <Row gutter={[16, 16]}>
        <Col xs={24} xl={8}>
          <section className="panel">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong><SafetyCertificateOutlined /> 当前生产模型</Text>
              <Descriptions size="small" bordered column={1}>
                <Descriptions.Item label="模型">{production?.model_name || '未设置'}</Descriptions.Item>
                <Descriptions.Item label="版本">{production?.model_version || '-'}</Descriptions.Item>
                <Descriptions.Item label="目标">{production?.target_symbol || '-'}</Descriptions.Item>
                <Descriptions.Item label="周期">{production?.input_interval || '-'}</Descriptions.Item>
                <Descriptions.Item label="目标函数">{production?.objective || '-'}</Descriptions.Item>
                <Descriptions.Item label="晋级时间">{production?.promoted_at || '-'}</Descriptions.Item>
              </Descriptions>
              <Button
                danger
                block
                disabled={!production?.previous_model_name}
                onClick={() => {
                  Modal.confirm({
                    title: '确认回滚生产模型？',
                    content: `将回滚到 ${production?.previous_model_name || '上一版本'}。`,
                    onOk: () => rollback.mutate(),
                  })
                }}
              >
                回滚到上一生产模型
              </Button>
            </Space>
          </section>
        </Col>

        <Col xs={24} xl={8}>
          <section className="panel">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong><SyncOutlined /> 治理配置</Text>
              <Descriptions size="small" bordered column={1}>
                <Descriptions.Item label="启用">{config?.enabled ? '是' : '否'}</Descriptions.Item>
                <Descriptions.Item label="周期">{config?.evaluation_period_days ?? 30} 天</Descriptions.Item>
                <Descriptions.Item label="训练窗">{config?.train_days ?? 720} 天</Descriptions.Item>
                <Descriptions.Item label="OOS窗">{config?.test_days ?? 90} 天</Descriptions.Item>
                <Descriptions.Item label="自动晋级">{config?.auto_promote ? '是' : '否'}</Descriptions.Item>
              </Descriptions>
              <Button
                block
                onClick={() =>
                  config &&
                  updateConfig.mutate({
                    ...config,
                    enabled: !config.enabled,
                  })
                }
              >
                {config?.enabled ? '暂停治理提示' : '启用治理提示'}
              </Button>
            </Space>
          </section>
        </Col>

        <Col xs={24} xl={8}>
          <section className="panel">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong><SwapOutlined /> 任务状态</Text>
              <Alert type={task.data?.status === 'failed' ? 'error' : 'info'} message={taskHint} />
              {currentResult?.candidate_id ? (
                <Text type="secondary">候选：{String(currentResult.candidate_id)}</Text>
              ) : null}
              {currentResult?.replay_id ? (
                <Button onClick={() => setSelectedReplayId(String(currentResult.replay_id))}>
                  查看本次回放报告
                </Button>
              ) : null}
            </Space>
          </section>
        </Col>
      </Row>

      <Tabs
        items={[
          {
            key: 'candidate',
            label: '评估与候选',
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} xl={8}>
                  <GovernanceForm form={form} onEvaluate={startEvaluate} onCandidate={startCandidate} />
                </Col>
                <Col xs={24} xl={16}>
                  <CandidateTable
                    candidates={candidates || []}
                    onReport={setSelectedReportId}
                    onPromote={(id) => {
                      Modal.confirm({
                        title: '确认晋级候选模型？',
                        content: '晋级后 Trading Console 将默认使用该生产模型版本。',
                        onOk: () => promote.mutate(id),
                      })
                    }}
                    onReject={(id) => reject.mutate(id)}
                  />
                  <ReportPanel report={latestReport} />
                </Col>
              </Row>
            ),
          },
          {
            key: 'replay',
            label: '治理回放回测',
            children: (
              <Row gutter={[16, 16]}>
                <Col xs={24} xl={8}>
                  <ReplayForm form={replayForm} onReplay={startReplay} />
                </Col>
                <Col xs={24} xl={16}>
                  <ReplayTable replays={replays || []} onOpen={setSelectedReplayId} />
                  <ReplayPanel replay={replay} />
                </Col>
              </Row>
            ),
          },
          {
            key: 'history',
            label: '历史',
            children: (
              <Table
                size="small"
                rowKey={(row) => `${row.ts}-${row.event_type}`}
                dataSource={history || []}
                columns={[
                  { title: '时间', dataIndex: 'ts', width: 190 },
                  { title: '事件', dataIndex: 'event_type', width: 220 },
                  { title: '摘要', render: (_, row) => JSON.stringify(row.payload).slice(0, 220) },
                ]}
              />
            ),
          },
        ]}
      />
    </Space>
  )
}

function GovernanceForm({ form, onEvaluate, onCandidate }: { form: any; onEvaluate: () => void; onCandidate: () => void }) {
  return (
    <section className="panel">
      <Form
        layout="vertical"
        form={form}
        initialValues={{
          name: `cnn_gov_${dayjs().format('MMDDHHmm')}`,
          target_symbol: '',
          input_data_kind: 'bar',
          input_interval: 'd',
          range: [dayjs().subtract(3, 'year'), dayjs()],
          train_days: 720,
          test_days: 90,
          step_days: 90,
          objective: 'classification',
          label_mode: 'next_bar',
          label_horizon: 3,
          label_threshold_pct: 0.5,
          neutral_policy: 'drop',
          price_ref: 'next_open',
          epochs: 20,
          lookback: 30,
          loss_weighting: 'none',
          buy_threshold: 0.6,
          sell_threshold: 0.4,
          reg_buy_threshold: 0.005,
          reg_sell_threshold: -0.005,
          min_win_rate: 0.5,
          min_core_score_delta: 0,
          require_positive_oos: true,
        }}
      >
        <Form.Item label="名称" name="name" rules={[{ required: true }]}>
          <Input />
        </Form.Item>
        <Form.Item label="目标标的" name="target_symbol" rules={[{ required: true }]}>
          <Input placeholder="000001.SZSE" />
        </Form.Item>
        <Row gutter={8}>
          <Col span={12}>
            <Form.Item label="数据类型" name="input_data_kind">
              <Select options={[{ value: 'bar', label: 'K线' }, { value: 'tick', label: 'Tick' }]} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="周期" name="input_interval">
              <Select options={['d', '1m', '5m', '10m', '15m', '30m', '60m'].map((v) => ({ value: v, label: v }))} />
            </Form.Item>
          </Col>
        </Row>
        <Form.Item label="评估区间" name="range" rules={[{ required: true }]}>
          <RangePicker style={{ width: '100%' }} />
        </Form.Item>
        <Row gutter={8}>
          <Col span={8}><Form.Item label="训练窗" name="train_days"><InputNumber min={30} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={8}><Form.Item label="OOS窗" name="test_days"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={8}><Form.Item label="步长" name="step_days"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
        </Row>
        <Row gutter={8}>
          <Col span={12}>
            <Form.Item label="目标函数" name="objective">
              <Select options={[{ value: 'classification', label: '方向分类' }, { value: 'regression', label: '收益回归' }]} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="Label" name="label_mode">
              <Select options={[{ value: 'next_bar', label: 'next_bar' }, { value: 'horizon_bars', label: 'horizon_bars' }]} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={8}>
          <Col span={12}><Form.Item label="Epochs" name="epochs"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={12}><Form.Item label="Lookback" name="lookback"><InputNumber min={2} style={{ width: '100%' }} /></Form.Item></Col>
        </Row>
        <Row gutter={8}>
          <Col span={12}><Form.Item label="分类买入阈值" name="buy_threshold"><InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={12}><Form.Item label="分类卖出阈值" name="sell_threshold"><InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} /></Form.Item></Col>
        </Row>
        <Row gutter={8}>
          <Col span={12}><Form.Item label="胜出折比例" name="min_win_rate"><InputNumber min={0} max={1} step={0.05} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={12}><Form.Item label="OOS为正" name="require_positive_oos" valuePropName="checked"><Switch /></Form.Item></Col>
        </Row>
        <Space wrap>
          <Button onClick={onEvaluate}>启动 WF 评估</Button>
          <Button type="primary" onClick={onCandidate}>训练候选模型</Button>
        </Space>
      </Form>
    </section>
  )
}

function ReplayForm({ form, onReplay }: { form: any; onReplay: () => void }) {
  return (
    <section className="panel">
      <Form
        layout="vertical"
        form={form}
        initialValues={{
          name: `cnn_replay_${dayjs().format('MMDDHHmm')}`,
          target_symbol: '',
          input_data_kind: 'bar',
          input_interval: 'd',
          range: [dayjs().subtract(3, 'year'), dayjs()],
          initial_train_days: 720,
          evaluation_period_days: 30,
          test_period_days: 30,
          capital: 1000000,
          objective: 'classification',
          label_mode: 'next_bar',
          label_horizon: 3,
          label_threshold_pct: 0.5,
          neutral_policy: 'drop',
          price_ref: 'next_open',
          epochs: 10,
          lookback: 30,
          loss_weighting: 'none',
          buy_threshold: 0.6,
          sell_threshold: 0.4,
          reg_buy_threshold: 0.005,
          reg_sell_threshold: -0.005,
          train_days: 720,
          test_days: 30,
          step_days: 30,
          min_win_rate: 0.5,
          min_core_score_delta: 0,
          require_positive_oos: true,
          baselines: ['fixed_initial_model', 'always_retrain', 'governed_promotion', 'buy_and_hold'],
        }}
      >
        <Form.Item label="回放名称" name="name" rules={[{ required: true }]}><Input /></Form.Item>
        <Form.Item label="目标标的" name="target_symbol" rules={[{ required: true }]}><Input placeholder="000001.SZSE" /></Form.Item>
        <Form.Item label="回放区间" name="range" rules={[{ required: true }]}><RangePicker style={{ width: '100%' }} /></Form.Item>
        <Row gutter={8}>
          <Col span={8}><Form.Item label="初始训练窗" name="initial_train_days"><InputNumber min={30} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={8}><Form.Item label="评估周期" name="evaluation_period_days"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
          <Col span={8}><Form.Item label="交易窗" name="test_period_days"><InputNumber min={1} style={{ width: '100%' }} /></Form.Item></Col>
        </Row>
        <Form.Item label="对照组" name="baselines">
          <Select
            mode="multiple"
            options={[
              { value: 'fixed_initial_model', label: '固定初始模型' },
              { value: 'always_retrain', label: '无脑重训' },
              { value: 'governed_promotion', label: '治理晋级' },
              { value: 'buy_and_hold', label: '买入持有' },
            ]}
          />
        </Form.Item>
        <Form.Item label="初始资金" name="capital"><InputNumber min={10000} style={{ width: '100%' }} /></Form.Item>
        <Button type="primary" block onClick={onReplay}>启动治理回放回测</Button>
      </Form>
    </section>
  )
}

function CandidateTable({
  candidates,
  onReport,
  onPromote,
  onReject,
}: {
  candidates: CNNCandidate[]
  onReport: (id: string) => void
  onPromote: (id: string) => void
  onReject: (id: string) => void
}) {
  return (
    <Table
      size="small"
      rowKey="candidate_id"
      dataSource={candidates}
      columns={[
        { title: '候选', dataIndex: 'candidate_id', width: 180 },
        { title: '模型', dataIndex: 'model_name' },
        { title: '状态', dataIndex: 'status', width: 100, render: (v) => <Tag color={statusColor[String(v)]}>{String(v)}</Tag> },
        { title: '胜率', render: (_, row) => String(row.summary?.candidate_win_rate ?? '-') },
        {
          title: '操作',
          width: 240,
          render: (_, row) => (
            <Space>
              <Button size="small" onClick={() => onReport(row.report_id)}>报告</Button>
              <Button size="small" type="primary" disabled={row.status === 'promoted'} onClick={() => onPromote(row.candidate_id)}>晋级</Button>
              <Button size="small" disabled={row.status === 'rejected'} onClick={() => onReject(row.candidate_id)}>拒绝</Button>
            </Space>
          ),
        },
      ]}
    />
  )
}

function ReportPanel({ report }: { report?: CNNGovernanceReport }) {
  if (!report) return <Alert style={{ marginTop: 12 }} type="info" message="选择候选报告或等待任务完成后查看 WF/OOS 摘要。" />
  return (
    <section className="panel" style={{ marginTop: 12 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Text strong>WF/OOS 报告：{report.report_id}</Text>
        <Descriptions size="small" bordered>
          <Descriptions.Item label="折数">{String(report.summary?.fold_count ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="胜出折数">{String(report.summary?.candidate_win_count ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="胜率">{String(report.summary?.candidate_win_rate ?? '-')}</Descriptions.Item>
          <Descriptions.Item label="结论">{report.summary?.passed ? <Tag color="green">通过</Tag> : <Tag color="red">未通过</Tag>}</Descriptions.Item>
        </Descriptions>
        <Table
          size="small"
          rowKey={(row) => String(row.fold)}
          dataSource={report.folds || []}
          columns={[
            { title: '折', dataIndex: 'fold', width: 60 },
            { title: '候选模型', dataIndex: 'candidate_model' },
            { title: '候选分数', dataIndex: 'candidate_score', width: 120 },
            { title: '生产分数', dataIndex: 'production_score', width: 120 },
            { title: '差值', dataIndex: 'score_delta', width: 100 },
          ]}
        />
      </Space>
    </section>
  )
}

function ReplayTable({ replays, onOpen }: { replays: Array<Record<string, any>>; onOpen: (id: string) => void }) {
  return (
    <Table
      size="small"
      rowKey="replay_id"
      dataSource={replays}
      columns={[
        { title: '回放', dataIndex: 'replay_id', width: 220 },
        { title: '名称', dataIndex: 'name' },
        { title: '标的', dataIndex: 'target_symbol', width: 130 },
        { title: '结论', render: (_, row) => String(row.conclusion?.verdict || '-') },
        { title: '操作', width: 90, render: (_, row) => <Button size="small" onClick={() => onOpen(String(row.replay_id))}>查看</Button> },
      ]}
    />
  )
}

function ReplayPanel({ replay }: { replay?: any }) {
  if (!replay) return <Alert style={{ marginTop: 12 }} type="info" message="启动或选择治理回放后查看四组对比结果。" />
  const rows = Object.entries(replay.baselines || {}).map(([key, value]: [string, any]) => ({
    key,
    ...(value.statistics || value.statistics === undefined ? value.statistics : {}),
  }))
  return (
    <section className="panel" style={{ marginTop: 12 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Text strong>治理回放：{replay.replay_id}</Text>
        <Alert type={replay.conclusion?.recommend_enable_promotion ? 'success' : 'warning'} message={replay.conclusion?.verdict || '无结论'} />
        <Table
          size="small"
          rowKey="key"
          dataSource={rows}
          columns={[
            { title: '策略', dataIndex: 'key' },
            { title: '总收益', dataIndex: 'total_return' },
            { title: 'Sharpe', dataIndex: 'sharpe_ratio' },
            { title: '最大回撤', dataIndex: 'max_ddpercent' },
            { title: '成交次数', dataIndex: 'total_trade_count' },
            { title: '成本', dataIndex: 'total_cost' },
          ]}
        />
        <Table<Record<string, unknown>>
          size="small"
          rowKey={(row) => `${String(row.date)}-${String(row.new_model || row.candidate_model || '')}`}
          dataSource={replay.promotion_events || []}
          columns={[
            { title: '日期', dataIndex: 'date', width: 130 },
            { title: '旧模型', dataIndex: 'old_model' },
            { title: '新模型', dataIndex: 'new_model' },
            { title: '原因', dataIndex: 'reason' },
          ]}
        />
      </Space>
    </section>
  )
}

export default CNNGovernance

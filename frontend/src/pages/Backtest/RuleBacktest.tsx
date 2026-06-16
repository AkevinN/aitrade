// 规则策略回测页面 — 以 AlphaBacktest.tsx 为骨架模板
import React, { useState, useCallback, useMemo } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, DatePicker,
  InputNumber, Select, Progress, message, Input, Alert,
  Collapse, Table,
} from 'antd'
import {
  PlayCircleOutlined, LineChartOutlined, ThunderboltOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { strategyService } from '../../api/strategy'
import { useTask } from '../../hooks/useTask'
import BacktestResults from './BacktestResults'
import BacktestCharts from './BacktestCharts'
import type { BacktestResultPayload, BacktestStatistics } from '../../types/alpha'
import type { SweepRow, StrategySweepRequest, StrategyWalkForwardRequest } from '../../types/strategy'

const { Text } = Typography
const { RangePicker } = DatePicker
const { TextArea } = Input

// 默认 ETF 标的（etf_momentum 策略默认 universe）
const DEFAULT_UNIVERSE = [
  '510300.SSE',
  '510500.SSE',
  '159915.SZE',
  '512010.SSE',
  '513100.SSE',
].join('\n')

/**
 * 规则策略回测页面。
 *
 * 提供三种评估模式：
 * - **单次回测**：选定信号源 + 参数，运行一次完整回测并展示统计指标与净值曲线。
 * - **参数扫描（Sweep）**：网格搜索多组参数，对比各组合的夏普/回撤/收益。
 * - **Walk-Forward 验证**：滚动窗口评估参数时序稳定性。
 *
 * 所有任务均为异步，通过 `useTask` 轮询至终态后渲染结果。
 */
const RuleBacktest: React.FC = () => {
  // 信号源与参数
  const [selectedSource, setSelectedSource] = useState('')
  const [universeText, setUniverseText] = useState(DEFAULT_UNIVERSE)
  const [signalParamValues, setSignalParamValues] = useState<Record<string, unknown>>({})

  // 策略参数（固定字段）
  const [topK, setTopK] = useState(3)
  const [nDrop, setNDrop] = useState(1)
  const [minDays, setMinDays] = useState(5)
  const [rebalanceFreq, setRebalanceFreq] = useState('W')

  // 成本假设
  const [commissionRate, setCommissionRate] = useState(0.0003)
  const [stampDuty, setStampDuty] = useState(0.0005)
  const [slippage, setSlippage] = useState(0.0005)
  const [tPlus1, setTPlus1] = useState(true)

  // 日期 & 资金
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(2, 'year'),
    dayjs(),
  ])
  const [capital, setCapital] = useState(1000000)
  const [interval, setInterval] = useState('d')

  // 回测任务
  const [taskId, setTaskId] = useState<string | null>(null)
  const task = useTask(taskId)
  const statistics = useMemo(
    () => (task.data?.result as { statistics?: BacktestStatistics } | undefined)?.statistics,
    [task.data?.result]
  )
  const result = useMemo(
    () => task.data?.result as BacktestResultPayload | undefined,
    [task.data?.result]
  )

  // Sweep 状态
  const [sweepTaskId, setSweepTaskId] = useState<string | null>(null)
  const sweepTask = useTask(sweepTaskId)
  const sweepRows = useMemo(
    () => (sweepTask.data?.result as { rows?: SweepRow[] } | undefined)?.rows ?? [],
    [sweepTask.data?.result]
  )
  const [sweepGridText, setSweepGridText] = useState(
    '{"strategy_params": {"top_k": 3}}\n{"strategy_params": {"top_k": 5}}'
  )
  const [sweepGridError, setSweepGridError] = useState<string | null>(null)

  // Walk-Forward 状态
  const [wfTaskId, setWfTaskId] = useState<string | null>(null)
  const wfTask = useTask(wfTaskId)
  const wfRows = useMemo(
    () => (wfTask.data?.result as { rows?: SweepRow[] } | undefined)?.rows ?? [],
    [wfTask.data?.result]
  )
  const [trainDays, setTrainDays] = useState(180)
  const [testDays, setTestDays] = useState(60)

  // 信号源列表
  const { data: sources } = useQuery({
    queryKey: ['strategy-sources'],
    queryFn: () => strategyService.listSources(),
  })

  // 选中的信号源 spec
  const selectedSourceInfo = useMemo(
    () => sources?.find((s) => s.name === selectedSource),
    [sources, selectedSource]
  )

  /**
   * 构建通用回测请求体（由三个操作共用：单次回测、Sweep、Walk-Forward）。
   *
   * universe 文本框按行分割并过滤空行，写入 signal_params.universe。
   *
   * @returns 完整的 `StrategyBacktestRequest` 对象（不含 grid / train_days / test_days）
   */
  const buildRequest = useCallback(() => {
    const universe = universeText
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
    return {
      signal_source: selectedSource,
      signal_params: {
        ...signalParamValues,
        universe,
      },
      strategy_params: {
        top_k: topK,
        n_drop: nDrop,
        min_days: minDays,
        rebalance_freq: rebalanceFreq,
      },
      interval,
      start: dateRange[0].format('YYYY-MM-DD'),
      end: dateRange[1].format('YYYY-MM-DD'),
      capital,
      cost: {
        commission_rate: commissionRate,
        stamp_duty: stampDuty,
        slippage,
        t_plus1: tPlus1,
      },
    }
  }, [
    universeText, selectedSource, signalParamValues,
    topK, nDrop, minDays, rebalanceFreq, interval,
    dateRange, capital, commissionRate, stampDuty, slippage, tPlus1,
  ])

  /**
   * 启动单次规则策略回测。
   *
   * 校验信号源已选，调用 `strategyService.runBacktest`，存储返回的 task_id 供轮询。
   */
  const handleRun = useCallback(async () => {
    if (!selectedSource) {
      message.warning('请选择一个信号源')
      return
    }
    try {
      const res = await strategyService.runBacktest(buildRequest())
      setTaskId(res.task_id)
      message.success('回测任务已启动')
    } catch {
      message.error('启动回测失败')
    }
  }, [selectedSource, buildRequest])

  /**
   * 启动参数网格扫描任务。
   *
   * 逐行解析 sweepGridText 中的 JSON 对象，构成 grid 列表后调用 `strategyService.runSweep`。
   * 任一行解析失败则设置 sweepGridError 并提前返回。
   */
  const handleSweep = useCallback(async () => {
    if (!selectedSource) {
      message.warning('请选择一个信号源')
      return
    }
    // 解析网格 JSON
    const lines = sweepGridText.split('\n').map((l) => l.trim()).filter(Boolean)
    const grid: StrategySweepRequest['grid'] = []
    for (const line of lines) {
      try {
        grid.push(JSON.parse(line))
      } catch {
        setSweepGridError(`JSON 解析失败：${line}`)
        return
      }
    }
    setSweepGridError(null)
    try {
      const req: StrategySweepRequest = { ...buildRequest(), grid }
      const res = await strategyService.runSweep(req)
      setSweepTaskId(res.task_id)
      message.success('参数扫描任务已启动')
    } catch {
      message.error('启动参数扫描失败')
    }
  }, [selectedSource, sweepGridText, buildRequest])

  /**
   * 启动 Walk-Forward 验证任务。
   *
   * 在 buildRequest() 基础上追加 train_days / test_days，调用 `strategyService.runWalkForward`。
   */
  const handleWalkForward = useCallback(async () => {
    if (!selectedSource) {
      message.warning('请选择一个信号源')
      return
    }
    try {
      const req: StrategyWalkForwardRequest = {
        ...buildRequest(),
        train_days: trainDays,
        test_days: testDays,
      }
      const res = await strategyService.runWalkForward(req)
      setWfTaskId(res.task_id)
      message.success('Walk-Forward 任务已启动')
    } catch {
      message.error('启动 Walk-Forward 失败')
    }
  }, [selectedSource, trainDays, testDays, buildRequest])

  /**
   * 动态渲染信号参数表单项（依据选中信号源的 param_spec）。
   *
   * 支持四种类型：int/float（InputNumber）、list[str]（TextArea 逐行）、str（Input）。
   * `universe` 字段由专用 TextArea 处理，此处跳过不渲染。
   *
   * @returns 参数表单 JSX 或 null（无 param_spec 时）
   */
  const renderDynamicParams = () => {
    if (!selectedSourceInfo?.param_spec) return null
    const spec = selectedSourceInfo.param_spec
    return (
      <div>
        <Text type="secondary">信号参数：</Text>
        <Space direction="vertical" style={{ width: '100%', marginTop: 8 }}>
          {Object.entries(spec).map(([key, meta]) => {
            // universe 由专用 TextArea 处理，跳过动态渲染
            if (key === 'universe') return null
            const label = meta.label ?? key
            const type = meta.type ?? 'str'
            const defaultVal = meta.default
            if (type === 'int' || type === 'float') {
              return (
                <div key={key}>
                  <Text style={{ fontSize: 12 }}>{label}</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    defaultValue={defaultVal as number | undefined}
                    step={type === 'int' ? 1 : 0.01}
                    onChange={(v) =>
                      setSignalParamValues((prev) => ({ ...prev, [key]: v }))
                    }
                  />
                </div>
              )
            }
            if (type === 'list[str]') {
              return (
                <div key={key}>
                  <Text style={{ fontSize: 12 }}>{label}（每行一个）</Text>
                  <TextArea
                    rows={3}
                    defaultValue={
                      Array.isArray(defaultVal) ? (defaultVal as string[]).join('\n') : ''
                    }
                    onChange={(e) =>
                      setSignalParamValues((prev) => ({
                        ...prev,
                        [key]: e.target.value.split('\n').map((s) => s.trim()).filter(Boolean),
                      }))
                    }
                  />
                </div>
              )
            }
            // default: str
            return (
              <div key={key}>
                <Text style={{ fontSize: 12 }}>{label}</Text>
                <Input
                  defaultValue={defaultVal as string | undefined}
                  onChange={(e) =>
                    setSignalParamValues((prev) => ({ ...prev, [key]: e.target.value }))
                  }
                />
              </div>
            )
          })}
        </Space>
      </div>
    )
  }

  /**
   * Sweep 与 Walk-Forward 结果表共用的列定义。
   *
   * 每行对应一组参数的回测指标：total_return / max_ddpercent 为小数比率，
   * 渲染时乘 100 显示为百分比；sharpe_ratio 保留 3 位小数。
   */
  const sweepColumns = [
    {
      title: '参数',
      dataIndex: 'params',
      key: 'params',
      render: (v: Record<string, unknown>) => JSON.stringify(v),
    },
    {
      title: '总收益',
      dataIndex: 'total_return',
      key: 'total_return',
      render: (v: number) => `${(v * 100).toFixed(2)}%`,
    },
    {
      title: '夏普',
      dataIndex: 'sharpe_ratio',
      key: 'sharpe_ratio',
      render: (v: number) => v?.toFixed(3),
    },
    {
      title: '最大回撤',
      dataIndex: 'max_ddpercent',
      key: 'max_ddpercent',
      render: (v: number) => `${(v * 100).toFixed(2)}%`,
    },
    {
      title: '成交数',
      dataIndex: 'trade_count',
      key: 'trade_count',
    },
  ]

  return (
    <Row gutter={[16, 16]}>
      {/* 左侧：配置区 */}
      <Col xs={24} lg={8}>
        <Card title={<><LineChartOutlined /> 规则策略回测配置</>} size="small">
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {/* 信号源 */}
            <div>
              <Text type="secondary">信号源：</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={selectedSource || undefined}
                onChange={setSelectedSource}
                placeholder="选择信号源"
                options={(sources ?? []).map((s) => ({
                  label: `${s.name} — ${s.description}`,
                  value: s.name,
                }))}
              />
            </div>

            {/* 动态信号参数 */}
            {renderDynamicParams()}

            {/* Universe */}
            <div>
              <Text type="secondary">标的池（每行一个 vt_symbol）：</Text>
              <TextArea
                style={{ marginTop: 8 }}
                rows={5}
                value={universeText}
                onChange={(e) => setUniverseText(e.target.value)}
                placeholder="510300.SSE"
              />
            </div>

            {/* 策略参数（固定字段） */}
            <div>
              <Text type="secondary">策略参数：</Text>
              <Row gutter={8} style={{ marginTop: 8 }}>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>持仓数 top_k</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={topK}
                    onChange={(v) => setTopK(v ?? 3)}
                    min={1}
                    max={20}
                  />
                </Col>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>换仓淘汰 n_drop</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={nDrop}
                    onChange={(v) => setNDrop(v ?? 1)}
                    min={0}
                  />
                </Col>
                <Col span={12} style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 12 }}>最短持仓天数</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={minDays}
                    onChange={(v) => setMinDays(v ?? 5)}
                    min={0}
                  />
                </Col>
                <Col span={12} style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 12 }}>再平衡频率</Text>
                  <Select
                    style={{ width: '100%' }}
                    value={rebalanceFreq}
                    onChange={setRebalanceFreq}
                    options={[
                      { label: '日', value: 'D' },
                      { label: '周', value: 'W' },
                      { label: '月', value: 'M' },
                    ]}
                  />
                </Col>
              </Row>
            </div>

            {/* 成本假设（从 CNNBacktest.tsx 移植） */}
            <div>
              <Text type="secondary">成本假设（A股默认，按需调整）：</Text>
              <Row gutter={8} style={{ marginTop: 8 }}>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>佣金率</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={commissionRate}
                    onChange={(v) => setCommissionRate(v ?? 0.0003)}
                    min={0}
                    max={0.01}
                    step={0.0001}
                    addonAfter="率"
                  />
                </Col>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>卖出印花税</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={stampDuty}
                    onChange={(v) => setStampDuty(v ?? 0.0005)}
                    min={0}
                    max={0.01}
                    step={0.0005}
                    addonAfter="率"
                  />
                </Col>
                <Col span={12} style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 12 }}>成交滑点</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={slippage}
                    onChange={(v) => setSlippage(v ?? 0.0005)}
                    min={0}
                    max={0.05}
                    step={0.0005}
                    addonAfter="率"
                  />
                </Col>
                <Col span={12} style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 12 }}>T+1 限制</Text>
                  <Select
                    style={{ width: '100%' }}
                    value={tPlus1 ? 'yes' : 'no'}
                    onChange={(v) => setTPlus1(v === 'yes')}
                    options={[
                      { label: '启用', value: 'yes' },
                      { label: '禁用', value: 'no' },
                    ]}
                  />
                </Col>
              </Row>
            </div>

            {/* 日期范围 & 资金 */}
            <div>
              <Text type="secondary">日期范围：</Text>
              <RangePicker
                style={{ width: '100%', marginTop: 8 }}
                value={dateRange}
                onChange={(dates) =>
                  dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])
                }
              />
            </div>
            <div>
              <Text type="secondary">初始资金：</Text>
              <InputNumber
                style={{ width: '100%', marginTop: 8 }}
                value={capital}
                onChange={(v) => setCapital(v || 1000000)}
                min={10000}
                step={100000}
                formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                parser={(value) => value?.replace(/\$\s?|(,*)/g, '') as unknown as number}
              />
            </div>
            <div>
              <Text type="secondary">回测周期：</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={interval}
                onChange={setInterval}
                options={[{ label: '日线', value: 'd' }]}
              />
            </div>

            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleRun}
              loading={task?.data?.status === 'running'}
              block
            >
              启动回测
            </Button>

            {task?.data && (
              <Card size="small">
                <Progress
                  percent={Math.round(task.data.progress)}
                  status={
                    task.data.status === 'failed'
                      ? 'exception'
                      : task.data.status === 'completed'
                      ? 'success'
                      : 'active'
                  }
                />
                <Text type="secondary">{task.data.message || task.data.status}</Text>
              </Card>
            )}

            <Alert
              type="info"
              showIcon
              message="回测结果基于本地已下载行情；信号与撮合使用同一数据仓库"
            />
          </Space>
        </Card>

        {/* Sweep & Walk-Forward 折叠面板 */}
        <Collapse style={{ marginTop: 16 }}>
          <Collapse.Panel header="参数扫描 / Walk-Forward" key="sweep">
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <div>
                <Text type="secondary">网格（每行一个 JSON 对象）：</Text>
                <TextArea
                  style={{ marginTop: 8, fontFamily: 'monospace' }}
                  rows={5}
                  value={sweepGridText}
                  onChange={(e) => {
                    setSweepGridText(e.target.value)
                    setSweepGridError(null)
                  }}
                  placeholder={'{"strategy_params": {"top_k": 3}}'}
                />
                {sweepGridError && (
                  <Alert
                    type="error"
                    message={sweepGridError}
                    style={{ marginTop: 8 }}
                    showIcon
                  />
                )}
              </div>
              <Button
                icon={<ThunderboltOutlined />}
                onClick={handleSweep}
                loading={sweepTask?.data?.status === 'running'}
                block
              >
                启动参数扫描
              </Button>
              {sweepRows.length > 0 && (
                <Table
                  size="small"
                  dataSource={sweepRows.map((r, i) => ({ ...r, key: i }))}
                  columns={sweepColumns}
                  pagination={false}
                  scroll={{ x: true }}
                />
              )}

              <Row gutter={8}>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>训练窗(天)</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={trainDays}
                    onChange={(v) => setTrainDays(v ?? 180)}
                    min={30}
                  />
                </Col>
                <Col span={12}>
                  <Text style={{ fontSize: 12 }}>测试窗(天)</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={testDays}
                    onChange={(v) => setTestDays(v ?? 60)}
                    min={10}
                  />
                </Col>
              </Row>
              <Button
                icon={<ThunderboltOutlined />}
                onClick={handleWalkForward}
                loading={wfTask?.data?.status === 'running'}
                block
              >
                启动 Walk-Forward
              </Button>
              {wfRows.length > 0 && (
                <Table
                  size="small"
                  dataSource={wfRows.map((r, i) => ({ ...r, key: i }))}
                  columns={sweepColumns}
                  pagination={false}
                  scroll={{ x: true }}
                />
              )}
            </Space>
          </Collapse.Panel>
        </Collapse>
      </Col>

      {/* 右侧：结果区 */}
      <Col xs={24} lg={16}>
        {task.data?.status === 'failed' && (
          <Alert
            type="error"
            message="回测失败"
            description={task.data.message}
            style={{ marginBottom: 16 }}
          />
        )}
        <BacktestResults statistics={statistics} capital={capital} />
        {task.data?.status === 'completed' && result && (
          <div style={{ marginTop: 16 }}>
            <BacktestCharts
              result={result}
              interval={interval}
              start={dateRange[0].format('YYYY-MM-DD')}
              end={dateRange[1].format('YYYY-MM-DD')}
            />
          </div>
        )}
      </Col>
    </Row>
  )
}

export default RuleBacktest

import React, { useState, useCallback, useMemo, useEffect } from 'react'
import {
  Card, Row, Col, Typography, Space, Button, DatePicker,
  InputNumber, Select, Progress, message, Slider, Input, Alert, Switch,
} from 'antd'
import {
  PlayCircleOutlined, ExperimentOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import dayjs from 'dayjs'

import { cnnService } from '../../api/cnn'
import { useTask } from '../../hooks/useTask'
import BacktestResults from './BacktestResults'
import BacktestCharts from './BacktestCharts'
import type { BacktestResultPayload, BacktestStatistics } from '../../types/alpha'

const { Text } = Typography
const { RangePicker } = DatePicker

/**
 * CNN 模型回测页面。
 *
 * 支持三类模型：
 * - `classification`（方向二分类）：阈值为概率口径（0~1）；
 * - `regression`（收益回归）：阈值为收益率口径（如 ±0.5%）；
 * - `path_class`（路径形态四分类）：买入阈值为先触止盈概率（prob_tp），
 *   并额外支持 veto_threshold（否决阈值，prob_sl ≥ 该值则放弃买入）。
 *
 * 出场模式：threshold（概率阈值）/ fixed_hold（固定持有）/ oco（止盈止损）/ auto（按 label 推导）。
 * 回测结果展示统计指标、成交明细与净值曲线，含 label↔策略一致性自检提示。
 */
const CNNBacktest: React.FC = () => {
  const [backtestName, setBacktestName] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [capital, setCapital] = useState(1000000)
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([
    dayjs().subtract(1, 'year'),
    dayjs(),
  ])
  const [buyThreshold, setBuyThreshold] = useState(0.6)
  const [sellThreshold, setSellThreshold] = useState(0.4)
  // 出场模式：threshold=概率阈值；fixed_hold=固定持有；oco=止盈止损；auto=按 label 自动对齐
  const [exitMode, setExitMode] = useState<'threshold' | 'fixed_hold' | 'oco' | 'auto'>('threshold')
  const [holdDays, setHoldDays] = useState(1)
  const [takeProfit, setTakeProfit] = useState(0.02)
  const [stopLoss, setStopLoss] = useState(0.03)
  const [tPlus1, setTPlus1] = useState(false)
  // 成本假设（A股默认：佣金万3、卖出印花税0.5‰、滑点5bp、限价缓冲20bp）
  const [commissionRate, setCommissionRate] = useState(0.0003)
  const [stampDuty, setStampDuty] = useState(0.0005)
  const [slippage, setSlippage] = useState(0.0005)
  const [priceAdd, setPriceAdd] = useState(0.002)
  /**
   * 否决阈值（仅 path_class 模型有效）：prob_sl ≥ 该值时放弃买入；1.0=关闭。
   * 默认 1.0，选中 path_class 模型时重置到该值。
   */
  const [vetoThreshold, setVetoThreshold] = useState(1.0)
  const [taskId, setTaskId] = useState<string | null>(null)

  const task = useTask(taskId)
  const statistics = useMemo(
    () => (task.data?.result as { statistics?: BacktestStatistics } | undefined)?.statistics,
    [task.data?.result]
  )
  // 完整回测结果载荷（含成交明细与逐日净值），供图表容器消费
  const result = useMemo(
    () => task.data?.result as BacktestResultPayload | undefined,
    [task.data?.result]
  )

  const { data: models } = useQuery({
    queryKey: ['cnn-models'],
    queryFn: () => cnnService.listModels(),
  })

  const selectedModelInfo = useMemo(
    () => models?.find((m) => m.name === selectedModel),
    [models, selectedModel]
  )
  // K 线周期取「产出该结果的模型」的 input_interval：优先按结果回传的 model 名查找，
  // 避免结果展示后用户改选模型导致的错配；缺省回退到当前选中模型。
  const chartInterval = useMemo(() => {
    const modelName = result?.model ?? selectedModel
    return models?.find((m) => m.name === modelName)?.input_interval ?? ''
  }, [models, result?.model, selectedModel])
  // 回归模型的 signal 是预测收益，阈值用收益尺度；分类/path_class 模型用概率尺度
  const isRegression = selectedModelInfo?.objective === 'regression'
  /** path_class 模型：输出四类剧本概率，买入阈值为 prob_tp，支持 veto_threshold 否决。 */
  const isPathClass = selectedModelInfo?.objective === 'path_class'

  // 切换模型时，把阈值重置为对应口径的合理默认值（回归: ±0.5%；分类/path_class: 0.6/0.4；path_class 重置 veto）
  useEffect(() => {
    if (isRegression) {
      setBuyThreshold(0.005)
      setSellThreshold(-0.005)
    } else {
      setBuyThreshold(0.6)
      setSellThreshold(0.4)
    }
    if (isPathClass) {
      setVetoThreshold(1.0)
    }
  }, [isRegression, isPathClass])

  /**
   * 启动 CNN 回测任务。
   *
   * 校验模型已选且 threshold 模式下买入阈值 > 卖出阈值，
   * 回测名称留空则自动生成（cnn_bt_{model}_{timestamp}）。
   * path_class 模型时额外传 veto_threshold（1.0=关闭）。
   */
  const handleRun = useCallback(async () => {
    if (!selectedModel) {
      message.warning('请选择一个 CNN 模型')
      return
    }
    if (exitMode === 'threshold' && buyThreshold <= sellThreshold) {
      message.warning('买入阈值必须大于卖出阈值')
      return
    }
    const name = backtestName || `cnn_bt_${selectedModel}_${Date.now()}`
    try {
      const res = await cnnService.runBacktest({
        name,
        model: selectedModel,
        capital,
        start: dateRange[0].format('YYYY-MM-DD'),
        end: dateRange[1].format('YYYY-MM-DD'),
        buy_threshold: buyThreshold,
        sell_threshold: sellThreshold,
        commission_rate: commissionRate,
        stamp_duty: stampDuty,
        slippage,
        price_add: priceAdd,
        exit_mode: exitMode,
        hold_days: holdDays,
        take_profit: takeProfit,
        stop_loss: stopLoss,
        t_plus1: tPlus1,
        ...(isPathClass ? { veto_threshold: vetoThreshold } : {}),
      })
      setTaskId(res.task_id)
      message.success('CNN 回测任务已启动')
    } catch {
      message.error('启动 CNN 回测失败')
    }
  }, [backtestName, buyThreshold, capital, commissionRate, dateRange, priceAdd, selectedModel, sellThreshold, slippage, stampDuty, exitMode, holdDays, takeProfit, stopLoss, tPlus1, isPathClass, vetoThreshold])

  return (
    <Row gutter={[16, 16]}>
      {/* Left: Config */}
      <Col xs={24} lg={8}>
        <Card title={<><ExperimentOutlined /> CNN 回测配置</>} size="small">
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <Text type="secondary">回测名称：</Text>
              <Input
                style={{ width: '100%', marginTop: 8 }}
                value={backtestName}
                onChange={(e) => setBacktestName(e.target.value)}
                placeholder="留空则自动生成"
              />
            </div>
            <div>
              <Text type="secondary">CNN 模型：</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={selectedModel || undefined}
                onChange={setSelectedModel}
                placeholder="选择已训练的模型"
                options={(models || []).map((m) => ({
                  label: `${m.name}${m.target_symbol ? ` (${m.target_symbol})` : ''}`,
                  value: m.name,
                }))}
              />
              {selectedModelInfo && (
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                  目标: {selectedModelInfo.target_symbol || '未知'} ·{' '}
                  类型: {isRegression ? '收益回归' : isPathClass ? '路径形态分类' : '方向分类'} ·{' '}
                  最佳 epoch: {selectedModelInfo.best_epoch || '-'} ·{' '}
                  验证损失: {selectedModelInfo.best_val_loss?.toFixed(4) || '-'}
                </Text>
              )}
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
              <Text type="secondary">日期范围：</Text>
              <RangePicker
                style={{ width: '100%', marginTop: 8 }}
                value={dateRange}
                onChange={(dates) => dates && setDateRange(dates as [dayjs.Dayjs, dayjs.Dayjs])}
              />
            </div>
            {isRegression ? (
              <>
                <div>
                  <Text type="secondary">买入阈值（预测收益 &gt; 此值时买入）：</Text>
                  <Slider
                    min={0}
                    max={0.03}
                    step={0.001}
                    value={buyThreshold}
                    onChange={setBuyThreshold}
                    tooltip={{ formatter: (v) => `${((v ?? 0) * 100).toFixed(1)}%` }}
                    marks={{ 0: '0', 0.005: '0.5%', 0.01: '1%', 0.02: '2%', 0.03: '3%' }}
                  />
                </div>
                <div>
                  <Text type="secondary">卖出阈值（预测收益 &lt; 此值时卖出）：</Text>
                  <Slider
                    min={-0.03}
                    max={0}
                    step={0.001}
                    value={sellThreshold}
                    onChange={setSellThreshold}
                    tooltip={{ formatter: (v) => `${((v ?? 0) * 100).toFixed(1)}%` }}
                    marks={{ '-0.03': '-3%', '-0.02': '-2%', '-0.01': '-1%', '-0.005': '-0.5%', 0: '0' }}
                  />
                </div>
              </>
            ) : (
              <>
                <div>
                  <Text type="secondary">
                    {isPathClass
                      ? '先触止盈概率阈值（prob_tp &gt; 此值时买入）：'
                      : '买入阈值（概率 &gt; 此值时买入）：'}
                  </Text>
                  <Slider
                    min={0.5}
                    max={0.95}
                    step={0.05}
                    value={buyThreshold}
                    onChange={setBuyThreshold}
                    marks={{ 0.5: '0.5', 0.6: '0.6', 0.7: '0.7', 0.8: '0.8', 0.95: '0.95' }}
                  />
                </div>
                <div>
                  <Text type="secondary">
                    {isPathClass
                      ? '先触止盈概率卖出阈值（prob_tp &lt; 此值时卖出）：'
                      : '卖出阈值（概率 &lt; 此值时卖出）：'}
                  </Text>
                  <Slider
                    min={0.1}
                    max={0.5}
                    step={0.05}
                    value={sellThreshold}
                    onChange={setSellThreshold}
                    marks={{ 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.4: '0.4', 0.5: '0.5' }}
                  />
                </div>
                {isPathClass && (
                  <div>
                    <Text type="secondary">
                      否决阈值 veto_threshold（先触止损概率 prob_sl ≥ 此值则放弃买入；1.0=关闭）：
                    </Text>
                    <Slider
                      min={0.1}
                      max={1.0}
                      step={0.05}
                      value={vetoThreshold}
                      onChange={setVetoThreshold}
                      marks={{ 0.1: '0.1', 0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 1.0: '关闭' }}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      设为 1.0 关闭否决（等同不过滤）；建议先在 0.4~0.6 区间探索。
                    </Text>
                  </div>
                )}
              </>
            )}
            <div>
              <Text type="secondary">出场模式：</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                value={exitMode}
                onChange={setExitMode}
                options={[
                  { label: '概率阈值（概率跌破卖出阈值才平仓）', value: 'threshold' },
                  { label: '固定持有（持有 N 个交易日后平仓）', value: 'fixed_hold' },
                  { label: '止盈止损 OCO', value: 'oco' },
                  { label: '自动对齐 label（推荐，按模型 label 推导固定持有）', value: 'auto' },
                ]}
              />
              {exitMode === 'auto' && (
                <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 4 }}>
                  将按模型训练 label 自动推导固定持有期，并做 label↔策略一致性自检。
                </Text>
              )}
              {(exitMode === 'fixed_hold' || exitMode === 'oco') && (
                <div style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 12 }}>
                    {exitMode === 'oco' ? '最大持有交易日（回退，0=不限）' : '固定持有交易日'}
                  </Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={holdDays}
                    onChange={(v) => setHoldDays(v ?? 1)}
                    min={exitMode === 'oco' ? 0 : 1}
                    max={60}
                    step={1}
                  />
                </div>
              )}
              {exitMode === 'oco' && (
                <Row gutter={8} style={{ marginTop: 8 }}>
                  <Col span={12}>
                    <Text style={{ fontSize: 12 }}>止盈幅度（0=不启用）</Text>
                    <InputNumber
                      style={{ width: '100%' }}
                      value={takeProfit}
                      onChange={(v) => setTakeProfit(v ?? 0)}
                      min={0}
                      max={0.5}
                      step={0.005}
                      addonAfter="率"
                    />
                  </Col>
                  <Col span={12}>
                    <Text style={{ fontSize: 12 }}>止损幅度（0=不启用）</Text>
                    <InputNumber
                      style={{ width: '100%' }}
                      value={stopLoss}
                      onChange={(v) => setStopLoss(v ?? 0)}
                      min={0}
                      max={0.5}
                      step={0.005}
                      addonAfter="率"
                    />
                  </Col>
                </Row>
              )}
              <div style={{ marginTop: 8 }}>
                <Space>
                  <Switch checked={tPlus1} onChange={setTPlus1} size="small" />
                  <Text style={{ fontSize: 12 }}>T+1 卖出限制（当日买入不可当日卖出）</Text>
                </Space>
              </div>
            </div>
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
                  <Text style={{ fontSize: 12 }}>限价缓冲</Text>
                  <InputNumber
                    style={{ width: '100%' }}
                    value={priceAdd}
                    onChange={(v) => setPriceAdd(v ?? 0.002)}
                    min={0}
                    max={0.05}
                    step={0.001}
                    addonAfter="率"
                  />
                </Col>
              </Row>
            </div>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleRun}
              loading={task?.data?.status === 'running'}
              block
            >
              启动 CNN 回测
            </Button>
            {task?.data && (
              <Card size="small">
                <Progress
                  percent={Math.round(task.data.progress)}
                  status={task.data.status === 'failed' ? 'exception' : task.data.status === 'completed' ? 'success' : 'active'}
                />
                <Text type="secondary">{task.data.message || task.data.status}</Text>
              </Card>
            )}
          </Space>
        </Card>
      </Col>

      <Col xs={24} lg={16}>
        {task.data?.status === 'failed' && (
          <Alert
            type="error"
            message="CNN 回测失败"
            description={task.data.message}
            style={{ marginBottom: 16 }}
          />
        )}
        {statistics?.consistency_warnings && statistics.consistency_warnings.length > 0 && (
          <Alert
            type="warning"
            showIcon
            message="label ↔ 策略出场 一致性提示"
            description={
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {statistics.consistency_warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            }
            style={{ marginBottom: 16 }}
          />
        )}
        <BacktestResults statistics={statistics} capital={capital} />
        {/* 回测完成后在统计卡片下方展示净值曲线与买卖点 K 线；不改 BacktestResults 既有展示 */}
        {task.data?.status === 'completed' && result && (
          <div style={{ marginTop: 16 }}>
            <BacktestCharts
              result={result}
              interval={chartInterval}
              start={dateRange[0].format('YYYY-MM-DD')}
              end={dateRange[1].format('YYYY-MM-DD')}
            />
          </div>
        )}
      </Col>
    </Row>
  )
}

export default CNNBacktest
